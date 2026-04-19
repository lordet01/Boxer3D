"""Convert the BoxerNet / YOLO ONNX models to FP16 to halve on-device memory.

The original BoxerNet ONNX is ~391 MB float32 and is the main reason iOS Jetsam
kills the app. Converting to float16 typically:
  * cuts the .onnx file size roughly in half (~195 MB);
  * cuts ORT/CoreML EP runtime memory for the weights by ~50%;
  * keeps the same input/output names and shapes (drop-in for the Swift code).

Usage:
    python -m venv .venv && source .venv/bin/activate
    pip install -r scripts/requirements.txt
    python scripts/quantize_models.py \\
        --input  boxer/BoxerNet.onnx \\
        --output boxer/BoxerNet.fp16.onnx
    python scripts/quantize_models.py \\
        --input  boxer/yolo11n.onnx \\
        --output boxer/yolo11n.fp16.onnx

Then in Xcode replace `BoxerNet.onnx` / `yolo11n.onnx` with the *.fp16.onnx
files in the bundle (or rename them back to the original names).

Notes
-----
* ORT will up-cast FP16 inputs to whatever the model expects internally,
  so the Swift float32 input tensors keep working without any code change.
* If accuracy is unacceptable, re-run with `--keep-io-types` to keep IO float32
  but still store weights in float16 (already the default here).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable


# Ops that often lose accuracy or fail on CoreML EP when converted to FP16.
# Keep them in FP32 by default — the size win still comes from large matmul/conv weights.
#
# 이 리스트는 두 가지 역할을 한다:
#  1) `--ort` 경로: onnxruntime.transformers.float16 의 기본 block list에 *추가*해서
#     ViT/DINOv3 attention 주변에서 GPU FP16 overflow → NaN 이 안 나도록 막는다.
#  2) `onnxconverter_common` 경로(`--safe`): 동일.
#
# 특히 attention의 `1/sqrt(d_k)` 부분의 `Sqrt`, LayerNorm 내부의 `Pow`/`ReduceMean`/`Div`
# 가 FP16에서 underflow/overflow 하면 단 하나의 Inf가 그래프 전체로 전파되어
# 출력이 전부 NaN이 된다. 그래서 이 산술 op들을 모두 FP32로 유지한다.
SAFE_OP_BLOCK_LIST = [
    # --- Normalization (LayerNorm/RMSNorm 류) ---
    "LayerNormalization",
    "InstanceNormalization",
    "GroupNormalization",
    "SkipLayerNormalization",
    "RMSNormalization",
    # --- Softmax / Reductions (overflow 위험) ---
    "Softmax",
    "LogSoftmax",
    "ReduceMean",
    "ReduceSum",
    "ReduceMax",
    "ReduceMin",
    "ReduceL2",
    # --- Attention scaling / division (1/sqrt(d_k) 류) ---
    "Sqrt",
    "Rsqrt",
    "Pow",
    "Div",
    "Reciprocal",
    "Erf",
    "Exp",
    "Log",
    # --- Resampling / lookup ---
    "Resize",
    "GridSample",
    "Range",
    # --- Boolean/conditional 분기 ---
    "Where",
    # --- Transformer 그래프에 자주 있는 명시적 Cast ---
    # ViT/DINOv3는 명시적 `Cast(to=float32)`를 가진 노드가 있어,
    # FP16 변환기가 그 attribute를 안 건드리면 dtype mismatch가 난다.
    # FP32로 유지하면 변환기가 주변에 맞춰 Cast를 끼워준다.
    "Cast",
    # --- 합쳐진 attention op ---
    "Attention",
    "MultiHeadAttention",
]


# Attention 경로의 노드를 FP32로 유지하기 위한 기본 정규식.
# BoxerNet(DINOv3 ViT + self-attention head) 그래프에는 다음과 같은 이름이 있다:
#   /blocks.0/attn/Sqrt, /blocks.0/attn/MatMul, /blocks.0/attn/MatMul_1, ...
#   /self_attn/0/Sqrt,   /self_attn/0/MatMul, ...
# 이 MatMul(=QK^T, attn·V) 두 개와 scaling Sqrt만 FP32로 유지하면
# ViT 본체(Linear projection MatMul)는 FP16으로 남아 모델 사이즈를 지킬 수 있다.
#
# 필요하면 `--node-block-regex` 로 override 하거나 추가할 수 있다.
DEFAULT_ATTENTION_NODE_REGEX = r"(attn|self_attn|cross_attn).*/(Sqrt|MatMul|Softmax|Div|Mul)(_\d+)?$"


def _collect_nodes_matching(model, patterns: Iterable[str]) -> list[str]:
    """모델 그래프에서 정규식에 매칭되는 노드 이름들을 모아 반환.

    여러 패턴을 OR로 합친다. 노드 이름이 비어있는 것은 건너뛴다.
    매칭된 노드 목록을 돌려주면 호출부가 이를 `node_block_list`로 전달한다.
    """
    compiled = [re.compile(p) for p in patterns if p]
    if not compiled:
        return []
    matched: list[str] = []
    for node in model.graph.node:
        name = node.name
        if not name:
            continue
        if any(rx.search(name) for rx in compiled):
            matched.append(name)
    return matched


def _convert_with_ort_transformers(
    model,
    keep_io_types: bool,
    extra_op_block_list,
    node_block_list,
):
    """ORT 자체의 transformer 친화 FP16 변환기.

    - LayerNorm/Softmax/Attention 주변에 Cast를 정확히 자동 삽입한다.
    - symbolic shape inference를 사용해 대형 ViT/DINO 그래프에서도 빠르다
      (`onnxconverter_common`의 일반 shape inference와 달리 분 단위가 아니라 초~수십 초).
    - ORT 1.24.x 기준 시그니처. 인자 이름이 다른 경우엔 자동으로 폴백한다.

    NOTE on `op_block_list`:
        ORT 변환기는 자체 default block list (TopK/Resize/Range/NMS 등 ML ops)를
        가지고 있고, `op_block_list=...`을 넘기면 그 리스트를 *교체*한다.
        그래서 default를 잃지 않으려면 default + 우리 SAFE 목록을 합쳐서 넘겨야 한다.

    NOTE on `node_block_list`:
        `op_block_list`가 *op 타입* 단위라면 이건 *노드 이름* 단위다. ViT의 Linear
        projection MatMul은 FP16으로 두고 싶고, attention scoring MatMul(QK^T, attn·V)만
        FP32로 유지하고 싶을 때 필수적이다. MatMul을 op_block_list에 넣으면 모든
        Linear 가중치가 FP32로 남아 사이즈 이득이 사라진다.
    """
    from onnxruntime.transformers import float16 as ort_fp16
    from onnxruntime.transformers.float16 import convert_float_to_float16

    default_block = list(getattr(ort_fp16, "DEFAULT_OP_BLOCK_LIST", []))
    merged_block = sorted(set(default_block) | set(extra_op_block_list or []))
    print(f"  op_block_list (FP32 유지, {len(merged_block)}개): {merged_block}", flush=True)
    if node_block_list:
        preview = node_block_list if len(node_block_list) <= 12 else (
            list(node_block_list[:12]) + [f"... (+{len(node_block_list) - 12})"]
        )
        print(f"  node_block_list (FP32 유지, {len(node_block_list)}개): {preview}",
              flush=True)

    common_kwargs = dict(
        keep_io_types=keep_io_types,
        disable_shape_infer=False,
        use_symbolic_shape_infer=True,
        op_block_list=merged_block,
        node_block_list=list(node_block_list) if node_block_list else None,
        # 가중치를 명시적으로 FP16으로 박아 디스크/RAM 둘 다 줄인다.
        force_fp16_initializers=True,
    )
    try:
        return convert_float_to_float16(model, **common_kwargs)
    except TypeError:
        # 일부 ORT 버전은 아래 인자가 없다 — 하나씩 떼면서 재시도.
        for k in ("force_fp16_initializers", "use_symbolic_shape_infer"):
            common_kwargs.pop(k, None)
            try:
                return convert_float_to_float16(model, **common_kwargs)
            except TypeError:
                continue
        # 마지막 폴백: 최소 인자만.
        return convert_float_to_float16(
            model,
            keep_io_types=keep_io_types,
            op_block_list=merged_block,
            node_block_list=list(node_block_list) if node_block_list else None,
        )


def convert(
    input_path: Path,
    output_path: Path,
    keep_io_types: bool,
    safe: bool,
    fast: bool,
    use_ort: bool,
    node_block_regex: list[str],
) -> None:
    import time
    import onnx

    t0 = time.time()
    print(f"Loading {input_path} ({input_path.stat().st_size / 1e6:.1f} MB)…", flush=True)
    model = onnx.load(str(input_path))
    print(f"  Loaded in {time.time() - t0:.1f}s. "
          f"{len(model.graph.node)} nodes, {len(model.graph.initializer)} initializers.",
          flush=True)

    t1 = time.time()
    if use_ort:
        print("Converting with onnxruntime.transformers.float16 (transformer-friendly)…",
              flush=True)
        # `--ort` 경로에서도 `--safe`로 SAFE_OP_BLOCK_LIST를 강제 적용할 수 있게 한다.
        # 이 리스트가 빠지면 attention의 Sqrt/Pow/Div가 FP16이 되어 GPU에서 NaN을 뱉는다.
        extra_block = SAFE_OP_BLOCK_LIST if safe else []

        # 정규식에 매칭되는 노드 이름들을 수집해서 node_block_list로 전달.
        # 사용자가 `--node-block-regex`를 주지 않았으면 기본 attention 패턴을 사용.
        regex_list = node_block_regex or [DEFAULT_ATTENTION_NODE_REGEX]
        matched_nodes = _collect_nodes_matching(model, regex_list)
        print(
            f"  node_block_regex (정규식 {len(regex_list)}개): {regex_list} "
            f"→ 매칭 노드 {len(matched_nodes)}개",
            flush=True,
        )

        fp16_model = _convert_with_ort_transformers(
            model,
            keep_io_types=keep_io_types,
            extra_op_block_list=extra_block,
            node_block_list=matched_nodes,
        )
    else:
        from onnxconverter_common import float16

        op_block_list = SAFE_OP_BLOCK_LIST if safe else None
        if op_block_list:
            print(f"Keeping these ops in FP32: {op_block_list}", flush=True)
        if fast:
            print("Skipping ONNX shape inference (fast mode).", flush=True)
        print("Converting with onnxconverter_common.float16…", flush=True)
        fp16_model = float16.convert_float_to_float16(
            model,
            keep_io_types=keep_io_types,
            # Shape inference can take 5-30 min on large transformer graphs when
            # the `--safe` block list is also enabled.
            disable_shape_infer=fast,
            op_block_list=op_block_list,
        )
    print(f"  Converted in {time.time() - t1:.1f}s.", flush=True)

    t2 = time.time()
    print(f"Saving {output_path}…", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(fp16_model, str(output_path))
    print(f"  Saved in {time.time() - t2:.1f}s.", flush=True)

    in_size = input_path.stat().st_size / 1e6
    out_size = output_path.stat().st_size / 1e6
    print(
        f"Done in {time.time() - t0:.1f}s total. "
        f"{in_size:.1f} MB -> {out_size:.1f} MB "
        f"({100 * (1 - out_size / in_size):.1f}% smaller)",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input .onnx path")
    parser.add_argument("--output", type=Path, required=True, help="Output .onnx path")
    parser.add_argument(
        "--cast-io",
        action="store_true",
        help="Convert inputs/outputs to fp16 too (default: keep float32 IO).",
    )
    parser.add_argument(
        "--safe",
        action="store_true",
        help="Keep accuracy-sensitive ops (LayerNorm, Softmax, Sqrt, Pow, Div, Cast, ...) "
             "in FP32. Strongly recommended for ViT/DINOv3 backbones — without this, "
             "GPU FP16 in CoreML EP overflows in attention and the whole graph emits NaN. "
             "Works with both `--ort` and the legacy onnxconverter_common path.",
    )
    parser.add_argument(
        "--full-shape-infer",
        action="store_true",
        help="Re-run ONNX shape inference during conversion. Slow on large transformer "
             "graphs (often 10+ minutes). Off by default. Ignored with --ort.",
    )
    parser.add_argument(
        "--ort",
        action="store_true",
        help="Use onnxruntime.transformers.float16 instead of onnxconverter_common. "
             "Recommended for ViT/DINO/transformer backbones (BoxerNet). "
             "Auto-inserts Casts around LayerNorm/Softmax/Attention and uses fast "
             "symbolic shape inference. Requires `pip install onnxruntime`.",
    )
    parser.add_argument(
        "--node-block-regex",
        action="append",
        default=None,
        metavar="PATTERN",
        help="Regex matched against `node.name`; matching nodes stay FP32. "
             "Can be passed multiple times (patterns are OR'd). "
             f"Default when `--ort` is set and this flag is omitted: "
             f"'{DEFAULT_ATTENTION_NODE_REGEX}' "
             "— keeps attention QK^T / attn·V MatMul, scale Sqrt, softmax Div "
             "in FP32 while Linear projection MatMuls remain FP16. "
             "Use '^$' to disable node blocking entirely.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found", file=sys.stderr)
        return 1

    convert(
        args.input,
        args.output,
        keep_io_types=not args.cast_io,
        safe=args.safe,
        fast=not args.full_shape_infer,
        use_ort=args.ort,
        node_block_regex=args.node_block_regex or [],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
