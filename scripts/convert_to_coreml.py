"""ONNX BoxerNet → CoreML(.mlpackage) 변환기.

Apple의 CoreML 컨버터는 ONNX Runtime + CoreML EP와 달리:
  * Transformer-aware 한 dtype 정책으로 LayerNorm/Softmax/Attention 주변을
    자동으로 안전하게 다룬다 (FP16 GPU overflow → NaN 회피).
  * `MLProgram` 으로 변환하면 ANE/GPU/CPU 어느 쪽에서도 컴파일/실행 가능.
  * 디스크/메모리 footprint가 ONNX보다 훨씬 작다.

파이프라인:
    ONNX (FP32)
        └─[onnx2torch]→ PyTorch nn.Module
              └─[torch.jit.trace]→ TorchScript
                    └─[coremltools.convert]→ .mlpackage (MLProgram)

Usage:
    source .venv/bin/activate
    pip install -r scripts/requirements.txt
    python scripts/convert_to_coreml.py \\
        --input  models_backup/BoxerNet.fp32.onnx \\
        --output boxer/BoxerNet.mlpackage

옵션:
    --num-boxes 8     CoreML은 dynamic dim 운용이 까다로워 정수로 고정.
                      Swift 쪽 `BoxerNet.fixedNumBoxes` 와 일치해야 한다.
    --precision fp16  연산/가중치 FP16 (기본). fp32로 바꾸면 사이즈 ↑, 안정성 ↑.
    --target ios16    minimum_deployment_target. iOS 17+로 올리면 더 큰 ANE op 지원.
    --check           변환 후 더미 입력으로 ONNX vs CoreML 출력 비교 (NaN/numeric drift).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple


def _patch_onnx2torch_missing_ops() -> None:
    """onnx2torch가 누락한 op들을 PyTorch 모듈로 직접 등록.

    BoxerNet이 사용하는데 onnx2torch가 빠뜨린 op:
        * Size (opset 13+) — 입력 텐서의 element 총개수를 int64 스칼라로 반환.

    이 함수는 import 부작용이 없도록 변환 직전에 한 번만 호출한다.
    """
    import torch
    import torch.nn as nn
    from onnx2torch.node_converters.registry import add_converter
    from onnx2torch.onnx_graph import OnnxGraph
    from onnx2torch.onnx_node import OnnxNode
    from onnx2torch.utils.common import OperationConverterResult
    from onnx2torch.utils.common import onnx_mapping_from_node

    class _OnnxSize(nn.Module):
        """ONNX `Size` 의 등가물: x.numel() (int64 스칼라)."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
            return torch.tensor(x.numel(), dtype=torch.int64)

    def _converter(
        node: OnnxNode, graph: OnnxGraph,
    ) -> OperationConverterResult:
        return OperationConverterResult(
            torch_module=_OnnxSize(),
            onnx_mapping=onnx_mapping_from_node(node=node),
        )

    # Size op는 opset 1/13/19/21 모두 동일 의미. 다 같은 컨버터로 등록.
    for v in (1, 13, 19, 21):
        try:
            add_converter(operation_type="Size", version=v)(_converter)
        except Exception:
            # 같은 (op, version) 이 이미 등록된 경우 스킵.
            pass


# BoxerNet 입력 텐서 spec.
# ONNX 그래프와 일치해야 한다. (image, sdp_patches, bb2d, ray_encoding)
# - image:        (1, 3, 960, 960) float32, [0,1] 정규화
# - sdp_patches:  (1, 1, 60, 60)   float32, LiDAR depth median/패치
# - bb2d:         (1, M, 4)        float32, 정규화된 2D 박스 (xmin/xmax/ymin/ymax)
# - ray_encoding: (1, 3600, 6)     float32, Plucker ray 인코딩
def _build_dummy_inputs(num_boxes: int, seed: int = 0) -> Tuple:
    import torch
    g = torch.Generator().manual_seed(seed)
    image = torch.randn(1, 3, 960, 960, generator=g, dtype=torch.float32) * 0.1 + 0.5
    sdp = torch.rand(1, 1, 60, 60, generator=g, dtype=torch.float32) * 2.0 + 0.5
    bb2d = torch.rand(1, num_boxes, 4, generator=g, dtype=torch.float32)
    ray = torch.randn(1, 3600, 6, generator=g, dtype=torch.float32) * 0.3
    return image, sdp, bb2d, ray


def _summarize(name: str, arr) -> str:
    """텐서/배열의 한 줄 요약: shape + min/max/mean + NaN/Inf 카운트."""
    import numpy as np
    a = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
    return (
        f"  [{name:<12}] shape={a.shape} "
        f"min={float(a.min()):+.4f} max={float(a.max()):+.4f} "
        f"mean={float(a.mean()):+.4f} "
        f"NaN={int(np.isnan(a).sum())} Inf={int(np.isinf(a).sum())}"
    )


def _max_diff(a, b) -> Tuple[float, float]:
    """두 배열의 |a-b|의 mean/max 반환."""
    import numpy as np
    a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
    b = b.detach().cpu().numpy() if hasattr(b, "detach") else np.asarray(b)
    d = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return float(d.mean()), float(d.max())


INPUT_NAMES: List[str] = ["image", "sdp_patches", "bb2d", "ray_encoding"]
OUTPUT_NAMES: List[str] = ["center", "size", "yaw", "confidence"]


def convert(
    input_path: Path,
    output_path: Path,
    num_boxes: int,
    precision: str,
    target: str,
    do_check: bool,
    trace_cache: Path | None = None,
    convert_to: str = "mlprogram",
    skip_diag: bool = False,
) -> None:
    import onnx
    import torch
    from onnx2torch import convert as onnx_to_torch
    import coremltools as ct

    t0 = time.time()
    inputs_a = _build_dummy_inputs(num_boxes, seed=0)
    inputs_b = _build_dummy_inputs(num_boxes, seed=1)

    # ---- trace 캐시 hit 면 ONNX→PyTorch→trace 단계 (보통 200초+) 통째로 skip.
    traced = None
    if trace_cache is not None and trace_cache.exists():
        print(f"[cache] Loading traced TorchScript from {trace_cache}…",
              flush=True)
        traced = torch.jit.load(str(trace_cache), map_location="cpu")
        traced.eval()
        print(f"        Loaded in {time.time() - t0:.1f}s.", flush=True)
    else:
        # ------------------------------------------------------------- load.
        print(f"[1/5] Loading {input_path} "
              f"({input_path.stat().st_size / 1e6:.1f} MB)…", flush=True)
        onnx_model = onnx.load(str(input_path))
        print(f"      Loaded in {time.time() - t0:.1f}s "
              f"({len(onnx_model.graph.node)} nodes).", flush=True)

        # --------------------------------------- ONNX → PyTorch nn.Module.
        t1 = time.time()
        print("[2/5] Converting ONNX → PyTorch (onnx2torch)…", flush=True)
        _patch_onnx2torch_missing_ops()
        torch_model = onnx_to_torch(onnx_model)
        torch_model.eval()
        print(f"      Done in {time.time() - t1:.1f}s.", flush=True)

        if not skip_diag:
            # ------------ 진단 A: untraced PyTorch가 입력에 반응하는지.
            print("\n[diag/A] PyTorch (untraced) sensitivity to inputs:",
                  flush=True)
            with torch.no_grad():
                out_a_pt = torch_model(*inputs_a)
                out_b_pt = torch_model(*inputs_b)
            if not isinstance(out_a_pt, (list, tuple)):
                out_a_pt = (out_a_pt,); out_b_pt = (out_b_pt,)
            for i, (oa, ob) in enumerate(zip(out_a_pt, out_b_pt)):
                nm = OUTPUT_NAMES[i] if i < len(OUTPUT_NAMES) else f"out{i}"
                mean_d, max_d = _max_diff(oa, ob)
                flag = "  ⚠ CONSTANT (input ignored!)" if max_d < 1e-6 else ""
                print(f"  [{nm:<10}] |a-b|.mean={mean_d:.4f}  "
                      f"max={max_d:.4f}{flag}", flush=True)

        # --------------------------------------------- TorchScript trace.
        t2 = time.time()
        print(f"\n[3/5] Tracing TorchScript (num_boxes fixed to {num_boxes})…",
              flush=True)
        dummy_inputs = _build_dummy_inputs(num_boxes, seed=0)
        with torch.no_grad():
            traced = torch.jit.trace(torch_model, dummy_inputs, strict=False,
                                     check_trace=False)
        print(f"      Traced in {time.time() - t2:.1f}s.", flush=True)

        if trace_cache is not None:
            trace_cache.parent.mkdir(parents=True, exist_ok=True)
            traced.save(str(trace_cache))
            print(f"      Saved trace cache → {trace_cache} "
                  f"({trace_cache.stat().st_size / 1e6:.1f} MB)", flush=True)

    if not skip_diag:
        # ------------ 진단 B: traced 모델이 입력에 반응하는지.
        print("\n[diag/B] TorchScript (traced) sensitivity to inputs:",
              flush=True)
        with torch.no_grad():
            out_a_tr = traced(*inputs_a)
            out_b_tr = traced(*inputs_b)
        if not isinstance(out_a_tr, (list, tuple)):
            out_a_tr = (out_a_tr,); out_b_tr = (out_b_tr,)
        for i, (oa, ob) in enumerate(zip(out_a_tr, out_b_tr)):
            nm = OUTPUT_NAMES[i] if i < len(OUTPUT_NAMES) else f"out{i}"
            mean_d, max_d = _max_diff(oa, ob)
            flag = "  ⚠ CONSTANT (trace fold!)" if max_d < 1e-6 else ""
            print(f"  [{nm:<10}] |a-b|.mean={mean_d:.4f}  "
                  f"max={max_d:.4f}{flag}", flush=True)

    # --------------------------------------------- TorchScript → CoreML mlpkg.
    t3 = time.time()
    print(f"[4/5] Converting TorchScript → CoreML (precision={precision}, "
          f"target={target})…", flush=True)
    target_map = {
        "ios15": ct.target.iOS15,
        "ios16": ct.target.iOS16,
        "ios17": ct.target.iOS17,
        "ios18": ct.target.iOS18,
    }
    # transformer/ViT 에서 fp16 으로 내리면 inf/NaN/saturate 가 자주 나는 op들.
    # 이 op 들은 fp32 로 유지하면서, 나머지는 fp16 으로 캐스트 → 안전 + 작은 사이즈.
    # (BoxerNet fp16 변환 시 모든 출력이 상수로 fold 되는 증상의 root cause)
    UNSAFE_FP16_OPS = {
        "layer_norm",
        "rms_norm",
        "instance_norm",
        "l2_norm",
        "batch_norm",
        "softmax",
        "log_softmax",
        "rsqrt",
        "sqrt",
        "gelu",
        "reduce_sum",
        "reduce_mean",
        "reduce_l2_norm",
        "real_div",
    }

    def _safe_fp16_selector(op):  # noqa: ANN001 - mil Op type
        # True 를 반환하면 fp16 으로 cast, False 면 fp32 유지.
        return op.op_type not in UNSAFE_FP16_OPS

    if precision == "fp16":
        compute_precision = ct.precision.FLOAT16
    elif precision in ("fp32", "fp32-w8", "fp32-w16"):
        # weight quantize 변형들은 일단 FP32 로 변환한 뒤 후처리에서 weight 만 압축.
        compute_precision = ct.precision.FLOAT32
    elif precision == "fp16-safe":
        compute_precision = ct.transform.FP16ComputePrecision(
            op_selector=_safe_fp16_selector
        )
    else:
        raise SystemExit(
            f"--precision unknown: {precision!r}"
        )
    if target not in target_map:
        raise SystemExit(f"--target must be one of {list(target_map)}")

    ct_inputs = [
        ct.TensorType(name="image",
                      shape=(1, 3, 960, 960), dtype=float),
        ct.TensorType(name="sdp_patches",
                      shape=(1, 1, 60, 60), dtype=float),
        ct.TensorType(name="bb2d",
                      shape=(1, num_boxes, 4), dtype=float),
        ct.TensorType(name="ray_encoding",
                      shape=(1, 3600, 6), dtype=float),
    ]
    ct_outputs = [ct.TensorType(name=name) for name in OUTPUT_NAMES]

    convert_kwargs = dict(
        inputs=ct_inputs,
        outputs=ct_outputs,
        convert_to=convert_to,
        minimum_deployment_target=target_map[target],
    )
    # neuralnetwork 형식은 compute_precision 인자를 받지 않는다 (FP32 only).
    if convert_to == "mlprogram":
        convert_kwargs["compute_precision"] = compute_precision
    mlmodel = ct.convert(traced, **convert_kwargs)

    # ----- 옵션: activation 은 FP32 유지하면서 weight 만 quantize 해서 사이즈 ↓.
    # transformer 류는 activation FP16 변환 시 saturate 가 잘 일어나서, 정확성을
    # 위해 activation 은 FP32 로 두고 weight 만 압축하는 게 가장 안전한 조합이다.
    if precision == "fp32-w8":
        from coremltools.optimize.coreml import (
            linear_quantize_weights, OptimizationConfig,
            OpLinearQuantizerConfig,
        )
        print("      [w8] Quantizing weights to int8 (linear_symmetric)…",
              flush=True)
        cfg = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(
                mode="linear_symmetric", weight_threshold=512,
            )
        )
        t_q = time.time()
        mlmodel = linear_quantize_weights(mlmodel, config=cfg)
        print(f"      [w8] Done in {time.time() - t_q:.1f}s.", flush=True)
    elif precision == "fp32-w16":
        from coremltools.optimize.coreml import (
            palettize_weights, OptimizationConfig, OpPalettizerConfig,
        )
        # 6-bit palettize (weight 만; activation fp32 유지) → 사이즈 ~75MB
        print("      [w6] Palettizing weights to 6-bit…", flush=True)
        cfg = OptimizationConfig(
            global_config=OpPalettizerConfig(
                nbits=6, mode="kmeans", weight_threshold=512,
            )
        )
        t_q = time.time()
        mlmodel = palettize_weights(mlmodel, config=cfg)
        print(f"      [w6] Done in {time.time() - t_q:.1f}s.", flush=True)
    print(f"      Converted in {time.time() - t3:.1f}s.", flush=True)

    # --------------------------------------------------------- save mlpackage.
    t4 = time.time()
    print(f"[5/5] Saving {output_path}…", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if output_path.is_dir():
            import shutil
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    mlmodel.save(str(output_path))

    # 결과 사이즈 합 (mlpackage는 디렉토리)
    def _dir_size(p: Path) -> float:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    out_size = _dir_size(output_path) / 1e6 if output_path.is_dir() else (
        output_path.stat().st_size / 1e6
    )
    in_size = input_path.stat().st_size / 1e6
    print(f"      Saved in {time.time() - t4:.1f}s. "
          f"{in_size:.1f} MB ONNX → {out_size:.1f} MB mlpackage "
          f"({100 * (1 - out_size / in_size):.1f}% smaller)",
          flush=True)

    # --------------------------------------------- (옵션) 출력 일치 sanity-check.
    if do_check:
        print("\n[check] Comparing ONNX vs CoreML on dummy inputs…", flush=True)
        _verify_outputs(input_path, output_path, num_boxes=num_boxes)

    print(f"Done in {time.time() - t0:.1f}s total.", flush=True)


def _verify_outputs(onnx_path: Path, mlpkg_path: Path, num_boxes: int) -> None:
    """ONNX 와 CoreML 출력을 두 가지 다른 입력에 대해 비교.

    핵심 질문 두 가지:
      1) CoreML 출력이 입력 변화에 반응하는가? (입력 a vs 입력 b 출력 차이)
      2) CoreML 출력이 ONNX baseline과 유사한가?
    1)에서 |a-b| 가 0이면 CoreML 모델이 trace 시 상수로 fold된 것.
    """
    import numpy as np
    import onnxruntime as ort
    import coremltools as ct

    inputs_a = _build_dummy_inputs(num_boxes, seed=0)
    inputs_b = _build_dummy_inputs(num_boxes, seed=1)

    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    feed_a = {name: arr.numpy() for name, arr in zip(INPUT_NAMES, inputs_a)}
    feed_b = {name: arr.numpy() for name, arr in zip(INPUT_NAMES, inputs_b)}
    onnx_a = sess.run(OUTPUT_NAMES, feed_a)
    onnx_b = sess.run(OUTPUT_NAMES, feed_b)

    mlm = ct.models.MLModel(str(mlpkg_path),
                            compute_units=ct.ComputeUnit.CPU_ONLY)
    cml_a = mlm.predict(feed_a)
    cml_b = mlm.predict(feed_b)

    print(f"  CoreML output keys: {sorted(cml_a.keys())}", flush=True)
    print(f"  --- ONNX vs CoreML  (seed=0 input)  AND  CoreML(seed=0) vs CoreML(seed=1) ---",
          flush=True)
    for i, name in enumerate(OUTPUT_NAMES):
        oa = onnx_a[i]
        ob = onnx_b[i]
        ca = cml_a.get(name)
        cb = cml_b.get(name)
        if ca is None or cb is None:
            print(f"  [{name}] MISSING from CoreML output")
            continue
        ca = np.asarray(ca); cb = np.asarray(cb)

        # 1) 입력에 따라 출력이 변하는가? (가장 중요)
        cml_self_diff = np.abs(ca.astype(np.float32) - cb.astype(np.float32))
        onnx_self_diff = np.abs(oa.astype(np.float32) - ob.astype(np.float32))
        cml_const = "  ⚠ CONSTANT" if cml_self_diff.max() < 1e-6 else ""
        # 2) ONNX와 비슷한가? (FP16 노이즈 정도면 OK)
        cml_vs_onnx = np.abs(oa.astype(np.float32) - ca.astype(np.float32))
        print(
            f"  [{name:<10}]"
            f"  ONNX|a-b|.max={float(onnx_self_diff.max()):.4f}"
            f"  CoreML|a-b|.max={float(cml_self_diff.max()):.4f}{cml_const}"
            f"  | ONNX↔CoreML(seed=0).max={float(cml_vs_onnx.max()):.4f}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True,
                        help="입력 ONNX 파일 (FP32 권장).")
    parser.add_argument("--output", type=Path, required=True,
                        help="출력 .mlpackage 디렉토리 경로 "
                             "(예: boxer/BoxerNet.mlpackage).")
    parser.add_argument("--num-boxes", type=int, default=8,
                        help="bb2d 의 num_boxes 차원을 이 정수로 고정. "
                             "Swift `BoxerNet.fixedNumBoxes`와 일치시킬 것 (default: 8).")
    parser.add_argument("--precision", default="fp32-w8",
                        choices=["fp16", "fp32", "fp16-safe",
                                 "fp32-w8", "fp32-w16"],
                        help="(default: fp32-w8) "
                             "fp16=풀 fp16(transformer 깨짐), "
                             "fp32=풀 fp32(정확/큰 사이즈), "
                             "fp16-safe=selective fp16(BoxerNet에선 작동 X), "
                             "fp32-w8=activation fp32 + weight int8(권장; ~100MB), "
                             "fp32-w16=activation fp32 + weight 6-bit palettize.")
    parser.add_argument("--target", default="ios16",
                        choices=["ios15", "ios16", "ios17", "ios18"],
                        help="minimum_deployment_target (default: ios16).")
    parser.add_argument("--check", action="store_true",
                        help="변환 후 ONNX vs CoreML 출력을 비교 (NaN/numeric).")
    parser.add_argument("--trace-cache", type=Path, default=None,
                        help="TorchScript trace 결과 캐시 경로. "
                             "있으면 ONNX→PyTorch→trace 단계 (200초+) 통째 skip.")
    parser.add_argument("--convert-to", default="mlprogram",
                        choices=["mlprogram", "neuralnetwork"],
                        help="CoreML 형식. neuralnetwork는 FP32 only이지만 "
                             "MIL constant-folding 버그를 우회할 때 시도.")
    parser.add_argument("--no-diag", action="store_true",
                        help="PyTorch 단계 진단 (diag/A,B) 생략. trace-cache 적중 시 의미 없음.")
    args = parser.parse_args()

    if not args.input.exists() and (args.trace_cache is None
                                     or not args.trace_cache.exists()):
        print(f"ERROR: input not found: {args.input} (그리고 trace-cache 없음)",
              file=sys.stderr)
        return 1

    convert(
        args.input,
        args.output,
        num_boxes=args.num_boxes,
        precision=args.precision,
        target=args.target,
        do_check=args.check,
        trace_cache=args.trace_cache,
        convert_to=args.convert_to,
        skip_diag=args.no_diag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
