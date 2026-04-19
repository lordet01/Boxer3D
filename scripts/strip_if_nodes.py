"""ONNX 그래프를 onnx2torch 가 받을 수 있는 형태로 다듬는다.

수행하는 작업:
  1) `If` 노드 폴딩
     BoxerNet의 `/rope_embed/If*` 는 condition이 입력 텐서에 무관한 컴파일타임
     상수라, ORT로 1회 추론해서 결과를 Constant 노드로 박아넣는다.
  2) `Clip` 노드의 빈 input 채우기
     ONNX `Clip(x, min='', max=...)` 처럼 일부 input이 빈 문자열이면 onnx2torch가
     `Dynamic value of min/max is not implemented` 로 죽는다. 비어있는 자리에
     ±inf 상수를 채워서 standard 3-input 형태로 만든다.

목적: ONNX → PyTorch (`onnx2torch`) → CoreML (`coremltools`) 변환 파이프라인의
사전 정리 단계.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


def _make_constant_node(name: str, output_name: str, value: np.ndarray) -> onnx.NodeProto:
    """주어진 numpy 값을 그대로 출력하는 Constant 노드 생성."""
    tensor = numpy_helper.from_array(value, name=f"{name}_value")
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[output_name],
        name=name,
        value=tensor,
    )


def _fill_clip_empty_inputs(model: onnx.ModelProto) -> int:
    """`Clip(x, min='', max=...)` 등의 빈 input 자리에 ±inf 상수를 채운다.

    onnx2torch 가 빈 문자열 input을 dynamic value로 잘못 분류해 죽는 문제를 회피.
    Clip의 dtype은 첫 번째 input과 동일해야 하므로 graph value_info / shape inference
    결과로 추정한다. 모르면 float32로 가정 (BoxerNet은 모두 float32).
    """
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    name_to_dtype: dict[str, int] = {}
    for vi in (list(inferred.graph.value_info)
               + list(inferred.graph.input)
               + list(inferred.graph.output)):
        name_to_dtype[vi.name] = vi.type.tensor_type.elem_type

    new_const_nodes: list[onnx.NodeProto] = []
    patched = 0
    for node in model.graph.node:
        if node.op_type != "Clip":
            continue
        # Clip 의 input 순서: [data, min, max] (ONNX opset >= 11).
        # 빈 문자열 = "없음" 을 의미 → ±inf 상수로 대체.
        data_name = node.input[0] if len(node.input) >= 1 else None
        if data_name is None:
            continue
        elem = name_to_dtype.get(data_name, onnx.TensorProto.FLOAT)
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(elem)

        for idx, sentinel, fill in [(1, "min", -np.inf), (2, "max", np.inf)]:
            if idx >= len(node.input):
                # input 자체가 없으면 추가 필요 (대부분 idx=2 max 가 누락된 경우).
                pass
            elif node.input[idx] != "":
                continue
            const_name = (
                f"{node.name}__{sentinel}_fill"
                if node.name else f"_clip_{sentinel}_fill_{patched}"
            )
            output_name = f"{const_name}_out"
            arr = np.array(fill, dtype=np_dtype)
            new_const_nodes.append(
                _make_constant_node(name=const_name, output_name=output_name, value=arr)
            )
            if idx >= len(node.input):
                node.input.append(output_name)
            else:
                node.input[idx] = output_name
            patched += 1

    if new_const_nodes:
        # Constant 노드는 그래프 앞쪽에 두어 topological order 깨지지 않게.
        model.graph.node.extend(new_const_nodes)
    return patched


def strip_if(input_path: Path, output_path: Path, num_boxes: int) -> None:
    print(f"Loading {input_path} ({input_path.stat().st_size / 1e6:.1f} MB)…",
          flush=True)
    model = onnx.load(str(input_path))

    # ─── 사전 정리: Clip 의 빈 input 자리에 ±inf 채우기 ───────────────────
    patched = _fill_clip_empty_inputs(model)
    if patched:
        print(f"Filled {patched} empty Clip input(s) with ±inf constants.",
              flush=True)

    if_nodes = [n for n in model.graph.node if n.op_type == "If"]
    if not if_nodes:
        print("No If nodes found — nothing to do.")
        if input_path != output_path:
            onnx.save(model, str(output_path))
        return

    print(f"Found {len(if_nodes)} If nodes. Their outputs will be folded "
          f"into Constants:", flush=True)
    if_outputs: list[str] = []
    for n in if_nodes:
        for o in n.output:
            if_outputs.append(o)
            print(f"  {n.name}.{o}")

    # ORT 추론 시 If 노드 출력들을 outputs 리스트에 추가해서 값을 끌어내기.
    # 원본 model의 graph.output 에 임시로 ValueInfo를 추가한 사본을 만든다.
    print("\nRunning ONNX shape inference to determine If output dtypes…",
          flush=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False,
                                                 data_prop=True)
    inferred_types: dict[str, int] = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.output):
        inferred_types[vi.name] = vi.type.tensor_type.elem_type

    print("\nRunning ONNX Runtime once to evaluate If outputs…", flush=True)
    eval_model = onnx.ModelProto()
    eval_model.CopyFrom(model)
    existing_outputs = {o.name for o in eval_model.graph.output}
    for name in if_outputs:
        if name in existing_outputs:
            continue
        elem = inferred_types.get(name, 0)
        if elem == 0:
            # shape inference가 If 출력 dtype을 알아내지 못한 경우, then_branch
            # 의 출력 dtype을 직접 들여다보고 거기서 가져온다.
            for n in if_nodes:
                if name in n.output:
                    for a in n.attribute:
                        if a.type == onnx.AttributeProto.GRAPH:
                            for sub_out in a.g.output:
                                if sub_out.name == name or len(a.g.output) == 1:
                                    elem = sub_out.type.tensor_type.elem_type
                                    if elem != 0:
                                        break
                        if elem != 0:
                            break
                    break
        if elem == 0:
            # 마지막 수단: int64 가정. RoPE rotary embed의 If는 INT64 또는 FLOAT.
            # 실제 dtype은 ORT가 돌리고 나서 if_values[name].dtype 로 다시 확인.
            elem = onnx.TensorProto.INT64
        vi = helper.make_tensor_value_info(name, elem, None)
        eval_model.graph.output.append(vi)

    # 임시 모델을 디스크에 쓰지 않고 직접 InferenceSession에 넘김.
    sess = ort.InferenceSession(
        eval_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )

    # 더미 입력 (값은 무관 — condition은 입력에 의존하지 않음).
    feed = {
        "image":        np.zeros((1, 3, 960, 960), dtype=np.float32),
        "sdp_patches":  np.zeros((1, 1, 60, 60),   dtype=np.float32),
        "bb2d":         np.zeros((1, num_boxes, 4), dtype=np.float32),
        "ray_encoding": np.zeros((1, 3600, 6),     dtype=np.float32),
    }
    out_names = [o.name for o in sess.get_outputs() if o.name in if_outputs]
    values = sess.run(out_names, feed)
    if_values: dict[str, np.ndarray] = dict(zip(out_names, values))

    print("Evaluated If outputs:")
    for name, arr in if_values.items():
        print(f"  {name}: dtype={arr.dtype}  shape={arr.shape}  "
              f"first={arr.ravel()[:4].tolist() if arr.size else '[]'}")

    # 그래프 수정: If 노드 제거 + 출력 이름을 Constant 노드로 대체.
    new_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.op_type == "If":
            for out_name in node.output:
                if out_name not in if_values:
                    raise RuntimeError(f"missing evaluated value for {out_name}")
                cnode = _make_constant_node(
                    name=f"{node.name}__folded__{out_name.replace('/', '_')}",
                    output_name=out_name,
                    value=if_values[out_name],
                )
                new_nodes.append(cnode)
            continue
        new_nodes.append(node)

    # 원본 graph.node 자리 갈아끼우기.
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    # 사용하지 않게 된 노드들은 ONNX 표준 검사에 무해하지만 dead code로 남는다.
    # onnx 모델 사이즈에 거의 영향 없고, onnx2torch 도 unused 노드는 무시한다.
    # 깔끔히 빼고 싶으면 onnx.utils.extract_model 로 graph 출력을 따로 다시 자르면 된다.

    print(f"\nSaving simplified model → {output_path}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))

    # 검증: 다시 로드해서 If가 사라졌는지 확인.
    re = onnx.load(str(output_path))
    remaining = [n for n in re.graph.node if n.op_type == "If"]
    print(f"If nodes remaining after strip: {len(remaining)}")
    print(f"Total nodes: {len(re.graph.node)} "
          f"(was {len(model.graph.node)} after edit)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-boxes", type=int, default=8,
                   help="bb2d 의 num_boxes 차원 (default: 8). 평가용 더미 입력에만 쓰임.")
    args = p.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found", file=sys.stderr)
        return 1
    strip_if(args.input, args.output, num_boxes=args.num_boxes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
