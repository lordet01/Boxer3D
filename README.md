# Boxer3D

AR 3D object detection for iPhone with LiDAR. Detects objects with YOLO and lifts them to 3D oriented bounding boxes using [BoxerNet](https://facebookresearch.github.io/boxer/) (Meta Research), displayed in augmented reality.

## Demo

<img width="1418" height="849" alt="banner" src="https://github.com/user-attachments/assets/05cd68b9-7df5-41d1-95fe-4197fdb539d5" />

![Boxer3D](https://github.com/user-attachments/assets/e5c5cbce-bcf3-48d4-9994-d317f647950d)

## How It Works

```
iPhone Camera + LiDAR
       │
       ├──► YOLO11n (2D detection) ──► top 3 bounding boxes
       │
       ├──► LiDAR depth ──► median depth per 16×16 patch
       │
       ├──► ARKit ──► camera pose + intrinsics + gravity
       │
       └──► BoxerNet (3D lifting) ──► oriented 3D bounding boxes
                                           │
                                     SceneKit AR rendering
```

1. **YOLO11n** detects objects in 2D (640×640, 80 COCO classes)
2. **BoxerNet** lifts 2D boxes to 7-DoF 3D boxes (center, size, yaw) using DINOv3 visual features + LiDAR depth
3. **ARKit** Camera poses + Gravity Vector + LiDAR depth
4. **SceneKit** renders 3D wireframe boxes anchored in the real world

## Requirements

- iPhone 12 Pro or later (LiDAR required)
- iOS 16.0+
- ~450 MB storage for models

## Setup

1. **Clone**
   ```bash
   git clone git@github.com:Barath19/Boxer3D.git
   cd Boxer3D
   ```

2. **Download models** from [Hugging Face](https://huggingface.co/Barath/boxer3d)
   ```bash
   pip install huggingface_hub
   huggingface-cli download Barath/boxer3d --local-dir boxer/
   ```
   This places the following in the `boxer/` directory:
   - `BoxerNet.onnx` (~391 MB, float32) — exported from BoxerNet checkpoint
   - `yolo11n.onnx` (~10 MB, float32) — exported from Ultralytics YOLO11n

3. **Open in Xcode**
   ```bash
   open boxer.xcodeproj
   ```
   Xcode will automatically resolve the ONNX Runtime SPM dependency.

4. **Build & Run** on your iPhone (Cmd+R)

## Models

| Model | Size | Input | Output |
|-------|------|-------|--------|
| **yolo11n** | 10 MB | (1, 3, 640, 640) RGB | (1, 84, 8400) boxes + classes |
| **BoxerNet** | 391 MB | (1, 3, 960, 960) RGB + (1, 1, 60, 60) depth + (1, M, 4) boxes + (1, 3600, 6) rays | (M, 3) center, (M, 3) size, (M,) yaw, (M,) confidence |

Both models run with ONNX Runtime CoreML Execution Provider for Metal/Neural Engine acceleration.

## Memory (Jetsam) on iPhone — incl. iPhone 15 Pro

The 391 MB float32 BoxerNet, plus YOLO, plus 960×960 tensors, plus ARKit / SceneKit / camera capture, plus **CoreML graph compilation** on the first run, can exceed the per-app memory limit and the OS will kill the app (`Terminated due to memory issue`).

**What this project already does to lower peak RAM**

- **Serial ORT sessions:** YOLO runs first; the YOLO session is released **before** building the 960×960 tensors and loading BoxerNet, so the two large ONNX sessions never live at the same time.
- **Flat depth buffer:** LiDAR depth is kept as a single row-major `[Float]` instead of `[[Float]]`.
- **Release BoxerNet** after each detection so idle memory stays low (next tap reloads it).

### Step 1 — FP16 quantize the ONNX models (recommended)

This is the single biggest win. Weights are stored as float16 while inputs/outputs stay float32, so **no Swift code change is needed** — just swap the `.onnx` files in the Xcode bundle.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r scripts/requirements.txt

python scripts/quantize_models.py \
    --input  boxer/BoxerNet.onnx \
    --output boxer/BoxerNet.fp16.onnx

python scripts/quantize_models.py \
    --input  boxer/yolo11n.onnx \
    --output boxer/yolo11n.fp16.onnx
```

Then in Xcode replace the original files in the bundle (or rename the FP16 files to `BoxerNet.onnx` / `yolo11n.onnx`). Expected: BoxerNet ~391 MB → ~195 MB on disk and roughly half the runtime weight memory.

### Step 2 — Reduce other peak sources

1. **Run without Xcode** (tap the app on the Home Screen). Debugger, view debugging, and wireless link add memory pressure.
2. **Use a wired connection** when debugging; close other heavy apps and Safari tabs.
3. **First launch after install** is worst: CoreML compiles each graph; later launches reuse caches.
4. Lower the **BoxerNet input resolution** (e.g. 640) in the exported ONNX, or run one path on **CPU EP** only — both require a re-export and re-test.

### Step 3 — Direct CoreML conversion (optional)

ORT already routes nodes to CoreML, so converting the model to a real `.mlpackage` is **not required** to use the Neural Engine. If you want to drop ONNX Runtime entirely, the supported path is **PyTorch → CoreML** (`coremltools` 4.0+ no longer converts ONNX directly):

```python
import coremltools as ct
import torch

model = ...                   # original BoxerNet PyTorch model in eval()
example = torch.randn(1, 3, 960, 960)
traced = torch.jit.trace(model, example)

mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(shape=example.shape)],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS17,
)
mlmodel.save("BoxerNet.mlpackage")
```

This requires the original PyTorch checkpoint (not just the ONNX file) and replacing the Swift inference code with `MLModel` / `MLMultiArray`.

## Dependencies

- [ONNX Runtime](https://github.com/microsoft/onnxruntime-swift-package-manager) v1.24.2 (via SPM)
- ARKit, SceneKit, SwiftUI (built-in)

## Roadmap

- [x] Port BoxerNet to Swift
- [x] Convert BoxerNet.pt to ONNX
- [x] Upload ONNX weights for download
- [ ] Optimize for portrait mode

## Acknowledgments

Based on **Boxer: Robust Lifting of Open-World 2D Bounding Boxes to 3D** by Daniel DeTone, Tianwei Shen, Fan Zhang, Lingni Ma, Julian Straub, Richard Newcombe, and Jakob Engel (Meta Reality Labs Research).

- [Project page](https://facebookresearch.github.io/boxer/)
- [GitHub](https://github.com/facebookresearch/boxer)

```bibtex
@article{boxer2026,
  title={Boxer: Robust Lifting of Open-World 2D Bounding Boxes to 3D},
  author={Daniel DeTone and Tianwei Shen and Fan Zhang and Lingni Ma 
          and Julian Straub and Richard Newcombe and Jakob Engel},
  year={2026},
}
```
