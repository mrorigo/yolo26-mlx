# YOLO26-MLX

A ground-up implementation of [YOLO26](https://docs.ultralytics.com/models/yolo26/) in
[Apple MLX](https://github.com/ml-explore/mlx). No PyTorch anywhere in the package: the model,
losses, label assignment, optimizer, data pipeline and metrics are written directly against the
MLX API.

## What is implemented

| Piece | Notes |
| --- | --- |
| Backbone / neck / head | C3k2, C2PSA (position-sensitive attention), SPPF, PAN, from `configs/yolo26.yaml` |
| Detection head | dual branch: one-to-many (with NMS) and one-to-one (NMS-free, top-300) |
| DFL-free regression | `reg_max=1`: four unconstrained ltrb channels, no distribution, no DFL term |
| Loss | CIoU (box) + BCE (class) + L1 on image-normalised ltrb, per branch |
| Label assignment | task-aligned (TAL) with YOLO26's STAL: small-target size floor and a second `topk2` filter |
| Progressive Loss | one-to-many weight decays 0.8 → 0.1 across training, shifting supervision to the inference-time branch |
| MuSGD | hybrid Muon (Newton-Schulz orthogonalization) + SGD, with Ultralytics' parameter grouping |
| Data | YOLO-format dataset, mosaic, affine, HSV, flip, letterbox |
| Metrics | greedy NMS, precision/recall, 101-point interpolated mAP@0.5:0.95 and mAP@0.5 |
| Export | safetensors weights (BN folded) loadable by the reference PyTorch implementation |

Parameter counts match the published values exactly:

```
n 2,572,280   s 10,009,784   m 21,896,248   l 26,299,704   x 58,993,368
```

## Install

```bash
uv sync            # or: pip install -e .
```

## Use

```bash
# print the parsed graph for a scale
yolo26-mlx info --scale n

# train (YOLO layout: images/train, labels/train, ...)
yolo26-mlx train --data coco8 --scale n --epochs 100 --imgsz 640 --batch 16

# validate, with NMS (one-to-many head) or without (one-to-one head)
yolo26-mlx val --data coco8 --weights runs/detect/yolo26n/last.safetensors
yolo26-mlx val --data coco8 --weights runs/detect/yolo26n/last.safetensors --nms-free

# predict
yolo26-mlx predict --source bus.jpg --weights runs/detect/yolo26n/last.safetensors --nms-free
```

As a library:

```python
import mlx.core as mx
from yolo26_mlx import build_model
from yolo26_mlx.cli import predict

model = build_model("yolo26", "n", pretrained="yolo26n.safetensors")
image = mx.array(...)                       # (1, H, W, 3) in 0..1
print(predict(model, image, conf=0.25, nms=False))   # NMS-free
```

## Correctness

The implementation is checked against PyTorch, not just against itself.
`tools/export_reference.py` instantiates the reference Ultralytics YOLO26, then exports weights
(converted to MLX layout), inputs, per-branch outputs, decoded outputs, assigner internals and
criterion values. `tools/check_parity.py` (and `tests/test_parity.py`) then compare everything:

```
one2many raw boxes:  max_abs_diff=7.153e-07   OK
one2many raw scores: max_abs_diff=0.000e+00   OK
one2one  raw boxes:  max_abs_diff=7.153e-07   OK
one-to-many decoded: max_abs_diff=3.052e-05   OK
one-to-one top-300:  max_abs_diff=0.000e+00   OK
one-to-many loss:    max_abs_diff=1.717e-05   OK
progressive loss:    max_abs_diff=6.676e-05   OK
PARITY OK
```

Reproduce with (PyTorch only needed for the exporter, never for the package):

```bash
uv venv refenv && uv pip install --python refenv/bin/python torch ultralytics safetensors
refenv/bin/python tools/export_reference.py --scale n --imgsz 256
.venv/bin/python tools/check_parity.py --scale n --imgsz 256
```

## Tests

```bash
uv run pytest -q          # 43 tests, ~17s
```

They cover the graph and parameter counts, layer shapes, BN folding, weight round-trips, the
assigner (including the STAL floor and multi-GT resolution), the loss terms and its differentiability,
the Progressive Loss schedule, MuSGD (orthogonalization, Nesterov, weight-decay grouping), the data
pipeline, NMS and mAP, plus an end-to-end overfit run that checks both heads detect.

## Layout

```
src/yolo26_mlx/
  tasks.py        model assembly from the YAML graph, stride probing, bias init
  config.py       YAML parsing, depth/width scaling (Ultralytics' parse_model rules)
  nn/             module system, conv/BN (with folding), blocks, detect head, ops
  loss/           TAL assigner with STAL, DFL-free criterion, Progressive Loss
  optim/musgd.py  Muon + SGD hybrid optimizer and parameter grouping
  data/           YOLO dataset, mosaic/affine/HSV augmentation, batching
  metrics.py      NMS, precision/recall, COCO-style mAP
  train.py        Trainer (MuSGD + schedules) and validate()
  cli.py          command line entry point
```

## Licence

The model architecture configuration in `configs/` follows Ultralytics YOLO26 (AGPL-3.0). All code
here is an independent MLX implementation.
