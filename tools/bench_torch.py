"""Benchmark the PyTorch reference implementation on the same work as ``tools/bench_mlx.py``.

    refenv/bin/python tools/bench_torch.py --scale n --imgsz 640 --batch 1 --device mps

Compares against the MLX implementation: same YAML, same weights, same inputs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics.nn.tasks import DetectionModel
from ultralytics.optim import MuSGD
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.tal import make_anchors


def sync(device: str) -> None:
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def peak_mb(device: str) -> float:
    if device == "mps":
        return torch.mps.current_allocated_memory() / 2**20
    return 0.0


def timeit(fn, device: str, warmup: int = 5, runs: int = 20) -> dict:
    for _ in range(warmup):
        fn()
    sync(device)
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        sync(device)
        samples.append((time.perf_counter() - t0) * 1e3)
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "peak_mb": peak_mb(device),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="n")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--nc", type=int, default=80)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--cfg", default="configs/yolo26.yaml")
    ap.add_argument("--refdir", default="/var/folders/7z/v9jlq_zj4jn5n04xjqkwr5400000gn/T/opencode")
    ap.add_argument("--json", default=None)
    ap.add_argument("--compile", action="store_true", help="wrap the model in torch.compile")
    args = ap.parse_args()

    device = args.device
    with open(args.cfg) as fh:
        cfg = yaml.safe_load(fh)
    cfg["scale"], cfg["nc"] = args.scale, args.nc
    torch.manual_seed(0)
    model = DetectionModel(cfg, verbose=False)
    weights = Path(args.refdir) / f"ref_{args.scale}_{args.imgsz}.safetensors"
    if weights.exists():
        from safetensors.numpy import load_file

        state = {}
        for k, v in load_file(str(weights)).items():
            # the file stores MLX kernels (O, kh, kw, I); torch wants (O, I, kh, kw)
            state[k if k.startswith("model.") else f"model.{k}"] = torch.tensor(
                np.ascontiguousarray(v.transpose(0, 3, 1, 2)) if v.ndim == 4 else v
            )
        model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.args = __import__("ultralytics.cfg", fromlist=["get_cfg"]).get_cfg(DEFAULT_CFG)
    model.nc = args.nc

    images = torch.randn(args.batch, 3, args.imgsz, args.imgsz, device=device)
    cls = torch.tensor([0.0, 5.0, 12.0] * args.batch, device=device)
    batch_idx = torch.tensor([0.0, 0.0, 0.0] * args.batch, device=device)
    bboxes = torch.tensor(
        [[0.4, 0.4, 0.3, 0.2], [0.7, 0.6, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]] * args.batch, device=device
    )
    batch = {"batch_idx": batch_idx.long(), "cls": cls.long(), "bboxes": bboxes}

    if args.compile:
        model = torch.compile(model)  # MLX's answer to eager mode

    results: dict[str, dict] = {"config": {**vars(args), "device": device, "torch": torch.__version__}}

    model.eval()
    model.end2end = True
    with torch.no_grad():
        results["inference_e2e"] = timeit(lambda: model(images), device, runs=args.runs)
    model.end2end = False
    with torch.no_grad():
        results["inference_o2m"] = timeit(lambda: model(images), device, runs=args.runs)
    fused = DetectionModel(cfg, verbose=False).to(device)
    fused.load_state_dict(model.state_dict(), strict=False)
    fused = fused.fuse().eval()
    with torch.no_grad():
        results["inference_fused"] = timeit(lambda: fused(images), device, runs=args.runs)

    model.train()
    criterion = E2ELoss(model, v8DetectionLoss)

    def fwd_loss():
        return criterion(model(images), batch)[0].sum()

    results["loss_only"] = timeit(fwd_loss, device, runs=args.runs)

    def fwd_loss_grad():
        model.zero_grad(set_to_none=True)
        loss = fwd_loss()
        loss.backward()
        return loss

    results["train_step"] = timeit(fwd_loss_grad, device, runs=args.runs)

    fwd_loss_grad()
    groups = [
        {"params": [p for n, p in model.named_parameters() if p.ndim >= 2], "use_muon": True, "lr": 0.01},
        {"params": [p for n, p in model.named_parameters() if p.ndim < 2], "use_muon": False, "lr": 0.01},
    ]
    optimizer = MuSGD(groups, lr=0.01, momentum=0.937, nesterov=True, muon=0.2, sgd=1.0)
    results["optimizer_step"] = timeit(lambda: optimizer.step(), device, runs=args.runs)

    with torch.no_grad():
        feats = model(images)["one2many"]["feats"]
    anchors, strides = make_anchors(feats, model.model[-1].stride, 0.5)
    pd_scores = torch.rand(args.batch, args.nc, anchors.shape[0], device=device).sigmoid()
    pd_boxes = torch.rand(args.batch, 4, anchors.shape[0], device=device) * 100
    gt = torch.tensor(
        [[0.0, 40.0, 40.0, 140.0, 130.0], [5.0, 20.0, 20.0, 70.0, 90.0]], device=device
    ).expand(args.batch, 2, 5).contiguous()
    mask_gt = (gt[..., 1:].sum(-1, keepdim=True) > 0).float()
    assigner = criterion.one2many.assigner
    with torch.no_grad():
        results["assigner"] = timeit(
            lambda: assigner(
                pd_scores.permute(0, 2, 1),
                pd_boxes.permute(0, 2, 1) * strides,
                anchors * strides,
                gt[..., :1],
                gt[..., 1:],
                mask_gt,
            ),
            device,
            runs=args.runs,
        )

    print(json.dumps(results, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
