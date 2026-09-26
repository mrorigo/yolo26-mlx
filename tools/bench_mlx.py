"""Benchmark the MLX implementation: inference, training step, loss and optimizer.

    .venv/bin/python tools/bench_mlx.py --scale n --imgsz 640 --batch 1

Companion to ``tools/bench_torch.py``; both measure the same work with the same weights.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from yolo26_mlx import build_model
from yolo26_mlx.loss import E2EDetectLoss, LossWeights
from yolo26_mlx.optim.musgd import MuSGD, muon_param_groups


def timeit(fn, warmup: int = 5, runs: int = 20) -> dict:
    """Median wall time in milliseconds, plus mean and the measured device peak memory."""
    for _ in range(warmup):
        mx.eval(fn())
    mx.reset_peak_memory()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        mx.eval(fn())
        samples.append((time.perf_counter() - t0) * 1e3)
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "peak_mb": mx.get_peak_memory() / 2**20,
    }


class Hyp:
    lr0 = 0.01
    weight_decay = 5e-4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="n")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--nc", type=int, default=80)
    ap.add_argument("--refdir", default="/var/folders/7z/v9jlq_zj4jn5n04xjqkwr5400000gn/T/opencode")
    ap.add_argument("--json", default=None, help="write the results to this file")
    ap.add_argument("--compile", action="store_true", help="wrap the forward passes in mx.compile")
    args = ap.parse_args()

    model = build_model("yolo26", args.scale, nc=args.nc, imgsz=args.imgsz, verbose=False)
    weights = Path(args.refdir) / f"ref_{args.scale}_{args.imgsz}.safetensors"
    if weights.exists():
        model.load_weights(str(weights))
    model.build_strides(args.imgsz)
    model.head.bias_init()

    images = mx.random.normal((args.batch, args.imgsz, args.imgsz, 3))
    batch = {
        "batch_idx": mx.array([0.0, 0.0, 0.0] * args.batch),
        "cls": mx.array([0.0, 5.0, 12.0] * args.batch),
        "bboxes": mx.repeat(
            mx.array([[0.4, 0.4, 0.3, 0.2], [0.7, 0.6, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]]), args.batch, axis=0
        ),
    }

    results: dict[str, dict] = {"config": vars(args)}

    model.train(False)
    model.head.end2end = True
    results["inference_e2e"] = timeit(lambda: model(images), runs=args.runs)
    model.head.end2end = False
    results["inference_o2m"] = timeit(lambda: model(images), runs=args.runs)

    fused = model.fuse().eval()
    results["inference_fused"] = timeit(lambda: fused(images), runs=args.runs)
    if args.compile:  # mx.compile fuses the per-layer graph; eager mode pays dispatch per op
        results["inference_compiled"] = timeit(lambda: model.compiled()(images), runs=args.runs)
        results["inference_fused_compiled"] = timeit(lambda: fused.compiled()(images), runs=args.runs)

    model.train(True)
    criterion = E2EDetectLoss(model, LossWeights())
    params = model.parameters()
    optimizer = MuSGD(params, muon_param_groups(model, Hyp()), lr=0.01)

    def fwd_loss(_params=None):
        if _params is not None:
            model.update(_params)
        return criterion(model(images), batch)[0]

    results["loss_only"] = timeit(fwd_loss, runs=args.runs)

    def fwd_loss_grad():
        loss, grads = mx.value_and_grad(fwd_loss)(params)
        return loss

    results["train_step"] = timeit(fwd_loss_grad, runs=args.runs)

    loss, grads = mx.value_and_grad(fwd_loss)(params)
    mx.eval(loss)
    results["optimizer_step"] = timeit(lambda: optimizer.step(grads), runs=args.runs)

    anchors = None
    from yolo26_mlx.nn.ops import make_anchors

    feats = model._forward_to_head(images)
    anchors, strides = make_anchors(feats, model.stride, 0.5)

    pd_scores = mx.random.uniform(shape=(args.batch, anchors.shape[0], args.nc))
    pd_boxes = mx.random.uniform(shape=(args.batch, anchors.shape[0], 4)) * 100
    gt = mx.broadcast_to(mx.array([[0.0, 40.0, 40.0, 140.0, 130.0], [5.0, 20.0, 20.0, 70.0, 90.0]]), (args.batch, 2, 5))
    mask_gt = mx.ones((*gt.shape[:2], 1))
    assigner = criterion.one2many.assigner
    results["assigner"] = timeit(
        lambda: assigner(pd_scores, pd_boxes, anchors * strides, gt[..., :1], gt[..., 1:], mask_gt), runs=args.runs
    )

    print(json.dumps(results, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
