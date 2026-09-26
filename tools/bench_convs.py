"""Microbenchmark the convolution shapes YOLO26 actually uses, MLX vs PyTorch MPS.

    .venv/bin/python tools/bench_convs.py                 # MLX side
    refenv/bin/python tools/bench_convs.py --device mps  # PyTorch side (same JSON keys)

Writes per-shape medians so the two can be compared with tools/bench_compare.py --convs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

# (name, batch, H, W, in_ch, out_ch, kernel, groups) at 640px, from the yolo26n graph
SHAPES = [
    ("stem 3x3 s2", 1, 320, 320, 3, 16, 3, 1),
    ("backbone 3x3 s2", 1, 320, 320, 16, 32, 3, 1),
    ("P3 3x3", 1, 80, 80, 64, 64, 3, 1),
    ("P3 1x1 (C3k2)", 1, 80, 80, 192, 64, 1, 1),
    ("P4 3x3", 1, 40, 40, 128, 128, 3, 1),
    ("P5 3x3", 1, 20, 20, 256, 256, 3, 1),
    ("head 1x1 cls", 1, 80, 80, 64, 80, 1, 1),
    ("head 3x3 dw cls", 1, 80, 80, 64, 64, 3, 64),
    ("attn pe 3x3 dw", 1, 20, 20, 128, 128, 3, 128),
    ("b8 P3 3x3", 8, 80, 80, 64, 64, 3, 1),
    ("b8 P5 3x3", 8, 20, 20, 256, 256, 3, 1),
    ("b8 head 3x3 dw", 8, 80, 80, 64, 64, 3, 64),
    ("b8 attn pe 3x3 dw", 8, 20, 20, 128, 128, 3, 128),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mlx", choices=["mlx", "mps", "cpu"])
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.device == "mlx":
        import mlx.core as mx

        def run(b, h, w, ci, co, k, g, weights, x):
            w_mlx = mx.random.normal((co, k, k, ci // g))
            x_mlx = mx.random.normal((b, h, w, ci))
            pad = k // 2
            fn = lambda t: mx.conv2d(t, w_mlx, padding=pad, groups=g)  # noqa: E731
            for _ in range(5):
                mx.eval(fn(x_mlx))
            samples = []
            for _ in range(args.runs):
                t0 = time.perf_counter()
                mx.eval(fn(x_mlx))
                samples.append((time.perf_counter() - t0) * 1e3)
            return statistics.median(samples)
    else:
        import torch

        dev = args.device
        torch.manual_seed(0)

        def run(b, h, w, ci, co, k, g, weights, x):
            w_t = torch.randn(co, ci // g, k, k, device=dev)
            x_t = torch.randn(b, ci, h, w, device=dev)
            fn = lambda t: torch.nn.functional.conv2d(t, w_t, padding=k // 2, groups=g)  # noqa: E731
            for _ in range(5):
                fn(x_t)
            if dev == "mps":
                torch.mps.synchronize()
            samples = []
            for _ in range(args.runs):
                t0 = time.perf_counter()
                fn(x_t)
                if dev == "mps":
                    torch.mps.synchronize()
                samples.append((time.perf_counter() - t0) * 1e3)
            return statistics.median(samples)

    results = {}
    for name, b, h, w_, ci, co, k, g in SHAPES:
        results[name] = run(b, h, w_, ci, co, k, g, None, None)
        print(f"{name:<20}{results[name]:8.3f} ms")
    out = {"device": args.device, "convs": results}
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
