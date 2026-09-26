"""Per-layer forward profile of the real YOLO26 graph, MLX vs PyTorch MPS.

Times every layer of the model at true shapes for a given input, so the hot layers and the
size of the non-convolution work (BatchNorm, attention reshapes) are visible.

    .venv/bin/python tools/profile_layers.py --imgsz 640 --batch 8
    refenv/bin/python tools/profile_layers.py --imgsz 640 --batch 8 --device mps
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

RUNS = 20
WARMUP = 5


def _time(fn, sync, warmup=WARMUP, runs=RUNS) -> float:
    """Median wall time in ms, with the device synchronised around every iteration.

    MLX is lazy, so the callback must force evaluation (``mx.eval``) or this only times graph
    construction.
    """
    for _ in range(warmup):
        sync()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def profile_mlx(imgsz: int, batch: int, scale: str) -> list[dict]:
    """Marginal cost of each layer, from cumulative prefix timings.

    Timing layers in isolation measures dispatch, not work (on MPS that inflates a single Conv to
    60 ms), so time the graph prefix and difference: marginal[k] = t(prefix_k) - t(prefix_{k-1}).
    """
    import mlx.core as mx

    from yolo26_mlx import build_model

    model = build_model("yolo26", scale, imgsz=imgsz, verbose=False)
    model.build_strides(imgsz)
    model.train(False)
    x = mx.random.normal((batch, imgsz, imgsz, 3))

    def sync():
        mx.synchronize()

    def run_prefix(k: int):
        outs = [x]
        for i, spec in enumerate(model.specs[:k]):
            outs.append(model.model[i](model._resolve(spec, outs)))
        return outs

    # one full pass to record each layer's input shape
    shapes = []
    outs = [x]
    for spec in model.specs:
        y_in = model._resolve(spec, outs)
        shapes.append([int(v) for v in (y_in[0] if isinstance(y_in, list) else y_in).shape[1:]])
        outs.append(model.model[spec.index](y_in))
    del outs

    rows, prev = [], 0.0
    for k in range(1, len(model.specs) + 1):
        total = _time(lambda kk=k: mx.eval(run_prefix(kk)[-1]), sync)
        rows.append(
            {
                "layer": k - 1,
                "type": model.specs[k - 1].name,
                "in_shape": shapes[k - 1],
                "ms": round(total - prev, 3),
            }
        )
        prev = total
    return rows




def profile_torch(imgsz: int, batch: int, scale: str) -> list[dict]:
    """Same prefix-differencing scheme for the PyTorch reference.

    Walks the layers by their stored ``.f`` connections (``parse_model`` records them) because the
    graph is not a plain chain: Concat and Detect take lists of earlier outputs.
    """
    import torch
    import yaml
    from ultralytics.nn.tasks import DetectionModel

    cfg_path = str(Path(__file__).resolve().parents[1] / "configs/yolo26.yaml")
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["scale"] = scale
    torch.manual_seed(0)
    dev = "mps"
    model = DetectionModel(cfg, verbose=False).to(dev).eval()
    x = torch.randn(batch, 3, imgsz, imgsz, device=dev)

    def sync():
        torch.mps.synchronize()

    def run_prefix(k: int, shapes: dict | None = None):
        outs = [x]
        for i, layer in enumerate(list(model.model)[:k]):
            f = layer.f
            if isinstance(f, list):
                inp = [outs[j + 1] if j >= 0 else outs[-1] for j in f]
            else:
                inp = outs[f + 1] if f >= 0 else outs[-1]
            if shapes is not None:
                first = inp[0] if isinstance(inp, list) else inp
                shapes[i] = [int(v) for v in first.shape[1:]]
            outs.append(layer(inp))
        return outs

    shapes: dict = {}
    with torch.no_grad():
        run_prefix(len(model.model), shapes)
        rows, prev = [], 0.0
        for k in range(1, len(model.model) + 1):
            t = _time(lambda kk=k: run_prefix(kk), sync)
            rows.append(
                {
                    "layer": k - 1,
                    "type": type(model.model[k - 1]).__name__,
                    "in_shape": shapes.get(k - 1, []),
                    "ms": round(t - prev, 3),
                }
            )
            prev = t
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--scale", default="n")
    ap.add_argument("--device", default="mlx", choices=["mlx", "mps"])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows = profile_mlx(args.imgsz, args.batch, args.scale) if args.device == "mlx" else profile_torch(
        args.imgsz, args.batch, args.scale
    )
    total = sum(r["ms"] for r in rows)
    for r in rows:
        r["pct"] = round(100 * r["ms"] / total, 1)
    print(f"{'layer':>5} {'type':<12}{'input shape':<22}{'ms':>8}{'%':>7}")
    for r in rows:
        print(f"{r['layer']:>5} {r['type']:<12}{r['in_shape']!s:<22}{r['ms']:>8.3f}{r['pct']:>6.1f}%")
    print(f"{'total':>5} {'':<12}{'':<22}{total:>8.3f}")
    if args.json:
        Path(args.json).write_text(json.dumps({"device": args.device, "layers": rows, "total_ms": total}, indent=2))


if __name__ == "__main__":
    main()
