"""Print the MLX-vs-PyTorch benchmark comparison.

Runs (or reuses) the JSON files written by ``tools/bench_mlx.py`` and ``tools/bench_torch.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STAGES = [
    ("inference_e2e", "inference, NMS-free head"),
    ("inference_o2m", "inference, one-to-many head"),
    ("inference_fused", "inference, BN folded"),
    ("inference_compiled", "inference, mx.compile"),
    ("inference_fused_compiled", "inference, folded + mx.compile"),
    ("inference_e2e_compiled", "inference, mx.compile, NMS-free"),
    ("loss_only", "criterion (fwd)"),
    ("train_step", "train step (fwd+loss+bwd)"),
    ("optimizer_step", "MuSGD step"),
    ("assigner", "TAL assigner"),
]


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/yolo26-bench")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    root = Path(args.dir)
    root.mkdir(parents=True, exist_ok=True)

    cases = [("n", 256, 1), ("n", 256, 8), ("n", 640, 1), ("n", 640, 8), ("s", 640, 1)]
    print(f"{'case':<14}{'stage':<28}{'mlx ms':>10}{'torch ms':>10}{'speedup':>10}")
    print("-" * 72)
    for scale, imgsz, batch in cases:
        tag = f"{scale}_{imgsz}_b{batch}"
        mlx = load(root / f"mlx_{tag}_{args.device}.json")
        torch = load(root / f"torch_{tag}_{args.device}.json")
        if not mlx or not torch:
            print(f"{tag:<14}(missing results: mlx={bool(mlx)} torch={bool(torch)})")
            continue
        for key, label in STAGES:
            if key not in mlx or key not in torch:
                continue
            a, b = mlx[key]["median_ms"], torch[key]["median_ms"]
            print(f"{tag:<14}{label:<28}{a:>10.2f}{b:>10.2f}{b / a:>9.2f}x")
        print()


if __name__ == "__main__":
    main()
