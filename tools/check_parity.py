"""Compare the MLX YOLO26 forward pass against exported PyTorch reference outputs.

Run the exporter (needs torch + ultralytics) first:

    python tools/export_reference.py --scale n --imgsz 256
    python tools/check_parity.py --scale n --imgsz 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

REFDIR = Path("/Users/origo/.cache/yolo26-mlx/ref")


def compare(
    name: str, got, expected: np.ndarray, atol: float = 2e-3, rtol: float = 2e-3, nhwc_to_nchw: bool = False
) -> bool:
    got_np = np.array(got)
    if nhwc_to_nchw and got_np.ndim == 3:
        got_np = got_np.transpose(0, 2, 1)
    if got_np.shape != expected.shape:
        print(f"  {name}: SHAPE MISMATCH mlx={got_np.shape} torch={expected.shape}")
        return False
    diff = np.abs(got_np - expected)
    scale = max(np.abs(expected).max(), 1e-6)
    ok = np.allclose(got_np, expected, atol=atol, rtol=rtol)
    print(f"  {name}: max_abs_diff={diff.max():.3e} (rel {diff.max() / scale:.2e}) {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="n")
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--refdir", default=str(REFDIR))
    args = ap.parse_args()
    refdir = Path(args.refdir)
    tag = f"{args.scale}_{args.imgsz}"

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from yolo26_mlx import build_model

    model = build_model("yolo26", args.scale, imgsz=args.imgsz, verbose=False)
    model.load_weights(refdir / f"ref_{tag}.safetensors")
    model.train(False)

    x = mx.array(np.load(refdir / f"ref_x_{tag}.npy").transpose(0, 2, 3, 1))  # NCHW -> NHWC
    ok = True
    # raw per-branch predictions, computed in eval mode so BatchNorm uses running stats
    feats = model._forward_to_head(x)
    head = model.head
    preds = {
        "one2many": head.forward_head(feats, **head.one2many),
        "one2one": head.forward_head(feats, **head.one2one),
    }
    for branch, stem in (("one2many", "o2m"), ("one2one", "o2o")):
        ok &= compare(
            f"{branch} raw boxes",
            preds[branch]["boxes"],
            np.load(refdir / f"ref_{stem}_boxes_{tag}.npy"),
            nhwc_to_nchw=True,
        )
        ok &= compare(
            f"{branch} raw scores",
            preds[branch]["scores"],
            np.load(refdir / f"ref_{stem}_scores_{tag}.npy"),
            nhwc_to_nchw=True,
        )
    model.head.end2end = False
    ok &= compare(
        "one-to-many decoded (xywh)", np.array(model(x)), np.load(refdir / f"ref_o2m_{tag}.npy"), nhwc_to_nchw=True
    )
    model.head.end2end = True
    det = np.array(model(x))
    ref_det = np.load(refdir / f"ref_e2e_{tag}.npy")
    # equal-score ties make the exact row order ambiguous, so compare the score multiset
    ok = (
        compare("one-to-one top-300 conf (sorted)", np.sort(det[0, :, 4])[::-1], np.sort(ref_det[0, :, 4])[::-1]) and ok
    )
    ok = compare("one-to-one output shape", np.array(det.shape), np.array(ref_det.shape)) and ok
    # ---- loss parity (train-mode BatchNorm, so recompute in train mode on the 2-image input)
    from yolo26_mlx.loss import DetectionLoss, E2EDetectLoss, LossWeights

    x2 = mx.array(np.load(refdir / f"ref_x2_{tag}.npy").transpose(0, 2, 3, 1))
    model.train(True)
    preds = model(x2)
    targets = np.load(refdir / f"ref_targets_{tag}.npy")
    batch = {
        "batch_idx": mx.array(targets[:, 0], dtype=mx.float32),
        "cls": mx.array(targets[:, 1], dtype=mx.float32),
        "bboxes": mx.array(targets[:, 2:]),
    }
    hyp = LossWeights(epochs=100)
    for name, ours, ref_file in (
        ("one-to-many loss", DetectionLoss(model, hyp)(preds["one2many"], batch), f"ref_loss_o2m_{tag}.npy"),
        ("e2e loss", E2EDetectLoss(model, hyp)(preds, batch), f"ref_loss_e2e_{tag}.npy"),
    ):
        ref = np.load(refdir / ref_file)
        ok &= compare(f"{name} (total)", mx.array([ours[0].sum().item()]), ref[:1], rtol=5e-3, atol=5e-2)
        ok &= compare(f"{name} (box/cls/l1)", np.array(list(ours[1].values())), ref[1:], rtol=5e-3, atol=5e-2)

    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
