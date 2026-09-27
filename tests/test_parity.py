"""Numerical parity against the PyTorch reference implementation.

These tests need the data produced by ``tools/export_reference.py`` (which is the only place
PyTorch appears in this project). They are skipped when that data is not present, so the suite
still runs on a clean checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from yolo26_mlx import build_model
from yolo26_mlx.loss import DetectionLoss, E2EDetectLoss, LossWeights
from yolo26_mlx.nn.ops import make_anchors

# where tools/export_reference.py writes its tensors; override with YOLO26_REF_DIR
REF_DIR = Path(os.environ.get("YOLO26_REF_DIR", "/Users/origo/.cache/yolo26-mlx/ref"))


def _require() -> Path:
    path = REF_DIR / "ref_n_256.safetensors"
    if not path.exists():
        pytest.skip("reference data missing; run tools/export_reference.py --scale n --imgsz 256")
    return path


@pytest.fixture(scope="module")
def reference():
    """Load the exported reference weights, inputs and targets."""
    _require()

    def load(name: str) -> np.ndarray:
        return np.load(REF_DIR / name)

    model = build_model("yolo26", "n", imgsz=256, verbose=False)
    model.load_weights(str(REF_DIR / "ref_n_256.safetensors"))
    model.build_strides(256)
    return model, load


def test_branch_outputs_match(reference):
    model, load = reference
    x = mx.array(load("ref_x_n_256.npy").transpose(0, 2, 3, 1))  # NCHW -> NHWC
    model.train(False)
    feats = model._forward_to_head(x)
    head = model.head
    for branch, stem in (("one2many", "o2m"), ("one2one", "o2o")):
        towers = head.one2many if branch == "one2many" else head.one2one
        preds = head.forward_head(feats, **towers)
        assert np.allclose(np.array(preds["boxes"]).transpose(0, 2, 1), load(f"ref_{stem}_boxes_n_256.npy"), atol=1e-3)
        assert np.allclose(
            np.array(preds["scores"]).transpose(0, 2, 1), load(f"ref_{stem}_scores_n_256.npy"), atol=1e-3
        )


def test_decoded_outputs_match(reference):
    model, load = reference
    x = mx.array(load("ref_x_n_256.npy").transpose(0, 2, 3, 1))
    model.train(False)
    model.head.end2end = False
    o2m = np.array(model(x)).transpose(0, 2, 1)
    assert np.allclose(o2m, load("ref_o2m_n_256.npy"), atol=1e-2)
    model.head.end2end = True
    e2e = np.array(model(x))
    ref_e2e = load("ref_e2e_n_256.npy")
    assert e2e.shape == ref_e2e.shape
    # top-k order is ambiguous where scores tie, so compare the score multiset
    assert np.allclose(np.sort(e2e[0, :, 4])[::-1], np.sort(ref_e2e[0, :, 4])[::-1], atol=1e-4)


def test_assigner_matches(reference):
    model, load = reference
    x = mx.array(load("ref_x2_n_256.npy").transpose(0, 2, 3, 1))
    model.train(True)
    preds = model(x)["one2many"]
    loss = DetectionLoss(model, LossWeights())
    targets = load("ref_targets_n_256.npy")
    padded = loss.preprocess(
        mx.concatenate(
            [mx.array(targets[:, 0])[:, None], mx.array(targets[:, 1])[:, None], mx.array(targets[:, 2:])], axis=-1
        ),
        2,
        mx.array([256.0, 256.0]),
    )
    assert np.allclose(np.array(padded), load("ref_padded_n_256.npy"), atol=1e-3)
    anchors, strides = make_anchors(preds["feats"], model.stride, 0.5)
    assert np.allclose(np.array(anchors * strides), load("ref_anchors_n_256.npy"), atol=1e-3)
    mask_gt = mx.expand_dims(mx.sum(padded[..., 1:], axis=-1) > 0, -1)
    inside = loss.assigner.select_candidates_in_gts(anchors * strides, padded[..., 1:], mask_gt)
    assert np.array_equal(np.array(inside), load("ref_maskin_n_256.npy").astype(bool))
    # the assigner consumes decoded boxes in pixel units, not raw distances
    decoded = loss.bbox_decode(anchors, preds["boxes"]) * strides
    align, overlaps = loss.assigner.get_box_metrics(
        mx.sigmoid(preds["scores"]), decoded, padded[..., :1], padded[..., 1:], inside * mask_gt
    )
    assert np.allclose(np.array(align), load("ref_align_n_256.npy"), atol=1e-6)
    assert np.allclose(np.array(overlaps), load("ref_overlaps_n_256.npy"), atol=1e-4)
    mask_pos = loss.assigner.select_topk_candidates(align, topk_mask=mask_gt > 0) * inside * (mask_gt > 0)
    _, _, mask_pos2 = loss.assigner.select_highest_overlaps(mask_pos, overlaps, align)
    assert np.array_equal(np.array(mask_pos2) > 0, load("ref_maskpos2_n_256.npy") > 0)


def test_losses_match(reference):
    model, load = reference
    x = mx.array(load("ref_x2_n_256.npy").transpose(0, 2, 3, 1))
    model.train(True)
    preds = model(x)
    targets = load("ref_targets_n_256.npy")
    batch = {
        "batch_idx": mx.array(targets[:, 0], dtype=mx.float32),
        "cls": mx.array(targets[:, 1], dtype=mx.float32),
        "bboxes": mx.array(targets[:, 2:]),
    }
    hyp = LossWeights()
    for name, (loss, _), ref_file in (
        ("one-to-many", DetectionLoss(model, hyp)(preds["one2many"], batch), "ref_loss_o2m_n_256.npy"),
        ("progressive", E2EDetectLoss(model, hyp)(preds, batch), "ref_loss_e2e_n_256.npy"),
    ):
        ref = load(ref_file)
        assert float(loss.sum()) == pytest.approx(float(ref[0]), rel=1e-3), name
