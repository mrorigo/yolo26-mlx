"""Loss and label-assignment tests, including the YOLO26-specific behaviours."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from yolo26_mlx import build_model
from yolo26_mlx.loss import DetectionLoss, E2EDetectLoss, LossWeights, TaskAlignedAssigner
from yolo26_mlx.nn.ops import bbox2dist, bbox_iou, dist2bbox, make_anchors


@pytest.fixture(scope="module")
def model_and_batch():
    model = build_model("yolo26", "n", nc=4, imgsz=64, verbose=False)
    model.build_strides(64)
    model.train(True)
    images = mx.random.normal((2, 64, 64, 3))
    preds = model(images)
    batch = {
        "batch_idx": mx.array([0.0, 0.0, 1.0]),
        "cls": mx.array([0.0, 3.0, 2.0]),
        "bboxes": mx.array([[0.4, 0.4, 0.3, 0.2], [0.7, 0.6, 0.1, 0.1], [0.5, 0.5, 0.4, 0.4]]),
    }
    return model, preds, batch


def test_loss_is_finite_and_differentiable(model_and_batch):
    model, preds, batch = model_and_batch
    images = mx.random.normal((2, 64, 64, 3))
    loss, items = E2EDetectLoss(model, LossWeights())(preds, batch)
    assert np.isfinite(float(loss))
    assert set(items) == {"box_loss", "cls_loss", "l1_loss"}
    assert all(v > 0 for v in items.values())

    params = model.parameters()

    def loss_fn(p):
        model.update(p)
        preds = model(images)
        return E2EDetectLoss(model, LossWeights())(preds, batch)[0]

    _, grads = mx.value_and_grad(loss_fn)(params)
    total = sum(float(mx.sum(g)) for g in grads.values())
    assert np.isfinite(total) and total != 0.0


def test_dfl_free_head_has_no_dfl_term(model_and_batch):
    model, _preds, _batch = model_and_batch
    loss = DetectionLoss(model, LossWeights())
    assert loss.use_dfl is False
    assert loss.loss_names[-1] == "l1_loss"  # L1 on ltrb instead of a distribution loss


def test_progressive_loss_decays_one_to_many_weight():
    hyp = LossWeights(epochs=10)
    model = build_model("yolo26", "n", nc=2, imgsz=64, verbose=False)
    model.build_strides(64)
    crit = E2EDetectLoss(model, hyp)
    assert crit.o2m == pytest.approx(0.8)
    assert crit.o2o == pytest.approx(0.2)
    seen = [crit.o2m]
    for _ in range(9):
        crit.update()
        seen.append(crit.o2m)
    assert seen[0] > seen[-1]
    assert seen[-1] == pytest.approx(hyp.final_o2m, abs=0.05)
    assert crit.o2m + crit.o2o == pytest.approx(1.0, abs=1e-6)


def test_assigner_respects_small_target_floor():
    """STAL floors tiny boxes at the second level's stride so they keep candidate anchors."""
    assigner = TaskAlignedAssigner(topk=5, num_classes=3, stride=[8.0, 16.0, 32.0])
    anchors = mx.array([[16.5, 16.5], [24.5, 24.5]])
    gt = mx.array([[[20.0, 20.0, 24.0, 24.0]]])  # 4x4 box: no anchor centre inside
    mask = mx.ones((1, 1, 1))
    # the STAL floor grows it to 16x16 (the second level's stride), which now covers both anchors
    assert np.array(assigner.select_candidates_in_gts(anchors, gt, mask)).all()
    # without the floor (stride_val = 0) neither anchor qualifies
    assigner.stride_val = 0.0
    assert not np.array(assigner.select_candidates_in_gts(anchors, gt, mask)).any()


def test_assigner_assigns_at_most_one_gt_per_anchor():
    assigner = TaskAlignedAssigner(topk=3, num_classes=2, stride=[8.0, 16.0, 32.0])
    anchors = mx.array([[16.5, 16.5], [24.5, 24.5], [8.5, 8.5]])
    scores = mx.random.uniform(shape=(1, 3, 2))
    boxes = mx.array([[[10.0, 10.0, 30.0, 30.0], [20.0, 20.0, 40.0, 40.0], [0.0, 0.0, 16.0, 16.0]]])
    gt = mx.array([[[10.0, 10.0, 34.0, 34.0], [18.0, 18.0, 44.0, 44.0]]])
    mask = mx.ones((1, 2, 1))
    _, target_scores, fg, idx = assigner(scores, boxes, anchors, mx.array([[[0.0], [1.0]]]), gt, mask)
    assert int(np.array(fg).sum()) <= 3
    assert np.array(idx)[np.array(fg)].max() < 2  # every positive points at a real ground truth
    assert np.array(target_scores).sum() > 0


def test_assigner_handles_empty_batch():
    assigner = TaskAlignedAssigner(topk=5, num_classes=2, stride=[8.0, 16.0, 32.0])
    scores = mx.zeros((2, 10, 2))
    boxes = mx.zeros((2, 10, 4))
    anchors = mx.zeros((10, 2))
    out = assigner(scores, boxes, anchors, mx.zeros((2, 0, 1)), mx.zeros((2, 0, 4)), mx.zeros((2, 0, 1)))
    assert int(np.array(out[2]).sum()) == 0
    assert np.array(out[1]).sum() == 0.0


def test_iou_and_dist_helpers():
    a = mx.array([[0.0, 0.0, 10.0, 10.0]])
    b = mx.array([[0.0, 0.0, 10.0, 10.0]])
    assert float(bbox_iou(a, b, xywh=False)) == pytest.approx(1.0, abs=1e-5)
    assert float(bbox_iou(a, mx.array([[20.0, 20.0, 30.0, 30.0]]), xywh=False)) == pytest.approx(0.0, abs=1e-5)
    # dist -> box -> dist round trip
    anchors = mx.array([[4.0, 4.0]])
    dist = mx.array([[1.0, 2.0, 3.0, 4.0]])
    box = dist2bbox(dist, anchors)
    assert np.allclose(np.array(box), [[3.0, 2.0, 7.0, 8.0]])
    assert np.allclose(np.array(bbox2dist(anchors, box)), np.array(dist))


def test_make_anchors_are_in_grid_units():
    feats = [mx.zeros((1, 4, 4, 8)), mx.zeros((1, 2, 2, 16))]
    anchors, strides = make_anchors(feats, [8.0, 16.0], 0.5)
    arr = np.array(anchors)
    assert arr.shape == (20, 2)
    assert arr[0].tolist() == [0.5, 0.5]  # not multiplied by the stride
    assert arr[16].tolist() == [0.5, 0.5]  # the second level restarts its own grid
    assert np.array(strides).reshape(-1).tolist() == [8.0] * 16 + [16.0] * 4
