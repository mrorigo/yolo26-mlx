"""Dataset, augmentation, metrics and end-to-end training tests."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from conftest import Hyp

from yolo26_mlx import build_model
from yolo26_mlx.data import DetectionDataset, YOLOBatchLoader, letterbox, mosaic4, random_affine
from yolo26_mlx.metrics import ap_per_class, mean_average_precision, non_max_suppression
from yolo26_mlx.train import TrainConfig, Trainer, validate


# --------------------------------------------------------------------------- data
def test_dataset_shapes_and_labels(shapes_root):
    ds = DetectionDataset(shapes_root, imgsz=64, augment=False, image_dir="images/train", label_dir="labels/train")
    image, target = ds[0]
    assert image.shape == (64, 64, 3) and image.dtype == np.uint8
    assert target.shape == (1, 5)  # [cls, x1, y1, x2, y2]
    # the label is centred where the square was drawn (square 0 is centred at (18, 32), side 14)
    assert target[0, 0] == 0.0
    assert 10 < target[0, 1] < 12 and 24 < target[0, 2] < 26  # x1, y1
    assert 24 < target[0, 3] < 26 and 38 < target[0, 4] < 40  # x2, y2


def test_letterbox_preserves_aspect_and_boxes():
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    target = np.array([[0.0, 20.0, 10.0, 60.0, 30.0]], dtype=np.float32)
    out, boxes = letterbox(image, target, (64, 64))
    assert out.shape == (64, 64, 3)
    # scale = 64/80 = 0.8; the resized image is 64x32, so 16 rows of padding top and bottom
    assert np.allclose(boxes[0, 1:], [20 * 0.8, 10 * 0.8 + 16, 60 * 0.8, 30 * 0.8 + 16], atol=0.6)
    assert boxes[0, 1] >= 0 and boxes[0, 3] <= 64


def test_mosaic_keeps_boxes_inside_canvas(shapes_root):
    ds = DetectionDataset(shapes_root, imgsz=64, augment=False, image_dir="images/train", label_dir="labels/train")
    images, targets = zip(*[ds[i] for i in range(4)], strict=True)
    canvas, merged = mosaic4([i for i in images], [t for t in targets], (64, 64))
    assert canvas.shape == (64, 64, 3)
    if len(merged):
        assert merged[:, 1:].min() >= 0 and merged[:, 3:].max() <= 64


def test_affine_keeps_boxes_consistent():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[20:40, 20:40] = 255
    target = np.array([[0.0, 20.0, 20.0, 40.0, 40.0]], dtype=np.float32)
    np.random.seed(0)
    for _ in range(5):
        out, boxes = random_affine(image, target.copy(), degrees=10, translate=0.1, scale=0.2, fliplr=0.5)
        assert out.shape == image.shape
        if len(boxes):
            assert boxes[0, 3] > boxes[0, 1] and boxes[0, 4] > boxes[0, 2]


def test_batches_are_normalised(shapes_root):
    ds = DetectionDataset(
        shapes_root, imgsz=64, augment=True, hyp=Hyp(), image_dir="images/train", label_dir="labels/train"
    )
    for images, batch in YOLOBatchLoader(ds, batch_size=2, shuffle=False):
        assert images.shape == (2, 64, 64, 3)
        assert float(images.min()) >= 0.0 and float(images.max()) <= 1.0
        assert set(batch) == {"batch_idx", "cls", "bboxes"}
        assert batch["bboxes"].shape[1] == 4
        assert float(batch["bboxes"].max()) <= 1.0
        break


# ----------------------------------------------------------------------- metrics
def test_nms_removes_overlaps():
    pred = np.array(
        [[[20.0, 20.0, 20.0, 20.0, 0.9, 0.0], [21.0, 21.0, 20.0, 20.0, 0.8, 0.0], [5.0, 5.0, 10.0, 10.0, 0.7, 0.0]]]
    )
    out = non_max_suppression(pred, conf_thres=0.001, iou_thres=0.5)[0]
    assert len(out) == 2  # the 0.8 box is suppressed by the 0.9 one
    assert out[0, 4] == pytest.approx(0.9)


def test_ap_is_one_for_perfect_detections():
    labels = [np.array([[0.0, 10.0, 10.0, 50.0, 50.0]])]
    dets = [np.array([[10.0, 10.0, 50.0, 50.0, 0.9, 0.0]])]
    ap, present = ap_per_class(dets, labels, 1)
    assert ap[0, 0] == pytest.approx(1.0)
    assert mean_average_precision(ap, present) == pytest.approx((1.0, 1.0))


def test_ap_drops_with_missed_detections():
    labels = [np.array([[0.0, 10.0, 10.0, 50.0, 50.0], [0.0, 100.0, 100.0, 140.0, 140.0]])]
    dets = [np.array([[10.0, 10.0, 50.0, 50.0, 0.9, 0.0]])]
    ap, _present = ap_per_class(dets, labels, 1)
    # one of two objects is found: recall 0.5, and 101-point interpolation lands just above it
    assert 0.49 < ap[0, 0] < 0.52


# --------------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def overfit_model(tmp_path_factory, shapes_root):
    """Train yolo26n on four images long enough to fit them; shared by the tests below."""
    import shutil

    data = tmp_path_factory.mktemp("overfit-data")
    shutil.copytree(shapes_root, data / "dataset", dirs_exist_ok=True)
    cfg = TrainConfig(
        data=str(data / "dataset"),
        imgsz=64,
        epochs=200,
        batch=4,
        nbs=4,
        nc=1,
        warmup_epochs=2.0,
        close_mosaic=0,
        mosaic=0.0,
        fliplr=0.0,
        translate=0.05,
        scale_aug=0.2,
        hsv_s=0.0,
        hsv_v=0.0,
        lr0=0.02,
        verbose=False,
        name="overfit",
        project=str(data),
    )
    trainer = Trainer(cfg)
    trainer.train()
    return trainer, cfg


def test_training_fits_a_tiny_dataset(overfit_model):
    trainer, _cfg = overfit_model
    # the model must clearly learn. Compare the per-term criterion values, not the total: the
    # total is scaled by the Progressive Loss schedule, which shrinks the one-to-many weight over
    # training and would make a cross-epoch comparison meaningless.
    assert trainer.history[-1]["cls_loss"] < 0.5 * trainer.history[0]["cls_loss"]
    assert trainer.history[-1]["box_loss"] < 0.8 * trainer.history[0]["box_loss"]
    assert max(r["map50"] for r in trainer.history) > 0.6
    assert max(r["map"] for r in trainer.history) > 0.3
    # losses must be finite all the way through
    assert all(np.isfinite(row["loss"]) for row in trainer.history)
    # Progressive Loss moves weight from one-to-many to one-to-one
    assert trainer.history[0]["o2m"] > trainer.history[-1]["o2m"]


def test_both_heads_validate(overfit_model):
    """The NMS path (one-to-many) and the NMS-free path (one-to-one) must both work."""
    trainer, cfg = overfit_model
    dataset = DetectionDataset(cfg.data, imgsz=cfg.imgsz, augment=False, image_dir="images/val", label_dir="labels/val")
    with_nms = validate(trainer.model, dataset, 1, conf=0.05, nms=True)
    without_nms = validate(trainer.model, dataset, 1, conf=0.05, nms=False)
    # the one-to-many head is what the long schedule trains hardest, so hold it to a higher bar;
    # the one-to-one branch only has to be functional (a handful of hits) at this budget
    assert with_nms["map50"] > 0.6, "one-to-many head + NMS"
    assert without_nms["map50"] > 0.2, "one-to-one head, no NMS pass at all"
    trainer.model.head.end2end = False  # restore the default


def test_checkpoint_roundtrip(overfit_model, tmp_path):
    trainer, _ = overfit_model
    path = tmp_path / "ckpt.safetensors"
    trainer.model.save_weights(str(path))
    fresh = build_model("yolo26", "n", nc=1, imgsz=64, verbose=False)
    fresh.build_strides(64)
    fresh.load_weights(str(path))
    fresh.train(False)
    trainer.model.train(False)
    fresh.head.end2end = trainer.model.head.end2end
    x = mx.random.normal((1, 64, 64, 3))
    assert mx.allclose(fresh(x), trainer.model(x), atol=1e-6).item()
