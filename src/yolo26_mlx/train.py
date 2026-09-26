"""Training and validation loops for YOLO26-MLX (MuSGD + Progressive Loss)."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from .data.dataset import DetectionDataset, YOLOBatchLoader
from .loss.detect import E2EDetectLoss, LossWeights
from .metrics import ap_per_class, detections_to_numpy, mean_average_precision, non_max_suppression
from .optim.musgd import MuSGD, muon_param_groups
from .tasks import build_model

__all__ = ["TrainConfig", "Trainer", "validate"]


@dataclass
class TrainConfig:
    """Training hyperparameters (YOLO26 recipe defaults)."""

    model: str = "yolo26"
    scale: str = "n"
    nc: int | None = None  # override the class count from the model YAML
    data: str = "coco8"
    imgsz: int = 640
    epochs: int = 100
    batch: int = 16
    workers: int = 2  # loader threads; augmentation is GIL-bound, so 2 is the sweet spot
    optimizer: str = "MuSGD"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 5e-4
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    cos_lr: bool = False
    box: float = 7.5
    cls: float = 0.5
    l1: float = 1.5
    nbs: int = 64  # nominal batch size the gains were tuned for
    mosaic: float = 1.0
    close_mosaic: int = 10
    mixup: float = 0.0
    degrees: float = 0.0
    translate: float = 0.1
    scale_aug: float = 0.5
    shear: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    conf: float = 0.001
    iou: float = 0.7
    max_det: int = 300
    nms: bool = True  # False selects the NMS-free one-to-one head
    patience: int = 100
    seed: int = 0
    device: str = "gpu"
    project: str = "runs/detect"
    name: str = "yolo26n"
    pretrained: str | None = None
    verbose: bool = True
    extra: dict = field(default_factory=dict)

    def loss_weights(self) -> LossWeights:
        return LossWeights(box=self.box, cls=self.cls, l1=self.l1, epochs=self.epochs)


class Trainer:
    """Minimal but complete trainer: MuSGD, warmup + linear/cosine LR, EMA-free, checkpointed."""

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        mx.random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)  # the augmentation pipeline draws from Python's RNG
        self.save_dir = Path(cfg.project) / cfg.name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model = build_model(
            cfg.model, cfg.scale, nc=cfg.nc, imgsz=cfg.imgsz, pretrained=cfg.pretrained, verbose=cfg.verbose
        )
        self.nc = self.model.head.nc
        self.criterion = E2EDetectLoss(self.model, cfg.loss_weights())
        self.params = self.model.parameters()
        self.optimizer = MuSGD(
            self.params,
            muon_param_groups(self.model, cfg),
            lr=cfg.lr0,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
            muon=0.2,
            sgd=1.0,
        )
        self.accumulate = max(round(cfg.nbs / max(cfg.batch, 1)), 1)
        self._grads: dict[str, mx.array] | None = None
        self._micro = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------------ schedules
    def lr_at(self, epoch: int, step: int, steps_per_epoch: int) -> float:
        """Linear or cosine decay from ``lr0`` to ``lr0 * lrf``, with momentum and bias warmup."""
        cfg = self.cfg
        if epoch < cfg.warmup_epochs:
            warm = (epoch + step / max(steps_per_epoch, 1)) / max(cfg.warmup_epochs, 1e-6)
            return cfg.warmup_bias_lr * (1 - warm) + cfg.lr0 * warm
        t = (epoch - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1e-6)
        t = min(max(t, 0.0), 1.0)
        factor = (1 - math.cos(math.pi * t)) / 2 if cfg.cos_lr else 1 - t
        return cfg.lr0 * (cfg.lrf + (1 - cfg.lrf) * factor)

    def momentum_at(self, epoch: int, step: int, steps_per_epoch: int) -> float:
        cfg = self.cfg
        if epoch < cfg.warmup_epochs:
            warm = (epoch + step / max(steps_per_epoch, 1)) / max(cfg.warmup_epochs, 1e-6)
            return cfg.warmup_momentum * (1 - warm) + cfg.momentum * warm
        return cfg.momentum

    # ---------------------------------------------------------------------- train
    def train(self) -> list[dict]:
        cfg = self.cfg
        val_set = self._dataset(split="val", augment=False, mosaic=0.0)
        steps = math.ceil(len(self._dataset(split="train", augment=True, mosaic=cfg.mosaic)) / cfg.batch)
        steps = max(steps, 1)
        self.model.train(True)
        for epoch in range(cfg.epochs):
            mosaic = cfg.mosaic if epoch < cfg.epochs - cfg.close_mosaic else 0.0
            # rebuild the loader each epoch: its worker threads capture the augmentation gains
            # (close_mosaic changes them), and the index order is drawn when iteration starts
            train_set = self._dataset(split="train", augment=True, mosaic=mosaic)
            loader = YOLOBatchLoader(train_set, batch_size=cfg.batch, shuffle=True, workers=cfg.workers)
            self.criterion.update()  # Progressive Loss: one-to-many weight decays
            self.model.train(True)
            t0, total, seen, lr = time.time(), 0.0, 0, cfg.lr0
            items: dict[str, float] = {"box_loss": 0.0, "cls_loss": 0.0, "l1_loss": 0.0}
            for step, (images, batch) in enumerate(loader):
                lr = self.lr_at(epoch, step, steps)
                warmup = min((epoch + step / max(steps, 1)) / max(self.cfg.warmup_epochs, 1e-6), 1.0)
                self._set_lr(lr, self.momentum_at(epoch, step, steps), warmup)
                loss, items = self._step(images, batch)
                total += float(loss) / max(steps, 1)
                seen += 1
            self.flush_gradients()
            self.model.train(False)
            metrics = validate(self.model, val_set, self.nc, conf=cfg.conf, iou=cfg.iou, nms=cfg.nms)
            row = {
                "epoch": epoch + 1,
                "lr": lr,
                "o2m": self.criterion.o2m,
                "loss": total / max(seen, 1),
                **{k: v for k, v in items.items()},
                **{k: v for k, v in metrics.items()},
                "time": time.time() - t0,
            }
            self.history.append(row)
            self._log(row)
            self.save(epoch)
        return self.history

    def _step(self, images: mx.array, batch: dict) -> tuple[float, dict]:
        """One micro-batch: accumulate its gradient and step once every ``nbs / batch`` micro-batches.

        The losses are tuned for a nominal batch of ``nbs`` images, so gradients are averaged over
        that many micro-batches before the optimizer sees them.
        """
        items: dict[str, float] = {}

        def loss_fn(params: dict[str, mx.array]) -> mx.array:
            nonlocal items
            self.model.update(params)
            loss, items = self.criterion(self.model(images), batch)
            return loss / self.accumulate

        loss, grads = mx.value_and_grad(loss_fn)(self.params)
        self._micro += 1
        if self._grads is None:
            self._grads = {k: mx.zeros_like(v) for k, v in grads.items()}
        for key, grad in grads.items():
            self._grads[key] += grad
        if self._micro >= self.accumulate:
            self.optimizer.step(self._grads)
            self._grads, self._micro = None, 0
        return loss * self.accumulate, items

    def flush_gradients(self) -> None:
        """Apply any partially accumulated gradients (end of an epoch)."""
        if self._grads is not None:
            self.optimizer.step(self._grads)
            self._grads, self._micro = None, 0

    def _set_lr(self, lr: float, momentum: float, warmup: float = 1.0) -> None:
        """Set the per-group learning rate; biases and norms warm up from ``warmup_bias_lr``."""
        bias_lr = self.cfg.warmup_bias_lr * (1 - warmup) + lr * warmup
        for group in self.optimizer.groups:
            group["momentum"] = momentum
            group["lr"] = bias_lr if group.get("param_group") == "norm" else lr

    def _dataset(self, split: str, augment: bool, mosaic: float = 0.0) -> DetectionDataset:
        cfg = self.cfg
        root = Path(cfg.data)
        image_dir, label_dir = "images", "labels"
        if (root / image_dir / split).is_dir():
            image_dir, label_dir = f"{image_dir}/{split}", f"{label_dir}/{split}"
        return DetectionDataset(
            root,
            imgsz=cfg.imgsz,
            hyp=_with_mosaic(cfg, mosaic if augment else 0.0),
            augment=augment,
            image_dir=image_dir,
            label_dir=label_dir,
        )

    # ----------------------------------------------------------------- checkpoint
    def save(self, epoch: int) -> None:
        self.model.save_weights(str(self.save_dir / "last.safetensors"), {"epoch": str(epoch + 1)})
        (self.save_dir / "config.json").write_text(repr(self.cfg))

    def _log(self, row: dict) -> None:
        if not self.cfg.verbose:
            return
        print(
            f"epoch {row['epoch']:>3}/{self.cfg.epochs}  loss {row['loss']:.4f}  "
            f"box {row['box_loss']:.4f}  cls {row['cls_loss']:.4f}  l1 {row['l1_loss']:.4f}  "
            f"o2m {row['o2m']:.2f}  mAP50-95 {row['map']:.4f}  mAP50 {row['map50']:.4f}  "
            f"({row['time']:.1f}s)"
        )


def _with_mosaic(cfg: TrainConfig, mosaic: float):
    class _Hyp:
        pass

    hyp = _Hyp()
    for key in (
        "mosaic",
        "degrees",
        "translate",
        "scale",
        "shear",
        "flipud",
        "fliplr",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "mixup",
    ):
        value = mosaic if key == "mosaic" else getattr(cfg, "scale_aug" if key == "scale" else key)
        setattr(hyp, key, value)
    return hyp


def validate(
    model,
    dataset: DetectionDataset,
    num_classes: int,
    conf: float = 0.001,
    iou: float = 0.7,
    nms: bool = True,
    batch_size: int = 8,
) -> dict[str, float]:
    """Run the model over a dataset and return mAP@0.5:0.95 / mAP@0.5 / precision / recall.

    Args:
        nms: True runs the one-to-many head through NMS; False uses the NMS-free one-to-one head.
    """
    model.train(False)
    model.head.end2end = not nms
    loader = YOLOBatchLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    detections: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    imgsz = dataset.imgsz
    for images, batch in loader:
        out = model(images)
        if nms:
            detections.extend(non_max_suppression(np.array(out), conf_thres=conf, iou_thres=iou))
        else:
            detections.extend(detections_to_numpy(out))
        labels.extend(_labels_from_batch(batch, imgsz, int(images.shape[0])))
    ap, present = ap_per_class(detections, labels, num_classes)
    map_, map50 = mean_average_precision(ap, present)
    return {"map": map_, "map50": map50}


def _labels_from_batch(batch: dict, imgsz: int, batch_size: int) -> list[np.ndarray]:
    """Normalised batch targets -> one (M, 5) [cls, x1, y1, x2, y2] array per image, in pixels."""
    image_ids = np.array(batch["batch_idx"], dtype=int)
    cls = np.array(batch["cls"])
    boxes = np.array(batch["bboxes"]) * imgsz
    xyxy = np.stack(
        [
            boxes[:, 0] - boxes[:, 2] / 2,
            boxes[:, 1] - boxes[:, 3] / 2,
            boxes[:, 0] + boxes[:, 2] / 2,
            boxes[:, 1] + boxes[:, 3] / 2,
        ],
        axis=-1,
    )
    rows = np.concatenate([cls[:, None], xyxy], axis=1)
    return [rows[image_ids == i] for i in range(batch_size)]
