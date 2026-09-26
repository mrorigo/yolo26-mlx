"""YOLO-format dataset with mosaic, affine, HSV and flip augmentation (pure numpy/PIL + MLX)."""

from __future__ import annotations

import math
import random
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

__all__ = ["COLOURS", "DetectionDataset", "YOLOBatchLoader", "augment_hsv", "letterbox", "mosaic4", "random_affine"]

IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")
# Ultralytics' default palette, for the mosaic fill and any visualisation
COLOURS = [
    (255, 56, 56),
    (255, 157, 151),
    (255, 112, 31),
    (255, 178, 29),
    (207, 210, 49),
    (72, 249, 10),
    (26, 147, 52),
    (61, 219, 134),
    (26, 178, 255),
    (0, 194, 255),
    (52, 69, 147),
    (100, 115, 255),
    (0, 24, 236),
    (132, 56, 255),
    (82, 0, 133),
    (203, 56, 255),
    (255, 149, 200),
    (255, 55, 199),
]


def letterbox(
    image: np.ndarray,
    target: np.ndarray,
    shape: tuple[int, int],
    scale_fill: bool = False,
    scaleup: bool = True,
    pad: float = 114.0,
):
    """Resize preserving aspect ratio and pad to ``shape`` (h, w).

    Works in place on the target's box columns (``target`` is (N, 5) = [cls, x1, y1, x2, y2]).
    """
    h, w = image.shape[:2]
    new_h, new_w = shape
    r = min(new_h / h, new_w / w)
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (new_w, new_h) if scale_fill else (round(w * r), round(h * r))
    image = np.asarray(Image.fromarray(image).resize(new_unpad, Image.BILINEAR))
    top, left = round((new_h - new_unpad[1]) / 2), round((new_w - new_unpad[0]) / 2)
    out = np.full((new_h, new_w, 3), pad, dtype=np.uint8)
    out[top : top + new_unpad[1], left : left + new_unpad[0]] = image
    target = target.copy()
    if len(target):
        target[:, 1:5] *= r
        target[:, [1, 3]] += left
        target[:, [2, 4]] += top
    return out, target


def augment_hsv(image: np.ndarray, hgain: float = 0.015, sgain: float = 0.7, vgain: float = 0.4) -> np.ndarray:
    """Random HSV jitter, done in YIQ space like Ultralytics (cheap and stable)."""
    if not (hgain or sgain or vgain):
        return image
    hsv = np.asarray(Image.fromarray(image).convert("HSV")).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + np.random.uniform(-hgain, hgain) * 180) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + np.random.uniform(-sgain, sgain)), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1 + np.random.uniform(-vgain, vgain)), 0, 255)
    return np.asarray(Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB"))


def _clip_target(target: np.ndarray, shape: tuple[int, int], min_size: float = 2.0) -> np.ndarray:
    """Clip the box columns of an (N, 5) target to ``shape`` and drop degenerate rows."""
    if not len(target):
        return target
    out = target.copy()
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, shape[1])
    out[:, [2, 4]] = out[:, [2, 4]].clip(0, shape[0])
    wh = out[:, 3:5] - out[:, 1:3]
    return out[(wh[:, 0] > min_size) & (wh[:, 1] > min_size)]


def random_affine(
    image: np.ndarray,
    target: np.ndarray,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    flipud: float = 0.0,
    fliplr: float = 0.5,
):
    """Random affine + flip augmentation.

    The transform is built as output -> input (the convention PIL's ``Image.transform`` needs),
    so boxes are mapped with the inverse.
    """
    h, w = image.shape[:2]
    target = target.copy()
    boxes = target[:, 1:5]
    if random.random() < fliplr:
        image = np.ascontiguousarray(image[:, ::-1])
        if boxes.size:
            x1 = boxes[:, 0].copy()
            boxes[:, 0] = w - boxes[:, 2]
            boxes[:, 2] = w - x1
    if random.random() < flipud:
        image = np.ascontiguousarray(image[::-1])
        if boxes.size:
            y1 = boxes[:, 1].copy()
            boxes[:, 1] = h - boxes[:, 3]
            boxes[:, 3] = h - y1
    centre = np.array([w / 2, h / 2], dtype=np.float64)
    angle = random.uniform(-degrees, degrees)
    shear_x = random.uniform(-shear, shear) * math.pi / 180
    shear_y = random.uniform(-shear, shear) * math.pi / 180
    s = random.uniform(1 - scale, 1 + scale)
    dx = random.uniform(-translate, translate) * w
    dy = random.uniform(-translate, translate) * h
    # forward (input -> output) affine
    a = math.cos(angle) * s
    b = math.sin(angle) * s
    c = -math.sin(angle) * s
    d = math.cos(angle) * s
    fwd = np.array([[a, c, 0.0], [b, d, 0.0], [0, 0, 1.0]])
    shear_m = np.array([[1.0, math.tan(shear_x), 0.0], [math.tan(shear_y), 1.0, 0.0], [0, 0, 1.0]])
    fwd = shear_m @ fwd
    fwd[:2, 2] = centre - fwd[:2, :2] @ centre + (dx, dy)
    inv = np.linalg.inv(fwd)
    coefficients = (inv[0, 0], inv[0, 1], inv[0, 2], inv[1, 0], inv[1, 1], inv[1, 2])
    image = np.asarray(Image.fromarray(image).transform((w, h), Image.AFFINE, coefficients, Image.BILINEAR))
    if boxes.size:
        corners = np.stack([boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]], axis=1)
        flat = corners.reshape(-1, 2)
        x = fwd[0, 0] * flat[:, 0] + fwd[0, 1] * flat[:, 1] + fwd[0, 2]
        y = fwd[1, 0] * flat[:, 0] + fwd[1, 1] * flat[:, 1] + fwd[1, 2]
        mapped = np.stack([x, y], axis=-1).reshape(-1, 4, 2)
        target[:, 1:5] = np.concatenate([mapped.min(axis=1), mapped.max(axis=1)], axis=-1)
        target = _clip_target(target, (h, w))
    return image, target


def mosaic4(
    images: list[np.ndarray],
    targets: list[np.ndarray],
    shape: tuple[int, int],
):
    """Four-image mosaic on a 2s x 2s canvas, where ``s = shape[0] // 2``.

    Each image keeps its bottom-right corner on a random point around the canvas centre, so
    objects survive the crop in every quadrant. Returns the canvas and the shifted boxes.
    """
    s = shape[0] // 2
    xc, yc = random.uniform(0.3 * s, 0.7 * s), random.uniform(0.3 * s, 0.7 * s)
    canvas = np.full((2 * s, 2 * s, 3), 114, dtype=np.uint8)
    out_boxes = []
    for i, (img, target) in enumerate(zip(images, targets, strict=True)):
        h, w = img.shape[:2]
        qx = xc if i % 2 == 0 else xc + s  # quadrant anchor points
        qy = yc if i < 2 else yc + s
        dx, dy = int(round(qx - w)), int(round(qy - h))  # canvas offset of the image
        x0, y0 = max(dx, 0), max(dy, 0)
        x1, y1 = min(dx + w, 2 * s), min(dy + h, 2 * s)
        canvas[y0:y1, x0:x1] = img[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
        if len(target):
            shifted = target.copy()
            shifted[:, 1:5] += (dx, dy, dx, dy)
            shifted = _clip_target(shifted, (2 * s, 2 * s))
            if len(shifted):
                out_boxes.append(shifted)
    merged = np.concatenate(out_boxes) if out_boxes else np.zeros((0, 5), dtype=np.float32)
    return canvas, merged


class DetectionDataset:
    """Images plus YOLO ``.txt`` labels (normalised ``cls cx cy w h`` per line).

    Args:
        root: dataset root; expects ``images/`` (optionally ``train``/``val``) and ``labels/``.
        imgsz: square training size.
        hyp: augmentation gains; anything with matching attribute names is accepted.
        augment: turn augmentation off for validation.
        stride: pad the size to a multiple of the model stride.
    """

    def __init__(
        self,
        root: str | Path,
        imgsz: int = 640,
        hyp=None,
        augment: bool = True,
        stride: int = 32,
        image_dir: str = "images",
        label_dir: str = "labels",
    ) -> None:
        self.root = Path(root)
        self.imgsz = int(round(imgsz / stride) * stride)
        self.augment = augment
        self.hyp = hyp
        self.names: dict[int, str] = {}
        self.samples: list[tuple[Path, Path]] = []
        for sub_dir in self._image_dirs(image_dir):
            for path in sorted(sub_dir.rglob("*")):
                if path.suffix.lower() in IMG_SUFFIXES:
                    label = self.root / label_dir / path.relative_to(sub_dir).with_suffix(".txt")
                    self.samples.append((path, label))
        if not self.samples:
            raise FileNotFoundError(f"no images found under {self.root}")

    def _image_dirs(self, image_dir: str) -> list[Path]:
        base = self.root / image_dir
        if (base / "train").is_dir():
            return [base / "train", base / "val"]
        return [base]

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def nc(self) -> int:
        return (max(self.names) + 1) if self.names else 1

    def load_label(self, path: Path) -> np.ndarray:
        if not path.exists():
            return np.zeros((0, 5), dtype=np.float32)
        rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            return np.zeros((0, 5), dtype=np.float32)
        arr = np.asarray(rows, dtype=np.float32)
        for cls in arr[:, 0]:
            self.names.setdefault(int(cls), str(int(cls)))
        return arr

    def xywh_to_xyxy(self, norm_xywh: np.ndarray, width: int, height: int) -> np.ndarray:
        out = np.empty((len(norm_xywh), 4), dtype=np.float32)
        cx, cy, w, h = (
            norm_xywh[:, 0] * width,
            norm_xywh[:, 1] * height,
            norm_xywh[:, 2] * width,
            norm_xywh[:, 3] * height,
        )
        out[:, 0], out[:, 1] = cx - w / 2, cy - h / 2
        out[:, 2], out[:, 3] = cx + w / 2, cy + h / 2
        return out

    def load_image(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        path, label_path = self.samples[index]
        image = np.asarray(Image.open(path).convert("RGB"))
        label = self.load_label(label_path)
        boxes = (
            self.xywh_to_xyxy(label[:, 1:5], image.shape[1], image.shape[0])
            if len(label)
            else np.zeros((0, 4), np.float32)
        )
        classes = label[:, 0:1] if len(label) else np.zeros((0, 1), np.float32)
        return image, np.concatenate([classes, boxes], axis=1) if len(label) else np.zeros((0, 5), np.float32)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(image uint8 HWC, target (N, 5) [cls, x1, y1, x2, y2])``."""
        image, target = self.load_image(index)
        hyp = self.hyp
        augment = self.augment and hyp is not None
        if augment and getattr(hyp, "mosaic", 0.0) > 0 and random.random() < hyp.mosaic:
            idxs = [index] + [random.randrange(len(self)) for _ in range(3)]
            images, targets = zip(*[self.load_image(i) for i in idxs], strict=True)
            canvas, target = mosaic4(list(images), list(targets), (self.imgsz, self.imgsz))
            image = np.asarray(Image.fromarray(canvas).resize((self.imgsz, self.imgsz), Image.BILINEAR))
            if len(target):
                target[:, 1:5] *= self.imgsz / canvas.shape[0]  # canvas is 2x the mosaic cell
            return image, target

        image, target = letterbox(image, target, (self.imgsz, self.imgsz))
        if augment:
            image, target = random_affine(
                image,
                target,
                degrees=getattr(hyp, "degrees", 0.0),
                translate=getattr(hyp, "translate", 0.1),
                scale=getattr(hyp, "scale", 0.5),
                shear=getattr(hyp, "shear", 0.0),
                flipud=getattr(hyp, "flipud", 0.0),
                fliplr=getattr(hyp, "fliplr", 0.5),
            )
            image = augment_hsv(
                image, getattr(hyp, "hsv_h", 0.015), getattr(hyp, "hsv_s", 0.7), getattr(hyp, "hsv_v", 0.4)
            )
        return image, target

    @staticmethod
    def to_mx(image: np.ndarray) -> mx.array:
        return mx.array(image.astype(np.float32) / 255.0)

    @staticmethod
    def to_batch(targets: list[np.ndarray], imgsz: int) -> dict[str, mx.array]:
        """Collate ``(N, 5)`` targets into normalised batch tensors."""
        rows = []
        for i, target in enumerate(targets):
            if len(target) == 0:
                continue
            norm = np.concatenate(
                [
                    np.full((len(target), 1), i, np.float32),
                    target[:, :1],
                    target[:, 1:5] / np.array([imgsz, imgsz, imgsz, imgsz], np.float32),
                ],
                axis=1,
            )
            rows.append(norm)
        if not rows:
            return {"batch_idx": mx.zeros((0,)), "cls": mx.zeros((0,)), "bboxes": mx.zeros((0, 4))}
        joined = np.concatenate(rows)
        return {
            "batch_idx": mx.array(joined[:, 0]),
            "cls": mx.array(joined[:, 1]),
            "bboxes": mx.array(joined[:, 2:6]),
        }


class YOLOBatchLoader:
    """Batches a :class:`DetectionDataset`; shuffles and drops the last partial batch when training."""

    def __init__(
        self, dataset: DetectionDataset, batch_size: int = 16, shuffle: bool = True, drop_last: bool = True
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def __iter__(self):
        order = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(order)
        for start in range(0, len(order), self.batch_size):
            ids = order[start : start + self.batch_size]
            if self.drop_last and len(ids) < self.batch_size:
                break
            images, targets = zip(*[self.dataset[i] for i in ids], strict=True)
            batch = np.stack([self.dataset.to_mx(im) for im in images])
            yield mx.array(batch), DetectionDataset.to_batch(list(targets), self.dataset.imgsz)
