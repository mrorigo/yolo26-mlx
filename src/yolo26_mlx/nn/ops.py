"""Tensor ops shared by heads, losses and post-processing."""

from __future__ import annotations

import math

import mlx.core as mx

__all__ = [
    "bbox2dist",
    "bbox_iou",
    "dist2bbox",
    "imgsz_from_feats",
    "make_anchors",
    "nms",
    "one_hot",
    "split_sizes",
    "topk",
    "xywh2xyxy",
    "xyxy2xywh",
]


def one_hot(idx: mx.array, n: int, dtype=mx.float32) -> mx.array:
    """One-hot encode integer indices along a new trailing axis (MLX has no one_hot)."""
    return (mx.arange(n)[None, :] == idx[..., None].astype(mx.int32)).astype(dtype)


def topk(x: mx.array, k: int, axis: int = -1) -> tuple[mx.array, mx.array]:
    """Top-k values and their indices, sorted descending.

    ``mx.topk`` returns values only, and the assigner and the NMS-free head both
    need the gather indices, so sort once and slice.
    """
    idx = mx.argsort(-x, axis=axis)[..., :k]
    return mx.take_along_axis(x, idx, axis=axis), idx


def split_sizes(x: mx.array, sizes, axis: int = -1) -> list[mx.array]:
    """Split by chunk sizes, torch's ``split(x, sizes)`` semantics.

    Every call site in this codebase wants sizes, while ``mx.split`` takes split points (and
    silently misbehaves on empty ones), so the slicing is done explicitly.
    """
    if isinstance(sizes, int):
        sizes = [x.shape[axis] // sizes] * sizes
    out, start = [], 0
    for size in sizes:
        window = [slice(None)] * x.ndim
        window[axis] = slice(start, start + size)
        out.append(x[tuple(window)])
        start += size
    if start < x.shape[axis]:  # keep any remainder as a trailing chunk
        window = [slice(None)] * x.ndim
        window[axis] = slice(start, None)
        out.append(x[tuple(window)])
    return out


def make_anchors(feats, strides, grid_cell_offset: float = 0.5):
    """Anchor centres for a list of feature maps, plus their per-anchor stride.

    Centres are in *grid* units (x + 0.5, y + 0.5); the stride is returned separately and
    applied to the decoded box, so regression targets stay in stride units until decode time.

    Returns ``(anchors, stride_tensor)`` with shapes ``(A, 2)`` and ``(A, 1)`` where
    ``A = sum(H_i * W_i)`` and the anchor order matches the head's concat order.
    """
    anchor_points, stride_tensor = [], []
    dtype = feats[0].dtype
    for i, feat in enumerate(feats):
        _, h, w, _ = feat.shape
        s = float(strides[i])
        ys = mx.arange(h, dtype=dtype) + grid_cell_offset
        xs = mx.arange(w, dtype=dtype) + grid_cell_offset
        gy, gx = mx.meshgrid(ys, xs, indexing="ij")
        anchor_points.append(mx.stack([gx, gy], axis=-1).reshape(-1, 2))
        stride_tensor.append(mx.full((h * w, 1), s, dtype=dtype))
    return mx.concatenate(anchor_points), mx.concatenate(stride_tensor)


def imgsz_from_feats(feats, stride0: float = 8.0):
    """Input image size (h, w) implied by the P3 feature map."""
    _, h, w, _ = feats[0].shape
    return h * stride0, w * stride0


def dist2bbox(distance: mx.array, anchor_points: mx.array, xywh: bool = False, axis: int = -1) -> mx.array:
    """Turn ltrb distances relative to anchors into xyxy (or xywh) boxes."""
    lt, rb = split_sizes(distance, 2, axis=axis)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        return mx.concatenate([(x1y1 + x2y2) / 2, x2y2 - x1y1], axis=axis)
    return mx.concatenate((x1y1, x2y2), axis=axis)


def bbox2dist(anchor_points: mx.array, bbox: mx.array, reg_max: float | None = None) -> mx.array:
    """Turn xyxy boxes into ltrb distances relative to anchors."""
    x1y1, x2y2 = split_sizes(bbox, 2, axis=-1)
    dist = mx.concatenate((anchor_points - x1y1, x2y2 - anchor_points), axis=-1)
    return mx.clip(dist, 0, reg_max - 0.01) if reg_max is not None else dist


def xywh2xyxy(x: mx.array) -> mx.array:
    cx, cy, w, h = split_sizes(x, 4, axis=-1)
    return mx.concatenate((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), axis=-1)


def xyxy2xywh(x: mx.array) -> mx.array:
    x1, y1, x2, y2 = split_sizes(x, 4, axis=-1)
    return mx.concatenate(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), axis=-1)


def bbox_iou(
    box1: mx.array,
    box2: mx.array,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> mx.array:
    """IoU family, faithful to ``ultralytics.utils.metrics.bbox_iou``.

    ``box1`` broadcasts against ``box2``; both are (..., 4). Note the details that matter
    numerically: heights carry an ``eps`` (they are not clipped), the centre distance is the
    *sum-of-edges* form divided by four, and CIoU scales the aspect penalty by an adaptive
    ``alpha = v / (1 - iou + v)``.
    """
    if xywh:
        x1, y1, w1, h1 = split_sizes(box1, 4, axis=-1)
        x2, y2, w2, h2 = split_sizes(box2, 4, axis=-1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = split_sizes(box1, 4, axis=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = split_sizes(box2, 4, axis=-1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = mx.clip(mx.minimum(b1_x2, b2_x2) - mx.maximum(b1_x1, b2_x1), 0, None) * mx.clip(
        mx.minimum(b1_y2, b2_y2) - mx.maximum(b1_y1, b2_y1), 0, None
    )
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if not (CIoU or DIoU or GIoU):
        return iou
    cw = mx.maximum(b1_x2, b2_x2) - mx.minimum(b1_x1, b2_x1)  # smallest enclosing box
    ch = mx.maximum(b1_y2, b2_y2) - mx.minimum(b1_y1, b2_y1)
    if CIoU or DIoU:
        c2 = cw**2 + ch**2 + eps
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        if CIoU:
            v = (4 / (math.pi**2)) * (mx.arctan(w2 / h2) - mx.arctan(w1 / h1)) ** 2
            alpha = mx.stop_gradient(v / (1 - iou + v + eps))
            return iou - (rho2 / c2 + v * alpha)
        return iou - rho2 / c2
    c_area = cw * ch + eps
    return iou - (c_area - union) / c_area


def nms(boxes: mx.array, scores: mx.array, iou_thresh: float = 0.7) -> mx.array:
    """Greedy NMS over (N, 4) xyxy boxes with (N,) scores; returns kept indices."""
    order = mx.argsort(-scores)
    n = boxes.shape[0]
    keep: list[int] = []
    suppressed = [False] * n
    order_list = order.tolist()
    box_list = boxes.tolist()
    for idx in order_list:
        if suppressed[idx]:
            continue
        keep.append(idx)
        bx = box_list[idx]
        for j in order_list:
            if suppressed[j] or j == idx:
                continue
            bj = box_list[j]
            iw = min(bx[2], bj[2]) - max(bx[0], bj[0])
            ih = min(bx[3], bj[3]) - max(bx[1], bj[1])
            inter = max(iw, 0.0) * max(ih, 0.0)
            a1 = max(bx[2] - bx[0], 0.0) * max(bx[3] - bx[1], 0.0)
            a2 = max(bj[2] - bj[0], 0.0) * max(bj[3] - bj[1], 0.0)
            if inter / (a1 + a2 - inter + 1e-9) > iou_thresh:
                suppressed[j] = True
    return mx.array(keep, dtype=mx.int32) if keep else mx.zeros((0,), dtype=mx.int32)
