"""YOLO26 detection criterion: DFL-free CIoU + BCE, dual head with Progressive Loss.

Port of ``ultralytics.utils.loss.v8DetectionLoss`` / ``E2ELoss`` specialised for
``reg_max == 1``: there is no distribution to learn, so the third loss term is a plain L1 on
the image-normalised ltrb distances (``l1_loss``) instead of DFL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

from ..nn.ops import bbox2dist, bbox_iou, dist2bbox, make_anchors, split_sizes
from .tal import TaskAlignedAssigner

__all__ = ["DetectionLoss", "E2EDetectLoss", "LossWeights"]


@dataclass
class LossWeights:
    """Loss gains and schedule knobs (defaults follow the YOLO26 training recipe)."""

    box: float = 7.5
    cls: float = 0.5
    l1: float = 1.5
    dfl: float = 1.5  # alias of `l1` for the DFL-free head
    tal_topk: int = 10
    tal_topk2: int | None = None
    alpha: float = 0.5
    beta: float = 6.0
    epochs: int = 100
    o2m_init: float = 0.8  # Progressive Loss: starting weight of the one-to-many branch
    final_o2m: float = 0.1  # ... and its weight on the last epoch
    extra: dict = field(default_factory=dict)

    @property
    def dist_gain(self) -> float:
        return self.l1 if self.l1 is not None else self.dfl


class DetectionLoss:
    """One branch's loss: CIoU (box) + BCE (class) + DFL or L1, with TAL assignment."""

    def __init__(self, model, hyp: LossWeights | None = None, topk: int | None = None, topk2: int | None = None):
        self.hyp = hyp or LossWeights()
        head = model.head
        self.head, self.nc, self.reg_max = head, head.nc, head.reg_max
        self.stride = list(head.stride)
        self.no = head.nc + head.reg_max * 4
        self.use_dfl = head.reg_max > 1
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss" if self.use_dfl else "l1_loss")
        self.assigner = TaskAlignedAssigner(
            topk=self.hyp.tal_topk if topk is None else topk,
            num_classes=self.nc,
            alpha=self.hyp.alpha,
            beta=self.hyp.beta,
            stride=self.stride,
            topk2=self.hyp.tal_topk2 if topk2 is None else topk2,
        )

    # ------------------------------------------------------------------ utilities
    def preprocess(self, targets: mx.array, batch_size: int, imgsz: mx.array) -> mx.array:
        """``(N, 6)`` [batch, cls, xywh] -> padded ``(b, G, 5)`` [cls, xyxy] in pixels."""
        n, ne = targets.shape
        if n == 0:
            return mx.zeros((batch_size, 0, ne - 1))
        batch_idx = targets[:, 0].astype(mx.int32)
        onehot_idx = mx.zeros((batch_size + 1,), dtype=mx.int32).at[batch_idx + 1].add(mx.ones_like(batch_idx))
        offsets = mx.cumsum(onehot_idx)  # offsets[b] = number of targets before image b
        g = int(onehot_idx[1:].max().item())  # most objects in any image
        within = mx.arange(n) - offsets[batch_idx]
        flat = mx.zeros((batch_size * g, ne - 1))
        flat = flat.at[batch_idx * g + within].add(targets[:, 1:])
        out = flat.reshape(batch_size, g, ne - 1)
        scale = mx.array([imgsz[1], imgsz[0], imgsz[1], imgsz[0]])
        cx, cy, w, h = split_sizes(out[..., 1:5] * scale, 4, axis=-1)
        boxes = mx.concatenate([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)
        return mx.concatenate([out[..., :1], boxes], axis=-1)

    def bbox_decode(self, anchor_points: mx.array, pred_dist: mx.array) -> mx.array:
        """Predicted ltrb -> xyxy in feature-grid units (identity when ``reg_max == 1``)."""
        if self.use_dfl:
            b, a, c = pred_dist.shape
            proj = mx.arange(self.reg_max, dtype=pred_dist.dtype)
            pred_dist = mx.softmax(pred_dist.reshape(b, a, 4, c // 4), axis=-1) @ proj
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    # ---------------------------------------------------------------------- losses
    def __call__(self, preds: dict, batch: dict) -> tuple[mx.array, dict]:
        """Return ``(loss * batch_size, {name: value})`` for one branch."""
        loss, items = self.compute(preds, batch)
        return loss * preds["boxes"].shape[0], items

    def compute(self, preds: dict, batch: dict) -> tuple[mx.array, dict]:
        pred_dist, pred_scores = preds["boxes"], preds["scores"]
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        feats0 = preds["feats"][0]
        imgsz = mx.array([feats0.shape[1] * self.stride[0], feats0.shape[2] * self.stride[0]], dtype)
        batch_size = pred_scores.shape[0]

        targets = mx.concatenate(
            [batch["batch_idx"][:, None].astype(dtype), batch["cls"][:, None], batch["bboxes"]], axis=-1
        )
        targets = self.preprocess(targets, batch_size, imgsz)
        gt_labels, gt_bboxes = targets[..., :1], targets[..., 1:]
        mask_gt = mx.expand_dims(mx.sum(gt_bboxes, axis=-1) > 0, -1)

        pred_bboxes = self.bbox_decode(anchor_points, pred_dist)  # grid units, xyxy
        target_bboxes, target_labels, target_scale, fg_mask, _ = self.assigner(
            mx.stop_gradient(mx.sigmoid(pred_scores)),
            mx.stop_gradient(pred_bboxes * stride_tensor),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        # Per-anchor alignment target: scale[b] on foreground anchors, 0 elsewhere. Summing this is
        # the reference's ``target_scores.sum()``, and it doubles as the box/distribution weight.
        weight = mx.where(fg_mask, target_scale, 0.0)[..., None]  # (b, A, 1)
        target_scores_sum = mx.maximum(mx.sum(weight), 1.0)

        loss_cls = self._cls_loss(pred_scores, target_labels, weight, target_scores_sum)

        idx = _fg_index(fg_mask)
        if idx.size > 0:
            n_anchors = pred_dist.shape[1]
            anchor_ids = idx % n_anchors  # the same anchors are shared by both branch tensors
            rows = lambda t: t.reshape(-1, t.shape[-1])[idx]  # noqa: E731 - gather (b, A, C) rows
            anchors = mx.take(anchor_points, anchor_ids, axis=0)  # (A, 2) -> (M, 2), grid units
            strides = mx.take(stride_tensor, anchor_ids, axis=0)
            target_grid = rows(target_bboxes) / strides  # pixel units -> grid units
            iou = bbox_iou(rows(pred_bboxes), target_grid, xywh=False, CIoU=True)
            loss_box = _aligned_mean(1.0 - iou, rows(weight))
            loss_dist = self._dist_loss(rows(pred_dist), anchors, strides, imgsz, target_grid, rows(weight))
        else:  # no assigned anchors: keep the graph alive with zero-valued terms
            loss_box = loss_dist = pred_dist.sum() * 0.0

        loss = mx.array([loss_box * self.hyp.box, loss_cls * self.hyp.cls, loss_dist * self.hyp.dist_gain])
        return mx.sum(loss), dict(zip(self.loss_names, [float(v) for v in loss], strict=False))

    def _cls_loss(self, pred_scores, target_labels, weight, target_scores_sum):
        """BCE against the sparse alignment targets.

        ``BCE(x, t) = softplus(x) - x * t``, and ``t`` is non-zero only at each foreground anchor's
        own class, so the sum collapses to ``sum(softplus)`` over the whole (b, A, nc) logit tensor
        minus one gathered value per foreground anchor. That avoids building the dense target matrix
        and computing BCE on it element by element.
        """
        total = mx.sum(_softplus(pred_scores))
        if weight.size and bool(mx.any(weight > 0).item()):
            per_anchor = weight[..., 0]  # target value per anchor: scale on foreground anchors
            cols = mx.where(per_anchor > 0, target_labels, 0)[:, :, None]
            gathered = mx.take_along_axis(pred_scores, cols, axis=-1)[..., 0]
            total = total - mx.sum(mx.where(per_anchor > 0, gathered * per_anchor, 0.0))
        return total / target_scores_sum

    def _dist_loss(self, pred_dist, anchor_points, stride_tensor, imgsz, target_bboxes, weight):
        """DFL term (reg_max > 1) or the DFL-free L1 term on image-normalised ltrb.

        All arguments are already restricted to foreground anchors, shaped (M, ...).
        """
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max - 1 if self.use_dfl else None)
        if self.use_dfl:
            bins = pred_dist.shape[-1] // 4
            per_side = pred_dist.reshape(*pred_dist.shape[:-1], 4, bins)
            return _aligned_mean(_dfl(per_side, target_ltrb), weight)
        # normalise ltrb by image size so the L1 term is scale free: (M, 4) for l, t, r, b
        scale = mx.concatenate([stride_tensor / imgsz[1], stride_tensor / imgsz[0]] * 2, axis=-1)
        l1 = mx.mean(mx.abs(pred_dist * scale - target_ltrb * scale), axis=-1, keepdims=True)
        return _aligned_mean(l1, weight)


class E2EDetectLoss:
    """Dual-head loss with Progressive Loss: supervision shifts from one-to-many to one-to-one.

    Early training leans on the one-to-many branch (dense, stable targets); its weight decays
    linearly to ``final_o2m`` by the last epoch, so the inference-time one-to-one branch is
    doing most of the work when the model is deployed NMS-free.
    """

    def __init__(self, model, hyp: LossWeights | None = None):
        self.hyp = hyp or LossWeights()
        self.one2many = DetectionLoss(model, self.hyp, topk=self.hyp.tal_topk, topk2=self.hyp.tal_topk2)
        self.one2one = DetectionLoss(model, self.hyp, topk=7, topk2=1)
        self.total = 1.0
        self.o2m = self.hyp.o2m_init
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m
        self.final_o2m = self.hyp.final_o2m
        self.updates = 0

    def __call__(self, preds: dict, batch: dict) -> tuple[mx.array, dict]:
        loss_o2m, _ = self.one2many(preds["one2many"], batch)
        loss_o2o, items = self.one2one(preds["one2one"], batch)
        return loss_o2m * self.o2m + loss_o2o * self.o2o, items

    def update(self) -> None:
        """Advance the Progressive Loss schedule (call once per epoch)."""
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def decay(self, x: int) -> float:
        span = max(self.hyp.epochs - 1, 1)
        return max(1 - x / span, 0) * (self.o2m_copy - self.final_o2m) + self.final_o2m


# ----------------------------------------------------------------------- helpers
def _fg_index(fg: mx.array) -> mx.array:
    """Flat (batch, anchor) indices of the foreground mask ``(b, A)``, shaped (M,).

    MLX has no ``nonzero``, so order the flattened mask and take the leading ones.
    """
    flat = (fg > 0).astype(mx.int32).reshape(-1)
    return mx.argsort(-flat)[: int(flat.sum().item())]


def _aligned_mean(values: mx.array, weight: mx.array) -> mx.array:
    """Alignment-weighted mean: ``(loss * weight).sum() / target_scores.sum()``."""
    return mx.sum(values * weight) / mx.maximum(mx.sum(weight), 1e-9)


def _bce_with_logits(x: mx.array, target: mx.array) -> mx.array:
    """Numerically stable elementwise BCE with logits."""
    return mx.maximum(x, 0) - x * target + mx.log1p(mx.exp(-mx.abs(x)))


def _softplus(x: mx.array) -> mx.array:
    """``log(1 + exp(x))`` without overflow."""
    return mx.maximum(x, 0) + mx.log1p(mx.exp(-mx.abs(x)))


def _dfl(pred_dist: mx.array, target: mx.array) -> mx.array:
    """Distribution Focal Loss over the reg_max bins, per row."""
    reg_max = pred_dist.shape[-1]
    target = mx.clip(target, 0, reg_max - 1 - 0.01)
    tl = mx.floor(target).astype(mx.int32)
    tr = tl + 1
    wl, wr = tr - target, 1 - (tr - target)
    logp = mx.log_softmax(pred_dist, axis=-1)
    gl = mx.take_along_axis(logp, tl[..., None], axis=-1)[..., 0]
    gr = mx.take_along_axis(logp, tr[..., None], axis=-1)[..., 0]
    return mx.mean(-(gl * wl + gr * wr), axis=-1, keepdims=True)
