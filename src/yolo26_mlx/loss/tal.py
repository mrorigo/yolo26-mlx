"""Task-aligned label assignment with YOLO26's small-target-aware (STAL) behaviour.

Port of ``ultralytics.utils.tal.TaskAlignedAssigner``. Two YOLO26-specific behaviours are
preserved:

* ``topk2`` -- a second, tighter top-k filter applied after multi-GT overlap resolution.
* ``stride_val`` -- the size floor applied to tiny ground-truth boxes (the stride of the
  second detection level), so small objects never lose all of their candidate anchors.
"""

from __future__ import annotations

import mlx.core as mx

from ..nn.ops import bbox_iou, one_hot, split_sizes, xywh2xyxy, xyxy2xywh

__all__ = ["TaskAlignedAssigner"]

EPS = 1e-9


def _align_and_overlaps(bbox_scores, gt_bboxes, pd_bboxes, mask, alpha, beta, eps):
    """Alignment metric and IoU for every (ground truth, anchor) pair, in one graph.

    CIoU over the broadcast (b, G, A, 4) geometry is ~20 elementwise passes; eager MLX runs each
    one separately, which made this the criterion's hot spot (18.7 ms at 640px/batch 8 with 100
    objects per image). Compiled it is 2.9 ms for the same result.
    """
    iou = mx.clip(
        bbox_iou(gt_bboxes[:, :, None, :], pd_bboxes[:, None, :, :], xywh=False, CIoU=True), 0, None
    )[..., 0]
    return (bbox_scores**alpha * iou**beta) * mask, iou * mask


def _count_topk(topk_idxs, n_anchors):
    """How many times each anchor appears in a ground truth's top-k list, as one reduction."""
    return (topk_idxs[..., None] == mx.arange(n_anchors)).astype(mx.float32).sum(axis=-2)


def _candidates_in_gts(xy_centers, gt_bboxes, mask_gt, stride_val, eps):
    """Anchor centres inside each (small-target-floored) ground-truth box."""
    cxcywh = xyxy2xywh(gt_bboxes)
    wh = cxcywh[..., 2:]
    small = (wh < stride_val) * mask_gt
    wh = mx.where(small > 0, mx.full(wh.shape, float(stride_val)), wh)
    boxes = xywh2xyxy(mx.concatenate([cxcywh[..., :2], wh], axis=-1))
    lt, rb = split_sizes(boxes, 2, axis=-1)
    xc, yc = xy_centers[None, None, :, 0], xy_centers[None, None, :, 1]
    mask = (xc[..., None] - lt[..., 0][:, :, None, None]) > eps
    mask = mask & ((yc[..., None] - lt[..., 1][:, :, None, None]) > eps)
    mask = mask & ((rb[..., 0][:, :, None, None] - xc[..., None]) > eps)
    return (mask & ((rb[..., 1][:, :, None, None] - yc[..., None]) > eps))[..., 0]


def _normalise_align(align_metric, overlaps, mask_pos, eps):
    """Normalisation factor for the alignment targets, shaped (b, A) - one value per anchor.

    The reference multiplies its dense one-hot target matrix by this factor; keeping it per anchor
    lets the criterion work with (labels, scale) and never materialise that matrix.
    """
    pos_align = align_metric * mask_pos
    pos_align_max = mx.max(pos_align, axis=-1, keepdims=True)  # (b, G, 1)
    pos_overlaps = mx.max(overlaps * mask_pos, axis=-1, keepdims=True)  # (b, G, 1)
    scaled = pos_align * pos_overlaps / (pos_align_max + eps)
    return mx.max(scaled, axis=-2)  # max over ground truths -> (b, A)


_ALIGN_AND_OVERLAPS = mx.compile(_align_and_overlaps)
_NORMALISE_ALIGN = mx.compile(_normalise_align)
_COUNT_TOPK = mx.compile(_count_topk)
_CANDIDATES_IN_GTS = mx.compile(_candidates_in_gts)


class TaskAlignedAssigner:
    """Assign ground-truth boxes to anchors by task-aligned score.

    Args:
        topk: candidates considered per ground-truth box.
        topk2: secondary, tighter candidate filter (``None`` or equal to ``topk`` disables it).
        num_classes: class count.
        alpha: exponent on the classification score; beta: exponent on the IoU.
        stride: per-level strides; the second level's stride is the small-target size floor.
    """

    def __init__(
        self,
        topk: int = 10,
        num_classes: int = 80,
        alpha: float = 0.5,
        beta: float = 6.0,
        stride: list[float] | None = None,
        eps: float = 1e-9,
        topk2: int | None = None,
    ) -> None:
        self.topk = topk
        self.topk2 = topk2 or topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.stride = list(stride or [8.0, 16.0, 32.0])
        self.stride_val = self.stride[1] if len(self.stride) > 1 else self.stride[0]
        self.eps = eps

    # ------------------------------------------------------------------ entry point
    def __call__(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """Assign boxes to anchors.

        Args:
            pd_scores: (b, A, nc) predicted class probabilities.
            pd_bboxes: (b, A, 4) predicted boxes in pixel units, xyxy.
            anc_points: (A, 2) anchor centres in pixel units.
            gt_labels: (b, G, 1) class ids; gt_bboxes: (b, G, 4) xyxy; mask_gt: (b, G, 1) validity.

        Returns ``(target_bboxes, target_labels, target_scale, fg_mask, target_gt_idx)``: the
        assigned box, the class each anchor was given, and a per-image scale. Together they are
        exactly the reference's dense ``(b, A, nc)`` target matrix - ``target_scores[b, a, label] =
        scale[b]`` on foreground anchors, zero elsewhere - without ever materialising it.
        """
        bs, n_anchors = pd_bboxes.shape[0], pd_bboxes.shape[1]
        n_max_boxes = gt_bboxes.shape[1]
        empty = (
            mx.zeros_like(pd_bboxes),
            mx.zeros((bs, n_anchors), dtype=mx.int32),
            mx.zeros((bs, n_anchors), dtype=pd_scores.dtype),  # per-anchor scale
            mx.zeros((bs, n_anchors), dtype=mx.bool_),
            mx.zeros((bs, n_anchors), dtype=mx.int32),
        )
        if n_max_boxes == 0:
            return empty

        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt > 0)
        mask_pos = mask_topk * mask_in_gts * (mask_gt > 0)
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, align_metric)

        idx = target_gt_idx.astype(mx.int32)[..., None]
        target_bboxes = mx.take_along_axis(gt_bboxes, idx, axis=1)
        target_labels = mx.take_along_axis(gt_labels, idx, axis=1)[..., 0].astype(mx.int32)
        # Scale each ground truth's positives by its normalised alignment metric, so a hard
        # example (low score, low IoU) is weighted below an easy one instead of counting fully.
        target_scale = _NORMALISE_ALIGN(align_metric, overlaps, mask_pos, self.eps)
        return target_bboxes, target_labels, target_scale, fg_mask > 0, target_gt_idx.astype(mx.int32)

    # ---------------------------------------------------------------------- pieces
    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Alignment metric ``score**alpha * iou**beta``, shaped (b, G, A)."""
        # class score of each anchor at each ground truth's class: (b, A, G) -> (b, G, A)
        onehot = one_hot(mx.clip(gt_labels, 0, None)[..., 0], self.num_classes, pd_scores.dtype)
        bbox_scores = mx.swapaxes(mx.matmul(pd_scores, mx.swapaxes(onehot, -1, -2)), -1, -2)
        mask = (mask_gt > 0).astype(pd_bboxes.dtype)  # already (b, G, A) after the candidate mask
        align_metric, overlaps = _ALIGN_AND_OVERLAPS(
            bbox_scores, gt_bboxes, pd_bboxes, mask, self.alpha, self.beta, self.eps
        )
        return align_metric.astype(pd_scores.dtype), overlaps.astype(pd_bboxes.dtype)

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        return mx.clip(bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True), 0, None)

    def select_topk_candidates(self, metrics, topk_mask=None):
        """1.0 where an anchor is in exactly one ground truth's top-k list, else 0.

        Mirrors the reference: when the caller supplies ``topk_mask`` (the criterion masks out
        padded ground-truth rows) it is authoritative, so zero-metric anchors inside a valid
        box are kept; without it, rows whose best metric is ~0 are dropped.
        """
        n_anchors = metrics.shape[-1]
        k = min(self.topk, n_anchors)
        topk_idxs = mx.argpartition(-metrics, kth=k - 1, axis=-1)[..., :k]
        if topk_mask is None:
            topk_values = mx.take_along_axis(metrics, topk_idxs, axis=-1)
            topk_mask = mx.max(topk_values, axis=-1, keepdims=True) > self.eps
        topk_mask = mx.broadcast_to(topk_mask, topk_idxs.shape)
        topk_idxs = mx.where(topk_mask, topk_idxs, 0)
        # count[b, g, a] = how many times anchor a appears in ground truth g's top-k list; an
        # anchor claimed by two ground truths is dropped rather than arbitrarily assigned. One
        # reduction over the whole (b, G, k, A) comparison, so the cost no longer grows with a
        # Python loop over ground truths.
        count = _COUNT_TOPK(topk_idxs, n_anchors)
        return mx.where(count > 1, mx.zeros_like(count), count)

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps: float = EPS):
        """Anchor centres strictly inside each ground-truth box, after the small-target floor."""
        return _CANDIDATES_IN_GTS(xy_centers, gt_bboxes, mask_gt, self.stride_val, eps)

    def select_highest_overlaps(self, mask_pos, overlaps, align_metric):
        """Give shared anchors to their best-overlap ground truth, then apply topk2."""
        n_max, n_anchors = mask_pos.shape[1], mask_pos.shape[-1]
        fg_mask = mx.max(mask_pos, axis=1)  # (b, A)
        multi = mx.expand_dims(mx.sum(mask_pos, axis=1) > 1, 1)  # (b, 1, A)
        best = mx.argmax(overlaps, axis=1)  # (b, A)
        is_best = (mx.arange(n_max)[None, :, None] == best[:, None, :]).astype(mask_pos.dtype)
        mask_pos = mx.where(multi > 0, is_best, mask_pos)

        if self.topk2 != self.topk:
            k = min(self.topk2, n_anchors)
            idx = mx.argpartition(-(align_metric * mask_pos), kth=k - 1, axis=-1)[..., :k]
            second = (
                (idx[..., None, :] == mx.arange(n_anchors)[None, None, None, :]).any(axis=-2).astype(mask_pos.dtype)
            )
            mask_pos = mask_pos * second

        target_gt_idx = mx.argmax(mask_pos, axis=1)
        return target_gt_idx.astype(mx.float32), fg_mask, mask_pos
