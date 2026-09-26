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

        Returns ``(target_bboxes, target_scores, fg_mask, target_gt_idx)``.
        """
        bs, n_anchors = pd_bboxes.shape[0], pd_bboxes.shape[1]
        n_max_boxes = gt_bboxes.shape[1]
        empty = (
            mx.zeros_like(pd_bboxes),
            mx.zeros_like(pd_scores),
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

        target_bboxes = mx.take_along_axis(gt_bboxes, target_gt_idx.astype(mx.int32)[..., None], axis=1)
        target_labels = mx.take_along_axis(gt_labels, target_gt_idx.astype(mx.int32)[..., None], axis=1)
        target_scores = self.to_scores(target_labels, fg_mask)

        # Scale each ground truth's positives by its normalised alignment metric, so a hard
        # example (low score, low IoU) is weighted below an easy one instead of counting fully.
        pos_align = align_metric * mask_pos
        pos_align_max = mx.max(pos_align, axis=-1, keepdims=True)
        pos_overlaps = mx.max(overlaps * mask_pos, axis=-1, keepdims=True)
        scaled = pos_align * pos_overlaps / (pos_align_max + self.eps)
        target_scores = target_scores * mx.max(scaled, axis=-2)[..., None]
        return target_bboxes, target_scores, fg_mask > 0, target_gt_idx.astype(mx.int32)

    # ---------------------------------------------------------------------- pieces
    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Alignment metric ``score**alpha * iou**beta``, shaped (b, G, A)."""
        # class score of each anchor at each ground truth's class: (b, A, G)
        onehot = one_hot(mx.clip(gt_labels, 0, None)[..., 0], self.num_classes, pd_scores.dtype)
        bbox_scores = mx.swapaxes(mx.matmul(pd_scores, mx.swapaxes(onehot, -1, -2)), -1, -2)  # (b, A, G) -> (b, G, A)
        # CIoU of every ground truth against every anchor: (b, G, A)
        iou = mx.clip(bbox_iou(gt_bboxes[:, :, None, :], pd_bboxes[:, None, :, :], xywh=False, CIoU=True), 0, None)[
            ..., 0
        ]
        mask = (mask_gt > 0).astype(pd_bboxes.dtype)  # already (b, G, A) after the candidate mask
        align_metric = (bbox_scores**self.alpha * iou**self.beta) * mask
        return align_metric.astype(pd_scores.dtype), (iou * mask).astype(pd_bboxes.dtype)

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
        arange = mx.arange(n_anchors)
        # count[b, g, a] = how many times anchor a appears in ground truth g's top-k list;
        # an anchor claimed by two ground truths is dropped rather than arbitrarily assigned
        counts = [
            (idx[..., None] == arange).astype(mx.float32).sum(axis=-2)
            for idx in topk_idxs.transpose(1, 0, 2)  # one (b, k) index list per ground truth
        ]
        count = mx.stack(counts, axis=1)
        return mx.where(count > 1, mx.zeros_like(count), count)

    def to_scores(self, target_labels, fg_mask):
        """One-hot alignment targets, zero except on foreground anchors."""
        onehot = one_hot(target_labels[..., 0], self.num_classes)
        return onehot * mx.expand_dims(fg_mask > 0, -1)

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps: float = 1e-9):
        """Anchor centres strictly inside each ground-truth box, after the small-target floor."""
        cxcywh = xyxy2xywh(gt_bboxes)
        wh = cxcywh[..., 2:]
        small = (wh < self.stride_val) * mask_gt
        wh = mx.where(small > 0, mx.full(wh.shape, float(self.stride_val)), wh)
        boxes = xywh2xyxy(mx.concatenate([cxcywh[..., :2], wh], axis=-1))
        lt, rb = split_sizes(boxes, 2, axis=-1)  # each (b, G, 2): left-top and right-bottom
        xc, yc = xy_centers[None, None, :, 0], xy_centers[None, None, :, 1]  # (1, 1, A)
        mask = (xc[..., None] - lt[..., 0][:, :, None, None]) > eps
        mask = mask & ((yc[..., None] - lt[..., 1][:, :, None, None]) > eps)
        mask = mask & ((rb[..., 0][:, :, None, None] - xc[..., None]) > eps)
        return (mask & ((rb[..., 1][:, :, None, None] - yc[..., None]) > eps))[..., 0]

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
