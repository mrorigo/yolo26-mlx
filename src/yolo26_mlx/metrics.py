"""Detection metrics: greedy NMS, precision/recall and COCO-style mAP."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .nn.ops import nms as nms_op

__all__ = ["ap_per_class", "detections_to_numpy", "mean_average_precision", "non_max_suppression"]


def non_max_suppression(prediction: np.ndarray, conf_thres: float = 0.001, iou_thres: float = 0.7, max_det: int = 300):
    """Greedy NMS for the one-to-many head.

    Args:
        prediction: (B, A, 4 + nc) with xywh boxes followed by class probabilities.
        max_det: cap on detections per image.

    Returns a list of ``(n, 6)`` arrays of ``[x1, y1, x2, y2, conf, cls]``.
    """
    out = []
    for pred in prediction:
        scores = pred[:, 4:]
        keep_all = scores > conf_thres
        boxes_xyxy = _xywh2xyxy(pred[:, :4])
        idx = np.argwhere(keep_all)
        if idx.shape[0] == 0:
            out.append(np.zeros((0, 6), dtype=np.float32))
            continue
        boxes, scores_keep, classes = boxes_xyxy[idx[:, 0]], scores[idx[:, 0], idx[:, 1]], idx[:, 1]
        order = np.argsort(-scores_keep)
        boxes, scores_keep, classes = boxes[order], scores_keep[order], classes[order]
        picked: list[int] = []
        for cls in np.unique(classes):
            sel = np.where(classes == cls)[0]
            keep = nms_op(mx.array(boxes[sel]), mx.array(scores_keep[sel]), iou_thres)
            keep = np.array(keep.tolist() if hasattr(keep, "tolist") else keep, dtype=int)
            picked.extend(sel[keep].tolist())
        picked = np.array(picked, dtype=int)
        if picked.size:
            picked = picked[np.argsort(-scores_keep[picked])][:max_det]
            det = np.concatenate(
                [boxes[picked], scores_keep[picked, None], classes[picked, None].astype(np.float32)], axis=1
            )
        else:
            det = np.zeros((0, 6), dtype=np.float32)
        out.append(det.astype(np.float32))
    return out


def _xywh2xyxy(box: np.ndarray) -> np.ndarray:
    out = box.copy()
    out[:, 0] = box[:, 0] - box[:, 2] / 2
    out[:, 1] = box[:, 1] - box[:, 3] / 2
    out[:, 2] = box[:, 0] + box[:, 2] / 2
    out[:, 3] = box[:, 1] + box[:, 3] / 2
    return out


def detections_to_numpy(detections) -> list[np.ndarray]:
    """Normalise head output (mx array or list) to a list of (N, 6) numpy arrays."""
    if isinstance(detections, mx.array):
        detections = np.array(detections)
    return [np.asarray(d).reshape(-1, d.shape[-1]) for d in detections]


def _iou_matrix(det: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if not len(det) or not len(gt):
        return np.zeros((len(det), len(gt)), dtype=np.float32)
    # float64: an untrained head can emit coordinates large enough to overflow float32 areas
    det, gt = det.astype(np.float64), gt.astype(np.float64)
    lt = np.maximum(det[:, None, :2], gt[None, :, :2])
    rb = np.minimum(det[:, None, 2:4], gt[None, :, 2:4])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_d = np.clip(det[:, 2] - det[:, 0], 0, None) * np.clip(det[:, 3] - det[:, 1], 0, None)
    area_g = np.clip(gt[:, 2] - gt[:, 0], 0, None) * np.clip(gt[:, 3] - gt[:, 1], 0, None)
    return inter / (area_d[:, None] + area_g[None, :] - inter + 1e-9)


def ap_per_class(
    detections: list[np.ndarray],
    labels: list[np.ndarray],
    num_classes: int,
    iou_thresholds: np.ndarray | None = None,
):
    """Match detections to ground truth and compute AP per class and IoU threshold.

    A detection is a true positive at threshold ``t`` when it is the best still-unmatched
    ground truth of its class with IoU >= t; otherwise it is a false positive. Scores are pooled
    across images before the precision/recall curve is built, which is what makes the metric
    independent of dataset ordering.

    Args:
        detections: per-image ``(N, 6)`` [x1, y1, x2, y2, conf, cls].
        labels: per-image ``(M, 5)`` [cls, x1, y1, x2, y2].

    Returns ``(ap, classes)`` with AP (101-point interpolated) in 0..1.
    """
    iou_thresholds = np.linspace(0.5, 0.95, 10) if iou_thresholds is None else np.asarray(iou_thresholds)
    n_iou, n_cls = len(iou_thresholds), num_classes
    n_gt = np.zeros(n_cls, dtype=np.float64)
    records: list[tuple[int, int, float, bool]] = []  # (class, iou index, score, is_tp)

    for img_det, img_gt in zip(detections, labels, strict=True):
        for gt in img_gt:
            if int(gt[0]) < n_cls:
                n_gt[int(gt[0])] += 1
        if not len(img_det):
            continue
        for row in img_det[np.argsort(-img_det[:, 4])]:
            cls = int(row[5])
            if cls >= n_cls:
                continue
            same = np.where(img_gt[:, 0] == cls)[0] if len(img_gt) else np.zeros(0, dtype=int)
            if not len(same):
                records.extend((cls, ti, float(row[4]), False) for ti in range(n_iou))
                continue
            ious = _iou_matrix(row[None, :4], img_gt[same][:, 1:5])[0]
            gi = int(np.argmax(ious))
            # each IoU threshold keeps its own match bookkeeping (a strict pass can match a
            # ground truth that a looser pass already consumed)
            already = {ti: False for ti in range(n_iou)}
            for ti, thr in enumerate(iou_thresholds):
                if ious[gi] < thr or already[ti]:
                    records.append((cls, ti, float(row[4]), False))
                else:
                    already[ti] = True
                    records.append((cls, ti, float(row[4]), True))

    ap = np.zeros((n_cls, n_iou), dtype=np.float64)
    recalls = np.linspace(0, 1, 101)
    for c in np.where(n_gt > 0)[0]:
        for ti in range(n_iou):
            rows = [(score, tp) for cls, t, score, tp in records if cls == c and t == ti]
            if not rows:
                continue
            rows.sort(key=lambda r: -r[0])
            tp_c = np.cumsum([r[1] for r in rows], dtype=np.float64)
            fp_c = np.cumsum([not r[1] for r in rows], dtype=np.float64)
            precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
            recall = tp_c / n_gt[c]
            ap[c, ti] = _interpolate_ap_curve(precision, recall, recalls)
    return ap, n_gt > 0


def _interpolate_ap_curve(precision: np.ndarray, recall: np.ndarray, points: np.ndarray) -> float:
    """101-point interpolated area under one precision/recall curve."""
    order = np.argsort(recall)
    p = np.maximum.accumulate(precision[order][::-1])[::-1]  # monotone envelope
    return float(np.interp(points, recall[order], p, left=p[0] if p.size else 0.0, right=0.0).mean())


def mean_average_precision(ap: np.ndarray, present: np.ndarray) -> tuple[float, float]:
    """Return ``(mAP@0.5:0.95, mAP@0.5)`` over the classes that appear in the labels."""
    if not present.any():
        return 0.0, 0.0
    return float(ap[present].mean()), float(ap[present][:, 0].mean())
