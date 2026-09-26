"""YOLO26 Detect head: dual one-to-many / one-to-one (NMS-free) branches, DFL-free."""

from __future__ import annotations

import math

import mlx.core as mx

from .conv import Conv, Conv2d, DWConv
from .module import Module, ModuleList, Sequential, _copy_module
from .ops import dist2bbox, make_anchors, topk

__all__ = ["Detect"]


class Detect(Module):
    """Detection head with an optional one-to-one (end-to-end) inference branch.

    With ``reg_max == 1`` (YOLO26) regression is a plain unconstrained ltrb
    prediction: four channels per anchor, no distribution, no DFL loss term.
    """

    def __init__(self, nc: int = 80, reg_max: int = 1, end2end: bool = False, ch: tuple = ()) -> None:
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        self.end2end = end2end
        self.max_det = 300
        self.stride = [8.0, 16.0, 32.0][: self.nl]
        self.shape = None
        self._anchors = None
        self._stride_tensor = None

        c2 = max(16, ch[0] // 4, reg_max * 4)
        c3 = max(ch[0], min(nc, 100))
        self.cv2 = ModuleList(*(Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), Conv2d(c2, 4 * reg_max, 1)) for x in ch))
        self.cv3 = ModuleList(
            *(
                Sequential(
                    Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    Conv2d(c3, nc, 1),
                )
                for x in ch
            )
        )
        if end2end:
            self.one2one_cv2 = self.cv2.deepcopy()
            self.one2one_cv3 = self.cv3.deepcopy()

    # ------------------------------------------------------------------ helpers
    @property
    def one2many(self) -> dict:
        return {"box_head": self.cv2, "cls_head": self.cv3}

    @property
    def one2one(self) -> dict:
        return {"box_head": self.one2one_cv2, "cls_head": self.one2one_cv3}

    def forward_head(self, x, box_head=None, cls_head=None) -> dict:
        """Run both branch towers and flatten to (batch, anchors, channels)."""
        if box_head is None or cls_head is None:
            return {}
        boxes, scores = [], []
        for i in range(self.nl):
            f = x[i]
            boxes.append(box_head[i](f).reshape(f.shape[0], -1, 4 * self.reg_max))
            scores.append(cls_head[i](f).reshape(f.shape[0], -1, self.nc))
        return {"boxes": mx.concatenate(boxes, axis=1), "scores": mx.concatenate(scores, axis=1), "feats": x}

    def forward(self, x):
        preds = {"one2many": self.forward_head(x, **self.one2many)}
        if hasattr(self, "one2one_cv2"):
            # detaching keeps the one-to-one branch out of the backbone graph
            x_det = [mx.stop_gradient(xi) for xi in x] if self.training else x
            preds["one2one"] = self.forward_head(x_det, **self.one2one)
        if self.training:
            return preds
        key = "one2one" if self.end2end and "one2one" in preds else "one2many"
        y = self.inference(preds[key])
        return self.postprocess(y) if key == "one2one" else y

    # ----------------------------------------------------------------- inference
    def build_anchors(self, feats) -> None:
        shape = feats[0].shape
        if self._anchors is None or self.shape != shape:
            self._anchors, self._stride_tensor = make_anchors(feats, self.stride, 0.5)
            self.shape = shape

    def inference(self, x: dict) -> mx.array:
        """Decode to (batch, anchors, 4 + nc): xywh boxes then sigmoid class scores."""
        self.build_anchors(x["feats"])
        dbox = dist2bbox(x["boxes"], self._anchors, xywh=not self.end2end, axis=-1) * self._stride_tensor
        return mx.concatenate([dbox, mx.sigmoid(x["scores"])], axis=-1)

    def postprocess(self, preds: mx.array) -> mx.array:
        """Top-k selection without NMS: (batch, min(max_det, A), 6)."""
        boxes, scores = preds[..., :4], preds[..., 4:]
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        return mx.concatenate([mx.take_along_axis(boxes, idx[..., None], axis=1), scores, conf], axis=-1)

    def get_topk_index(self, scores: mx.array, max_det: int):
        """Two-stage top-k: best class per anchor, then best detection per class."""
        anchors, nc = scores.shape[1], scores.shape[2]
        k = min(max_det, anchors)
        _, ori_index = topk(mx.max(scores, axis=-1), k, axis=-1)
        flat = mx.take_along_axis(scores, ori_index[:, :, None], axis=1).reshape(ori_index.shape[0], -1)
        flat_scores, flat_index = topk(flat, k, axis=-1)
        labels = (flat_index % nc)[:, :, None].astype(mx.float32)
        return flat_scores[:, :, None], labels, mx.take_along_axis(ori_index, flat_index // nc, axis=-1)

    # ---------------------------------------------------------------------- init
    def bias_init(self, stride0: float = 8.0) -> None:
        """Ultralytics Detect bias init: box bias 2.0, class bias from object density."""
        for heads in [self.one2many] + ([self.one2one] if hasattr(self, "one2one_cv2") else []):
            for i, (box, cls) in enumerate(zip(heads["box_head"], heads["cls_head"], strict=True)):
                box[-1].bias[...] = 2.0
                cls[-1].bias[...] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)

    def fuse(self) -> Detect:
        clone = _copy_module(self)
        for name in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
            if hasattr(self, name):
                clone._children[name] = getattr(self, name).fuse()
        return clone
