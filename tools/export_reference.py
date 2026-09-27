"""Export a deterministically-initialised PyTorch YOLO26 as MLX-parity reference data.

    python tools/export_reference.py --scale n --imgsz 256

Writes, into --out (default: a temp dir):
  ref_<scale>_<imgsz>.safetensors  weights in MLX layout (NHWC conv kernels)
  ref_x_*.npy, ref_x2_*.npy        single- and two-image inputs (NCHW)
  ref_{o2m,o2o}_{boxes,scores}_*.npy   raw branch outputs (NCHW, channels first)
  ref_o2m_*.npy, ref_e2e_*.npy      decoded one-to-many and NMS-free one-to-one outputs
  ref_act_<i>_*.npy                per-layer activations in train mode
  ref_{align,overlaps,maskin,masktopk,maskpos,maskpos2,fg,tscores}_*.npy
  ref_loss_{o2m,e2e}_*.npy         criterion outputs (total, box, cls, l1)

This is a development aid only: it is the single place where PyTorch appears, and it is
not imported by the package.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.numpy import save_file
from ultralytics.nn.tasks import DetectionModel

TARGETS = [[0.0, 0.0, 0.4, 0.4, 0.2, 0.3], [0.0, 3.0, 0.7, 0.6, 0.05, 0.05], [1.0, 7.0, 0.5, 0.5, 0.4, 0.4]]


def build(scale: str, imgsz: int, cfg_path: str) -> DetectionModel:
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["scale"] = scale
    torch.manual_seed(0)
    model = DetectionModel(cfg, verbose=False)
    model.eval()
    # break the symmetry of the zero-initialised classification head: with a constant score
    # every top-k tie-break is arbitrary and the reference stops being well defined
    g = torch.Generator().manual_seed(7)
    detect = model.model[-1]
    for cls_head in (detect.cv3, detect.one2one_cv3):
        for tower in cls_head:
            last = tower[-1]
            last.weight.data += 0.05 * torch.randn(last.weight.shape, generator=g)
            last.bias.data += 0.05 * torch.randn(last.bias.shape, generator=g)
    return model


def save_weights(model, out: Path, scale: str, imgsz: int) -> None:
    sd = {k: v.detach().numpy() for k, v in model.state_dict().items() if not k.endswith("num_batches_tracked")}
    # torch conv kernels are (O, I, kh, kw); MLX wants (O, kh, kw, I)
    for k, v in sd.items():
        if v.ndim == 4:
            sd[k] = np.ascontiguousarray(v.transpose(0, 2, 3, 1))
    save_file(sd, str(out / f"ref_{scale}_{imgsz}.safetensors"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="n")
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--cfg", default=str(Path(__file__).resolve().parents[1] / "configs/yolo26.yaml"))
    ap.add_argument("--out", default="/Users/origo/.cache/yolo26-mlx/ref")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.scale}_{args.imgsz}"
    with open(args.cfg) as fh:
        nc = yaml.safe_load(fh)["nc"]

    model = build(args.scale, args.imgsz, args.cfg)
    save_weights(model, out, args.scale, args.imgsz)

    torch.manual_seed(1)
    x = torch.randn(1, 3, args.imgsz, args.imgsz)
    np.save(out / f"ref_x_{tag}.npy", x.numpy())
    with torch.no_grad():
        model.end2end = False
        decoded = model(x)
        np.save(out / f"ref_o2m_{tag}.npy", decoded[0].numpy())
        model.end2end = True
        decoded = model(x)
        np.save(out / f"ref_e2e_{tag}.npy", decoded[0].numpy())
        preds = decoded[1]
        for stem, branch in (("o2m", "one2many"), ("o2o", "one2one")):
            np.save(out / f"ref_{stem}_boxes_{tag}.npy", preds[branch]["boxes"].numpy())
            np.save(out / f"ref_{stem}_scores_{tag}.npy", preds[branch]["scores"].numpy())

    # ---- train mode: activations, criterion and assigner internals
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
    from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors

    model.args = get_cfg(DEFAULT_CFG)  # the criterion reads its gains from model.args
    model.nc = nc
    x2 = torch.cat([x, 0.5 * x], dim=0)  # two images, so both batch rows carry targets
    np.save(out / f"ref_x2_{tag}.npy", x2.numpy())
    targets = torch.tensor(TARGETS)
    np.save(out / f"ref_targets_{tag}.npy", targets.numpy())

    acts: dict[str, np.ndarray] = {}
    hooks = [
        layer.register_forward_hook(
            lambda mod, i, o, name=str(ix): (
                acts.__setitem__(name, o.detach().numpy()) if isinstance(o, torch.Tensor) else None
            )
        )
        for ix, layer in enumerate(model.model)
    ]
    model.train()
    preds = model(x2)
    for h in hooks:
        h.remove()
    for key, value in acts.items():
        np.save(out / f"ref_act_{key}_{tag}.npy", value)

    crit = v8DetectionLoss(model, tal_topk=10)
    batch = {"batch_idx": targets[:, 0].long(), "cls": targets[:, 1].long(), "bboxes": targets[:, 2:]}
    loss_o2m, items_o2m = crit(preds["one2many"], batch)
    e2e = E2ELoss(model, v8DetectionLoss)  # Progressive Loss: one-to-many weight 0.8 at step 0
    loss_e2e, items_e2e = e2e(preds, batch)
    print("reference e2e gains: o2m", e2e.o2m, "o2o", e2e.total - e2e.o2m)
    print("torch loss o2m", loss_o2m.sum().item(), items_o2m)
    print("torch loss e2e", loss_e2e.sum().item(), items_e2e)
    np.save(out / f"ref_loss_o2m_{tag}.npy", np.array([loss_o2m.sum().item(), *items_o2m.values()]))
    np.save(out / f"ref_loss_e2e_{tag}.npy", np.array([loss_e2e.sum().item(), *items_e2e.values()]))

    with torch.no_grad():
        strides = model.model[-1].stride
        anchors, stride_t = make_anchors(preds["one2many"]["feats"], strides, 0.5)
        padded = crit.preprocess(targets, 2, scale_tensor=torch.tensor([args.imgsz] * 4, dtype=torch.float32))
        gt_bboxes, mask_gt = padded[..., 1:], (padded[..., 1:].sum(-1, keepdim=True) > 0).float()
        np.save(out / f"ref_padded_{tag}.npy", padded.numpy())
        np.save(out / f"ref_anchors_{tag}.npy", (anchors * stride_t).numpy())

        def assign(branch, topk, topk2, stem):
            tal = TaskAlignedAssigner(
                topk=topk, num_classes=nc, alpha=0.5, beta=6.0, stride=strides.tolist(), topk2=topk2
            )
            tal.bs, tal.n_max_boxes = 2, padded.shape[1]
            scores = preds[branch]["scores"].detach().sigmoid().permute(0, 2, 1)
            boxes = crit.bbox_decode(anchors, preds[branch]["boxes"].detach().permute(0, 2, 1)) * stride_t
            labels, boxes_t, scores_t, fg, idx = tal(
                scores, boxes, anchors * stride_t, padded[..., :1], gt_bboxes, mask_gt
            )
            np.save(out / f"ref_fg_{stem}_{tag}.npy", fg.numpy())
            np.save(out / f"ref_tscores_{stem}_{tag}.npy", scores_t.numpy())
            np.save(out / f"ref_tidx_{stem}_{tag}.npy", idx.numpy())
            np.save(out / f"ref_tbboxes_{stem}_{tag}.npy", boxes_t.numpy())
            np.save(out / f"ref_pboxes_{stem}_{tag}.npy", boxes.numpy())
            print(f"reference assigner {stem}: fg={fg.sum().item()} scores={scores_t.sum().item():.4f}")
            return tal, boxes, scores

        tal, boxes, scores = assign("one2many", 10, 10, "o2m")
        assign("one2one", 7, 1, "o2o")

        # per-anchor ltrb tensors behind the DFL-free L1 term
        from ultralytics.utils.tal import bbox2dist

        # the criterion derives the image size from the P3 feature map, not the input tensor
        imgsz_t = torch.tensor(preds["one2many"]["feats"][0].shape[2:], dtype=torch.float32) * strides[0]
        l1_scale = torch.cat([stride_t / imgsz_t[1], stride_t / imgsz_t[0]], -1).repeat(1, 2)
        fg = torch.from_numpy(np.load(out / f"ref_fg_o2m_{tag}.npy"))
        pred_dist = preds["one2many"]["boxes"].detach().permute(0, 2, 1)
        tgt_ltrb = bbox2dist(
            anchors, torch.from_numpy(np.load(out / f"ref_tbboxes_o2m_{tag}.npy")).to(anchors.dtype) / stride_t
        )
        rows = fg[..., None].expand(-1, -1, 4).reshape(-1, 4)
        np.save(out / f"ref_l1_pred_{tag}.npy", (pred_dist * l1_scale).reshape(-1, 4)[rows].numpy())
        np.save(out / f"ref_l1_tgt_{tag}.npy", (tgt_ltrb * l1_scale).reshape(-1, 4)[rows].numpy())
        mask_in = tal.select_candidates_in_gts(anchors * stride_t, gt_bboxes, mask_gt)
        align, overlaps = tal.get_box_metrics(scores, boxes, padded[..., :1], gt_bboxes, mask_in * mask_gt)
        mask_topk = tal.select_topk_candidates(align, topk_mask=mask_gt.expand(-1, -1, tal.topk).bool())
        mask_pos = mask_topk * mask_in * mask_gt.bool()
        _, fg2, mask_pos2 = tal.select_highest_overlaps(mask_pos, overlaps, padded.shape[1], align)
        for name, tensor in (
            ("maskin", mask_in),
            ("align", align),
            ("overlaps", overlaps),
            ("masktopk", mask_topk),
            ("maskpos", mask_pos),
            ("maskpos2", mask_pos2),
            ("fg2", fg2),
        ):
            np.save(out / f"ref_{name}_{tag}.npy", tensor.numpy().astype(np.float32))
    print("exported", tag, "->", out)


if __name__ == "__main__":
    main()
