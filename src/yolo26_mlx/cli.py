"""Command line interface: ``yolo26-mlx train | val | predict | info``."""

from __future__ import annotations

import argparse
import json

import mlx.core as mx
import numpy as np
from PIL import Image

from .data.dataset import DetectionDataset, letterbox
from .metrics import detections_to_numpy, non_max_suppression
from .tasks import build_model
from .train import TrainConfig, Trainer, validate

__all__ = ["main", "predict"]


def _load_image(path: str, imgsz: int = 640) -> mx.array:
    image = np.asarray(Image.open(path).convert("RGB"))
    target = np.zeros((0, 5), dtype=np.float32)
    image, _ = letterbox(image, target, (imgsz, imgsz))
    return mx.array(image.astype(np.float32) / 255.0)[None]


def predict(
    model,
    image: mx.array,
    conf: float = 0.25,
    iou: float = 0.7,
    nms: bool = True,
    max_det: int = 300,
    compile: bool = True,
) -> list[dict]:
    """Run inference on a batched NHWC image tensor; returns per-image detections.

    Each detection is ``{boxes: (N, 4) xyxy, scores: (N,), classes: (N,)}``. With ``nms=False``
    the one-to-one head is used, which needs no NMS pass at all.

    The head mode is set *before* the graph is compiled, since ``mx.compile`` freezes the Python
    values it reads (including which head the ``end2end`` flag selects).
    """
    model.train(False)
    model.head.end2end = not nms
    out = (model.compiled() if compile else model)(image)
    if nms:
        raw = non_max_suppression(np.array(out), conf_thres=conf, iou_thres=iou, max_det=max_det)
    else:
        raw = detections_to_numpy(out)
    results = []
    for det in raw:
        keep = det[:, 4] >= conf if len(det) else np.zeros(0, dtype=bool)
        det = det[keep][:max_det]
        results.append(
            {
                "boxes": np.round(det[:, :4], 1).tolist(),
                "scores": np.round(det[:, 4], 4).tolist(),
                "classes": det[:, 5].astype(int).tolist(),
            }
        )
    return results


def _cmd_info(args) -> int:
    from .config import load_config, scale_config

    cfg = load_config(args.model, args.scale)
    specs, meta = scale_config({**cfg, "scale": args.scale})
    for spec in specs:
        print(f"{spec.index:>3} {spec.frm!s:>10} {spec.n:>2} {spec.name:<14}{spec.args}")
    print(f"scale={meta['scale']} nc={meta['nc']} reg_max={meta['reg_max']} end2end={meta['end2end']}")
    return 0


def _config_from_args(args) -> TrainConfig:
    """Build a TrainConfig from parsed arguments, coercing values to each field's declared type."""
    values = {}
    for key, value in vars(args).items():
        declared = TrainConfig.__dataclass_fields__.get(key)
        if declared is None or value is None:
            continue
        if value != "" and declared.type in ("int", int):
            value = int(value)
        elif value != "" and declared.type in ("float", float):
            value = float(value)
        values[key] = value
    return TrainConfig(**values)


def _cmd_train(args) -> int:
    cfg = _config_from_args(args)
    trainer = Trainer(cfg)
    history = trainer.train()
    best = max(history, key=lambda r: r["map"])
    print(f"best mAP50-95 {best['map']:.4f} (epoch {best['epoch']}), mAP50 {best['map50']:.4f}")
    (trainer.save_dir / "history.json").write_text(json.dumps(history, indent=2))
    return 0


def _cmd_val(args) -> int:
    model = build_model(args.model, args.scale, nc=args.nc, imgsz=args.imgsz, pretrained=args.weights, verbose=False)
    dataset = DetectionDataset(args.data, imgsz=args.imgsz, augment=False)
    metrics = validate(model, dataset, model.head.nc, conf=args.conf, iou=args.iou, nms=not args.nms_free)
    print(json.dumps(metrics, indent=2))
    return 0


def _cmd_predict(args) -> int:
    model = build_model(args.model, args.scale, nc=args.nc, imgsz=args.imgsz, pretrained=args.weights, verbose=False)
    image = _load_image(args.source, args.imgsz)
    results = predict(model, image, conf=args.conf, iou=args.iou, nms=not args.nms_free)
    print(json.dumps(results[0], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yolo26-mlx", description="YOLO26 for Apple MLX")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default="yolo26")
    common.add_argument("--scale", default="n", choices=list("nsmlx"))
    common.add_argument("--nc", type=int, default=None)
    common.add_argument("--imgsz", type=int, default=640)
    common.add_argument("--weights", default=None, help="safetensors weights from a previous run")

    p_info = sub.add_parser("info", parents=[common], help="print the parsed model graph")
    p_info.set_defaults(func=_cmd_info)

    p_train = sub.add_parser("train", parents=[common], help="train a detector")
    for field, value in TrainConfig().__dataclass_fields__.items():
        if field in ("model", "scale", "imgsz", "nc") or field.startswith("_"):
            continue  # already provided by the shared parser
        if field == "scale_aug":
            p_train.add_argument("--scale-aug", dest="scale_aug", type=float, default=None)
        elif isinstance(value, bool):
            p_train.add_argument(
                f"--{field.replace('_', '-')}", dest=field, type=lambda v: v.lower() == "true", default=None
            )
        elif isinstance(value, int) and not isinstance(value, bool):
            p_train.add_argument(f"--{field.replace('_', '-')}", dest=field, type=int, default=None)
        elif isinstance(value, float):
            p_train.add_argument(f"--{field.replace('_', '-')}", dest=field, type=float, default=None)
        else:
            p_train.add_argument(f"--{field.replace('_', '-')}", dest=field, default=None)
    p_train.set_defaults(func=_cmd_train)

    p_val = sub.add_parser("val", parents=[common], help="validate a detector")
    p_val.add_argument("--data", required=True)
    p_val.add_argument("--conf", type=float, default=0.001)
    p_val.add_argument("--iou", type=float, default=0.7)
    p_val.add_argument("--nms-free", dest="nms_free", action="store_true", help="use the one-to-one head")
    p_val.set_defaults(func=_cmd_val)

    p_pred = sub.add_parser("predict", parents=[common], help="run inference on an image")
    p_pred.add_argument("--source", required=True)
    p_pred.add_argument("--conf", type=float, default=0.25)
    p_pred.add_argument("--iou", type=float, default=0.7)
    p_pred.add_argument("--nms-free", dest="nms_free", action="store_true", help="use the one-to-one head")
    p_pred.set_defaults(func=_cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
