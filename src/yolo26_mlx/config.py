"""Model-config parsing: YAML -> scaled layer specs, mirroring Ultralytics parse_model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

__all__ = ["LayerSpec", "load_config", "make_divisible", "scale_config"]

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


class LayerSpec:
    """One graph node: where it reads from, how many repeats, and its constructor."""

    __slots__ = ("args", "c2", "frm", "index", "n", "name")

    def __init__(self, index: int, frm, n: int, name: str, args: list, c2: int) -> None:
        self.index, self.frm, self.n, self.name, self.args, self.c2 = index, frm, n, name, args, c2

    def __repr__(self) -> str:
        return f"{self.index:>3} from={self.frm!s:>10} n={self.n} {self.name}({self.args}) -> {self.c2}"


def make_divisible(value: float, divisor: int = 8) -> int:
    """Round up to the nearest multiple of ``divisor`` (Ultralytics' channel rule)."""
    return math.ceil(value / divisor) * divisor


def load_config(path_or_name: str, scale: str | None = None) -> dict[str, Any]:
    """Read a model YAML by name (`yolo26`) or path and select a compound scale."""
    path = Path(path_or_name)
    if not path.exists() and not path.suffix:
        for candidate in (f"{path_or_name}.yaml", f"{path_or_name}-n.yaml"):
            if (CONFIG_DIR / candidate).exists():
                path = CONFIG_DIR / candidate
                break
        else:
            path = CONFIG_DIR / f"{path_or_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"model config not found: {path_or_name}")
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if scale:
        if scale not in cfg["scales"]:
            raise KeyError(f"scale {scale!r} not in {sorted(cfg['scales'])}")
        cfg["scale"] = scale
    return cfg


def scale_config(cfg: dict[str, Any]) -> tuple[list[LayerSpec], dict[str, Any]]:
    """Apply depth/width/max-channel scaling and resolve repeated channels."""
    scale = cfg.get("scale") or next(iter(cfg["scales"]))
    depth, width, max_channels = cfg["scales"][scale]
    rows = list(cfg["backbone"]) + list(cfg["head"])

    # single pass: resolve each node's output channels, in scaled units
    c2s: list[int] = []
    layers: list[LayerSpec] = []
    for i, (frm, n, name, args) in enumerate(rows):
        args = list(args)
        n = max(round(n * depth), 1) if n > 1 else n
        idxs = frm if isinstance(frm, list) else [frm]
        if name in ("nn.Upsample", "nn.MaxPool", "nn.AvgPool"):
            c1 = c2s[idxs[0] if idxs[0] >= 0 else len(c2s) - 1]
            c2 = c1
            args = [c2, *args]
        elif name == "Concat":
            c2 = sum(c2s[j if j >= 0 else len(c2s) - 1] for j in idxs)
            args = [c2, *args]
        elif name in ("Detect", "Segment", "Pose", "OBB", "Classify"):
            c2 = 0
            if name == "Detect":
                nc = cfg.get("nc", 80)
                args = [nc if a == "nc" else a for a in args[:1]] + [
                    cfg.get("reg_max", 1),
                    bool(cfg.get("end2end", True)),
                    [c2s[j if j >= 0 else len(c2s) - 1] for j in idxs],
                ]
        else:
            c1 = 3 if i == 0 else c2s[frm if frm >= 0 else i - 1]
            c2 = make_divisible(min(int(args[0]), max_channels) * width, 8)
            rest = args[1:]
            if name in ("C3k2", "C2f", "C2fPSA", "C2PSA", "C3", "C2"):
                args, n = [c1, c2, n, *rest], 1
            else:
                args = [c1, c2, *rest]
            if name == "C3k2" and scale in ("m", "l", "x"):
                args[3:4] = [True]  # C3k2 uses C3k blocks for m/l/x
        c2s.append(c2)
        layers.append(LayerSpec(i, frm, n, name, args, c2))
    return layers, {
        "scale": scale,
        "nc": cfg.get("nc", 80),
        "reg_max": cfg.get("reg_max", 1),
        "end2end": cfg.get("end2end", True),
    }
