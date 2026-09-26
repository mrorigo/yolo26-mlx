"""DetectionModel: builds a YOLO26 graph from a model YAML and runs it end to end."""

from __future__ import annotations

import mlx.core as mx

from .config import LayerSpec, load_config, scale_config
from .nn.block import C2PSA, SPPF, C2f, C3k2, Concat, Upsample
from .nn.conv import Conv
from .nn.head import Detect
from .nn.module import Module, Sequential

__all__ = ["DetectionModel", "build_model"]

REPEATABLE = (C3k2, C2f, C2PSA, SPPF)


class DetectionModel(Module):
    """Layer list + channel bookkeeping, executed sequentially like Ultralytics' nn.Sequential."""

    def __init__(
        self, cfg: str = "yolo26", scale: str | None = "n", nc: int | None = None, verbose: bool = True
    ) -> None:
        super().__init__()
        config = load_config(cfg, scale)
        if nc is not None:
            config["nc"] = nc
        specs, meta = scale_config(config)
        self.cfg_path, self.meta, self.specs = cfg, meta, specs
        self.model = Sequential(*(self._make(s) for s in specs))
        self.stride = [8.0] * len(specs)
        self.names = None
        self._built = False
        if verbose:
            self._print()

    # ------------------------------------------------------------------ building
    def _make(self, spec: LayerSpec):
        name, args = spec.name, list(spec.args)
        if name == "Conv":
            return Conv(*args)
        if name == "C3k2":
            return C3k2(*args)
        if name == "C2f":
            return C2f(*args)
        if name == "C2PSA":
            return C2PSA(*args)
        if name == "SPPF":
            return SPPF(*args)
        if name == "Concat":
            return Concat(args[1] if len(args) > 1 else 1)
        if name == "nn.Upsample":
            return Upsample(scale_factor=args[2] if len(args) > 2 else 2, mode=args[3] if len(args) > 3 else "nearest")
        if name == "Detect":
            return Detect(
                nc=args[0], reg_max=self.meta["reg_max"], end2end=bool(self.meta["end2end"]), ch=tuple(args[3])
            )
        raise KeyError(f"unsupported module {name!r}")

    def _print(self) -> None:
        print(f"{'':>3}{'from':>8}{'n':>3}{'params':>10}  {'module':<12}{'arguments'}")
        total = 0
        for spec in self.specs:
            layer = self.model[spec.index]
            n_params = layer.num_parameters() if isinstance(layer, Module) else 0
            total += n_params
            print(f"{spec.index:>3}{spec.frm!s:>8}{spec.n:>3}{n_params:>10,}  {spec.name:<12}{spec.args}")
        print(f"\nscale={self.meta['scale']} nc={self.meta['nc']} total params: {total:,}")
        self._built = True

    # ------------------------------------------------------------------ execution
    def _resolve(self, spec: LayerSpec, outputs: list):
        """YAML `from` indices are 0-based module indices; -1 means "the previous layer"."""
        frm = spec.frm
        if isinstance(frm, list):
            return [outputs[j + 1] if j >= 0 else outputs[-1] for j in frm]
        return outputs[frm + 1] if frm >= 0 else outputs[-1]

    def forward(self, x):
        outputs: list = [x]
        for i, spec in enumerate(self.specs):
            y = self.model[i](self._resolve(spec, outputs))
            outputs.append(y)
        return outputs[-1]

    # -------------------------------------------------------------------- strides
    def build_strides(self, imgsz: int = 256) -> None:
        """Probe the graph with a dummy image to record per-level strides."""
        self.train(False)
        head = self.model[-1]
        feats = self._forward_to_head(mx.zeros((1, imgsz, imgsz, 3)))
        head.stride = [float(imgsz) / f.shape[1] for f in feats]
        head._anchors, head.shape = None, None
        self.stride = list(head.stride)
        self.train(True)

    def _forward_to_head(self, x) -> list:
        """Run everything up to the Detect head and return its per-level input features."""
        outputs: list = [x]
        for i, spec in enumerate(self.specs[:-1]):
            outputs.append(self.model[i](self._resolve(spec, outputs)))
        return self._resolve(self.specs[-1], outputs)

    @property
    def head(self) -> Detect:
        return self.model[-1]


def build_model(
    cfg: str = "yolo26",
    scale: str = "n",
    nc: int | None = None,
    imgsz: int = 256,
    pretrained: str | None = None,
    verbose: bool = True,
) -> DetectionModel:
    """Instantiate a YOLO26 model, compute its strides and optionally load weights."""
    model = DetectionModel(cfg, scale, nc, verbose)
    model.build_strides(imgsz)
    model.head.bias_init(model.stride[0])
    model.train(False)
    if pretrained:
        model.load_weights(pretrained)
    return model
