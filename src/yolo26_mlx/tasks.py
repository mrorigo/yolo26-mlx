"""DetectionModel: builds a YOLO26 graph from a model YAML and runs it end to end."""

from __future__ import annotations

import mlx.core as mx

from .config import LayerSpec, load_config, scale_config
from .nn.block import C2PSA, SPPF, C2f, C3k2, Concat, Upsample
from .nn.conv import Conv
from .nn.head import Detect
from .nn.module import Module, Sequential

__all__ = ["CompiledModel", "DetectionModel", "build_model"]


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

    # ----------------------------------------------------------------- execution
    def _resolve(self, spec: LayerSpec, outputs: list):
        """YAML `from` indices are 0-based module indices; -1 means "the previous layer"."""
        frm = spec.frm
        if isinstance(frm, list):
            return [outputs[j + 1] if j >= 0 else outputs[-1] for j in frm]
        return outputs[frm + 1] if frm >= 0 else outputs[-1]

    def forward(self, x):
        outputs: list = [x]
        for i, spec in enumerate(self.specs):
            outputs.append(self.model[i](self._resolve(spec, outputs)))
        return outputs[-1]

    def _forward_to_head(self, x) -> list:
        """Run everything up to the Detect head and return its per-level input features."""
        outputs: list = [x]
        for i, spec in enumerate(self.specs[:-1]):
            outputs.append(self.model[i](self._resolve(spec, outputs)))
        return self._resolve(self.specs[-1], outputs)

    # -------------------------------------------------------------------- strides
    def build_strides(self, imgsz: int = 256) -> None:
        """Probe the graph with a dummy image to record per-level strides."""
        self.train(False)
        head = self.model[-1]
        head.stride = [float(imgsz) / f.shape[1] for f in self._forward_to_head(mx.zeros((1, imgsz, imgsz, 3)))]
        head._anchors, head.shape = None, None
        self.stride = list(head.stride)
        self.train(True)

    @property
    def head(self) -> Detect:
        return self.model[-1]

    def compiled(self) -> CompiledModel:
        """Return a view of this model whose forward pass is a single ``mx.compile``d graph.

        Since every Conv is already a compiled conv+BatchNorm+SiLU block, this whole-graph view no
        longer pays off (640px/batch 1: 8.1 ms eager, 11.7 ms compiled) - prefer plain eager, and
        prefer *unfused* weights too, since folding now removes work the block graph would have
        fused for free (640px/batch 8: 39.9 ms unfolded, 53.0 ms folded).

        Two sharp edges, both measured:

        * Captured arrays are frozen. ``mx.compile`` does not see in-place updates to arrays it
          captured, so this graph keeps serving the weights and BatchNorm statistics it traced with.
          Build it after training has finished, and never reuse it across a training step.
        * Inference only. Compiling a *training* forward either freezes the parameters outright
          (scope capture becomes a constant: the loss stops moving after one step) or, with the
          parameter tree passed in as an argument, re-traces the ~1000-input graph every step
          (25x slower). Training stays eager; that is where the per-block graphs earn their keep.
        """
        if self.training:
            raise RuntimeError(
                "compiled() is an inference view; call model.eval() first. Compiling a training "
                "forward either freezes the parameters or re-traces every step (see the docstring)."
            )
        return CompiledModel(self)


class CompiledModel(Module):
    """An ``mx.compile``d view of a :class:`DetectionModel` for inference.

    Kept for completeness, but note that it is currently *slower* than plain eager execution:
    every Conv already runs as a compiled conv+BatchNorm+SiLU block (:func:`_conv_block`), so the
    model-level graph only adds a fixed cost (640px/batch 1: 8.1 ms eager, 9.1 ms folded+compiled,
    11.7 ms compiled). It also has the sharp edges documented on
    :meth:`DetectionModel.compiled` - captured arrays are frozen for the lifetime of the graph.
    """

    def __init__(self, model: DetectionModel) -> None:
        super().__init__()
        self.inner = model
        self.graph = mx.compile(lambda x: model(x))

    def forward(self, x):
        return self.graph(x)


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
