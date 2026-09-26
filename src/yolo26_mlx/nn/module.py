"""Minimal MLP-style module system for YOLO26-MLX.

Parameters live in per-module dicts so the full model flattens into a single
``dict[str, mx.array]`` tree. This is what MLX's ``value_and_grad`` consumes and
what ``mx.save_safetensors`` writes, so there is exactly one representation of a
model's weights end to end.
"""

from __future__ import annotations

import mlx.core as mx

__all__ = ["Module", "ModuleList", "Sequential", "tree_flatten_params", "tree_unflatten_params"]


class Module:
    """Base class: named array parameters, named child modules, `forward` dispatch."""

    def __init__(self) -> None:
        object.__setattr__(self, "_params", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "_children", {})
        object.__setattr__(self, "_cache", {})
        object.__setattr__(self, "training", True)

    # ---------------------------------------------------------------- attributes
    def __setattr__(self, key: str, value) -> None:
        if key in ("_params", "_buffers", "_children", "_cache", "training"):
            object.__setattr__(self, key, value)
        elif key.startswith("_") and not key.startswith("__"):
            # underscore-prefixed state is derived (e.g. cached anchors): never saved, never optimized
            self._cache[key] = value
            self.__dict__.pop(key, None)
        elif isinstance(value, mx.array):
            self._params[key] = value
            self.__dict__.pop(key, None)  # a stale plain attribute would shadow _params
        elif isinstance(value, Module):
            self._children[key] = value
        elif isinstance(value, (list, tuple)) and len(value) and all(isinstance(v, Module) for v in value):
            self._children[key] = ModuleList(*value) if not isinstance(value, ModuleList) else value
        else:
            object.__setattr__(self, key, value)

    def __getattr__(self, key: str):
        if key in ("_params", "_buffers", "_children", "_cache"):
            raise AttributeError(key)  # these are set in __init__; guard against copy/pickle paths
        cache = object.__getattribute__(self, "_cache")
        if key in cache:
            return cache[key]
        params = object.__getattribute__(self, "_params")
        if key in params:
            return params[key]
        children = object.__getattribute__(self, "_children")
        if key in children:
            return children[key]
        buffers = object.__getattribute__(self, "_buffers")
        if key in buffers:
            return buffers[key]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {key!r}")

    def register_buffer(self, key: str, value: mx.array) -> None:
        """State that is saved and restored but never optimized (e.g. BN running stats)."""
        self._buffers[key] = value
        self.__dict__.pop(key, None)

    def __dir__(self):
        return list(super().__dir__()) + list(self._children) + list(self._params)

    # ------------------------------------------------------------------ traversal
    def named_modules(self, prefix: str = "") -> list[tuple[str, Module]]:
        out = [(prefix, self)]
        for name, child in self._children.items():
            out.extend(child.named_modules(f"{prefix}.{name}" if prefix else name))
        return out

    def named_parameters(self, prefix: str = "") -> list[tuple[str, mx.array]]:
        out = [(f"{prefix}.{k}" if prefix else k, v) for k, v in self._params.items()]
        for name, child in self._children.items():
            out.extend(child.named_parameters(f"{prefix}.{name}" if prefix else name))
        return out

    def parameters(self) -> dict[str, mx.array]:
        return dict(self.named_parameters())

    def buffers(self) -> dict[str, mx.array]:
        out = {f"{p}.{k}" if p else k: v for p, m in self.named_modules() for k, v in m._buffers.items()}
        return out

    def state(self) -> dict[str, mx.array]:
        """Everything needed to resume: trainable parameters plus persistent buffers."""
        return {**self.parameters(), **self.buffers()}

    def num_parameters(self) -> int:
        return sum(v.size for v in self.parameters().values())

    # ------------------------------------------------------------------ lifecycle
    def train(self, mode: bool = True) -> Module:
        object.__setattr__(self, "training", mode)
        for child in self._children.values():
            child.train(mode)
        return self

    def eval(self) -> Module:
        return self.train(False)

    def update(self, params: dict[str, mx.array]) -> Module:
        """Overwrite parameters in place, keyed by dotted parameter name."""
        own = self.state()
        for key, value in params.items():
            if key not in own:
                raise KeyError(f"unknown parameter {key!r}")
            own[key][...] = value
        return self

    def load_weights(self, path) -> None:
        """Load weights from a safetensors file (or an in-memory dict)."""
        if isinstance(path, (str, bytes)) or hasattr(path, "__fspath__"):
            path = str(path)
            weights = mx.load(str(path)) if path.endswith(".safetensors") else _load_npz(path)
        else:
            weights = path
        weights = {
            (k if k.startswith("model.") else f"model.{k}"): v  # reference checkpoints nest under `model.`
            for k, v in weights.items()
            if k not in ("anchors", "strides")
        }
        self.update(weights)

    def save_weights(self, path: str, metadata: dict[str, str] | None = None) -> None:
        mx.save_safetensors(str(path), self.state(), metadata or {})

    # -------------------------------------------------------------------- compute
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def fuse(self) -> Module:
        """Return a copy with BatchNorm folded into the preceding convolution."""
        return fuse_module(self)

    def deepcopy(self) -> Module:
        """Return an independent copy (new parameter arrays) of this subtree."""
        clone = _copy_module(self)
        for key, child in self._children.items():
            clone._children[key] = child.deepcopy()
        return clone

    def __repr__(self) -> str:
        params = self.num_parameters()
        return f"{type(self).__name__}({params} params)" if params else f"{type(self).__name__}()"


def _load_npz(path: str) -> dict[str, mx.array]:
    import numpy as np

    with np.load(path) as data:
        return {k: mx.array(data[k]) for k in data.files}


def _copy_module(module: Module) -> Module:
    """Shallow copy: same child modules, copied parameter dict."""
    clone = object.__new__(type(module))
    Module.__init__(clone)
    object.__setattr__(clone, "_children", dict(module._children))
    for key, value in vars(module).items():
        if key not in ("_params", "_buffers", "_children", "_cache"):
            object.__setattr__(clone, key, value)
    for key, value in module._cache.items():
        clone._cache[key] = value
    for key, value in module._params.items():
        clone._params[key] = mx.array(value)
    for key, value in module._buffers.items():
        clone._buffers[key] = mx.array(value)
    return clone


def fuse_module(module: Module) -> Module:
    """Recursively fuse Conv+BatchNorm pairs; non-fusable modules are deep-copied."""
    from .conv import Conv

    clone = _copy_module(module)
    for name, child in list(module._children.items()):
        clone._children[name] = child.fuse() if isinstance(child, Module) else child
    if isinstance(module, Conv):
        return module.fuse_into()
    return clone


class ModuleList(Module):
    """Ordered list of modules, addressable by integer index."""

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        for i, module in enumerate(modules):
            self._children[str(i)] = module

    def __len__(self) -> int:
        return len(self._children)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._children[str(i)] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        return self._children[str(index)]

    def __iter__(self):
        return (self._children[str(i)] for i in range(len(self)))

    def append(self, module: Module) -> None:
        self._children[str(len(self))] = module

    def extend(self, modules) -> None:
        for module in modules:
            self.append(module)

    def fuse(self) -> ModuleList:
        clone = _copy_module(self)
        for key, child in self._children.items():
            clone._children[key] = child.fuse()
        return clone


class Sequential(Module):
    """Chain of modules (or plain callables) applied in order."""

    def __init__(self, *modules) -> None:
        super().__init__()
        for i, module in enumerate(modules):
            if isinstance(module, Module):
                self._children[str(i)] = module
            else:
                object.__setattr__(self, f"_op{i}", module)

    def __len__(self) -> int:
        return len(self._children)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._children[str(i)] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        return self._children[str(index)]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def forward(self, x):
        for i in range(len(self)):
            x = self._children[str(i)](x)
        return x

    def fuse(self) -> Sequential:
        clone = _copy_module(self)
        for key, child in self._children.items():
            clone._children[key] = child.fuse()
        return clone


def tree_flatten_params(module: Module) -> dict[str, mx.array]:
    return module.parameters()


def tree_unflatten_params(module: Module, flat: dict[str, mx.array]) -> Module:
    return module.update(flat)
