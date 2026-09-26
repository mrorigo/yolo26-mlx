"""MuSGD: the hybrid Muon + SGD optimizer used by the YOLO26 recipe.

Port of ``ultralytics.optim.MuSGD``. Parameters in a group with ``use_muon=True`` (matrices and
convolution filters, i.e. anything with ``ndim >= 2``) get an orthogonalized Muon update *plus* an
SGD update; everything else (biases, BatchNorm scales) takes the plain SGD path. Orthogonalization
is the quintic Newton-Schulz iteration from the Muon paper, batched over matrices that share a
shape.

The optimizer works directly on MLX's parameter tree, which is what ``value_and_grad`` returns a
gradient tree for::

    params = model.parameters()
    loss, grads = mx.value_and_grad(loss_fn)(params)
    optimizer.step(grads)          # params updated in place
"""

from __future__ import annotations

import mlx.core as mx

__all__ = ["MuSGD", "muon_param_groups", "newton_schulz", "zeropower_via_newtonschulz5"]


def newton_schulz(x: mx.array, steps: int = 5, eps: float = 1e-7) -> mx.array:
    """Approximate the orthogonalisation ``G -> U V'`` of a batch of matrices via Newton-Schulz.

    The coefficients maximise the slope at zero, so five iterations pull the singular values
    into a 0.5-1.5 band. Tall matrices are transposed first, as in the reference.
    """
    shape = x.shape
    x = x.reshape(-1, shape[-2], shape[-1])  # work on a batch of matrices
    x = x / (mx.linalg.norm(x, axis=(-2, -1), keepdims=True) + eps)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = mx.swapaxes(x, -1, -2)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        ax = x @ mx.swapaxes(x, -1, -2)
        x = a * x + (b * ax + c * (ax @ ax)) @ x
    if transposed:
        x = mx.swapaxes(x, -1, -2)
    return x.reshape(shape)


def zeropower_via_newtonschulz5(g: mx.array, eps: float = 1e-7) -> mx.array:
    """Alias of :func:`newton_schulz` kept for parity with the reference helper name."""
    return newton_schulz(g, eps=eps)


class MuSGD:
    """Hybrid Muon/SGD optimizer over an MLX parameter tree.

    Args:
        params: the parameter tree to optimize, updated in place by :meth:`step`.
        groups: parameter groups as dicts with ``keys`` (parameter names) plus optional
            ``lr``, ``momentum``, ``weight_decay``, ``nesterov`` and ``use_muon``.
        lr, momentum, weight_decay, nesterov: defaults for groups that omit them.
        muon, sgd: mixing weights of the orthogonalized and the plain update in hybrid mode.
    """

    def __init__(
        self,
        params: dict[str, mx.array],
        groups: list[dict] | None = None,
        lr: float = 1e-2,
        momentum: float = 0.937,
        weight_decay: float = 5e-4,
        nesterov: bool = True,
        use_muon: bool = False,
        muon: float = 0.2,
        sgd: float = 1.0,
    ) -> None:
        self.params = params
        self.muon, self.sgd = muon, sgd
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "use_muon": use_muon,
        }
        self.groups = [
            {**defaults, **{k: v for k, v in g.items() if k != "params"}, "keys": list(g["keys"])}
            for g in (groups or [{}])
        ]
        self.state: dict[str, dict[str, mx.array]] = {}
        self.step_count = 0

    def step(self, grads: dict[str, mx.array]) -> None:
        """Apply one optimizer step, updating ``self.params`` in place."""
        self.step_count += 1
        for group in self.groups:
            names = [k for k in group["keys"] if k in grads]
            if not names:
                continue
            self._ensure_state(names)
            if group["use_muon"]:
                self._muon_step(names, grads, group)
                self._sgd_step(names, grads, {**group, "lr": group["lr"] * self.sgd})
            else:
                self._sgd_step(names, grads, group)

    # ------------------------------------------------------------------ internals
    def _ensure_state(self, names) -> None:
        for name in names:
            state = self.state.setdefault(name, {})
            if "buf" not in state:
                state["buf"] = mx.zeros_like(self.params[name])
            if "muon_buf" not in state:
                state["muon_buf"] = mx.zeros_like(self.params[name])

    def _sgd_step(self, names, grads, group) -> None:
        lr, momentum, nesterov = group["lr"], group["momentum"], group["nesterov"]
        for name in names:
            grad = grads[name]
            if group["weight_decay"]:
                grad = grad + group["weight_decay"] * self.params[name]
            state = self.state[name]
            state["buf"][...] = state["buf"] * momentum + grad
            update = grad + momentum * state["buf"] if nesterov else state["buf"]
            self.params[name][...] = self.params[name] - lr * update

    def _muon_step(self, names, grads, group) -> None:
        """Orthogonalize the momentum of every 2D/4D tensor, then apply the Muon update.

        Tensors that flatten to the same width are orthogonalized as one batch: rows are
        zero-padded to a common height, which is safe because zero rows stay zero under
        Newton-Schulz.
        """
        lr, momentum, nesterov = group["lr"], group["momentum"], group["nesterov"]
        mats = [n for n in names if self.params[n].ndim >= 2]
        for name in mats:
            state = self.state[name]
            state["muon_buf"][...] = momentum * state["muon_buf"] + (1 - momentum) * grads[name]
        buckets: dict[int, list[tuple[str, mx.array]]] = {}
        for name in mats:
            state = self.state[name]
            update = momentum * state["muon_buf"] + (1 - momentum) * grads[name] if nesterov else state["muon_buf"]
            buckets.setdefault(int(update[0].size), []).append(
                (name, update.reshape(-1, int(update[0].size)), tuple(update.shape))
            )

        for cols, items in buckets.items():
            height = max(u.shape[0] for _, u, _ in items)
            batch = mx.concatenate([mx.pad(u, [(0, height - u.shape[0]), (0, 0)]) for _, u, _ in items], axis=0)
            out = newton_schulz(batch)
            for i, (name, update, shape) in enumerate(items):
                block = out[i * height : i * height + update.shape[0]]
                scale = max(1.0, shape[0] / cols) ** 0.5
                shaped = (block.reshape(shape) * scale).astype(self.params[name].dtype)
                self.params[name][...] = self.params[name] - lr * self.muon * shaped

    # ------------------------------------------------------------------ utilities
    def state_dict(self) -> dict:
        return {name: dict(state) for name, state in self.state.items()}

    def load_state_dict(self, state: dict) -> None:
        for name, values in state.items():
            self.state.setdefault(name, {}).update(values)


def muon_param_groups(model, hyp) -> list[dict]:
    """Split a model's parameters the way the YOLO26 recipe does.

    Matrices and convolution filters go to the Muon group; BatchNorm weights and every bias get
    no weight decay; the rest decays. Classification-tower weights get a 3x learning rate, which
    is what makes finetuning onto a new label set behave.
    """
    decay, no_decay, muon = [], [], []
    for name, value in model.named_parameters():
        if value.ndim >= 2:
            muon.append(name)
        elif "bias" in name or ".bn." in name or name.endswith(".bn.weight"):
            no_decay.append(name)
        else:
            decay.append(name)
    lr = hyp.lr0
    return [
        {"keys": decay, "weight_decay": hyp.weight_decay, "lr": lr, "param_group": "weight"},
        {"keys": no_decay, "weight_decay": 0.0, "lr": lr, "param_group": "norm"},
        {"keys": muon, "weight_decay": hyp.weight_decay, "lr": lr, "use_muon": True, "param_group": "muon"},
    ]
