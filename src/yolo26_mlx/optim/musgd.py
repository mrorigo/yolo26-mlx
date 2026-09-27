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


# ------------------------------------------------------------------ compiled passes
# The per-tensor arithmetic below is launch-bound, not arithmetic-bound: a step touches
# ~370 tensors (the count is the same for every scale - only the widths differ), and eager
# execution spends ~40us per tensor on kernel launches. Compiling the whole pass collapses
# the SGD half of a step from 14.5 ms to 3.2 ms.
#
# The learning rate, momentum and weight decay are passed as 0-d *arrays*, never as Python
# floats: `mx.compile` specialises on scalar argument *values*, so a float lr that changes
# every step under a schedule re-traces the graph (~230 ms a step). A same-shape array is a
# graph input, so the trace is reused and the new value is honoured.
def _sgd_update_nesterov(params, grads, bufs, lr, momentum, weight_decay):
    """``buf = momentum*buf + g'; u = g' + momentum*buf; p -= lr*u`` for a whole tree.

    Note the asymmetry with :func:`_muon_momentum`: this buffer accumulates ``g'`` with no
    ``(1 - momentum)`` factor, which is what the reference does.
    """
    new_params, new_bufs = {}, {}
    for name, p in params.items():
        g = grads[name] + weight_decay * p
        buf = momentum * bufs[name] + g
        new_params[name] = p - lr * (g + momentum * buf)
        new_bufs[name] = buf
    return new_params, new_bufs


def _sgd_update(params, grads, bufs, lr, momentum, weight_decay):
    """Plain (non-Nesterov) SGD pass."""
    new_params, new_bufs = {}, {}
    for name, p in params.items():
        g = grads[name] + weight_decay * p
        buf = momentum * bufs[name] + g
        new_params[name] = p - lr * buf
        new_bufs[name] = buf
    return new_params, new_bufs


def _muon_momentum_nesterov(params, grads, muon_bufs, momentum):
    """Muon EMA (which *does* have the ``(1 - momentum)`` factor) and the Nesterov input."""
    new_bufs, updates = {}, {}
    for name in params:
        g = grads[name]
        buf = momentum * muon_bufs[name] + (1 - momentum) * g
        new_bufs[name] = buf
        updates[name] = momentum * buf + (1 - momentum) * g
    return new_bufs, updates


def _muon_momentum(params, grads, muon_bufs, momentum):
    """Muon EMA without Nesterov: the update is the buffer itself."""
    new_bufs, updates = {}, {}
    for name in params:
        buf = momentum * muon_bufs[name] + (1 - momentum) * grads[name]
        new_bufs[name] = updates[name] = buf
    return new_bufs, updates


def _apply_muon(params, updates, scales, lr, muon_weight):
    """``p -= lr * muon_weight * scale * update`` for a whole tree."""
    return {name: params[name] - lr * muon_weight * scales[name] * updates[name] for name in params}


_SGD_NESTEROV = mx.compile(_sgd_update_nesterov)
_SGD_PLAIN = mx.compile(_sgd_update)
_MUON_NESTEROV = mx.compile(_muon_momentum_nesterov)
_MUON_PLAIN = mx.compile(_muon_momentum)
_APPLY_MUON = mx.compile(_apply_muon)


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
            # The SGD component of a Muon group runs on the *post-Muon* parameters, so the Muon
            # update lands first: the SGD component sees the post-Muon parameter, which also
            # makes its weight-decay term depend on the Muon step.
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
        """SGD component for a whole group in one compiled pass, then one write-back per tensor."""
        params = {n: self.params[n] for n in names}
        bufs = {n: self.state[n]["buf"] for n in names}
        weight_decay = mx.array(group["weight_decay"], dtype=mx.float32)
        new_params, new_bufs = (
            _SGD_NESTEROV if group["nesterov"] else _SGD_PLAIN
        )(
            params,
            {n: grads[n] for n in names},
            bufs,
            mx.array(group["lr"], dtype=mx.float32),
            mx.array(group["momentum"], dtype=mx.float32),
            weight_decay,
        )
        for name, value in new_params.items():
            self.params[name][...] = value
        for name, value in new_bufs.items():
            self.state[name]["buf"][...] = value

    def _muon_step(self, names, grads, group) -> None:
        """Orthogonalize the momentum of every 2D/4D tensor, then apply the Muon update.

        The momentum recurrence and the final parameter update are each one compiled pass; the
        Newton-Schulz iterations in between stay eager because they run on a handful of
        differently-shaped batches, and tracing one graph per shape would cost more than it saves.
        Tensors that flatten to the same width are orthogonalized as one batch: rows are
        zero-padded to a common height, which is safe because zero rows stay zero under
        Newton-Schulz.
        """
        momentum = mx.array(group["momentum"], dtype=mx.float32)
        mats = [n for n in names if self.params[n].ndim >= 2]
        if not mats:
            return
        params = {n: self.params[n] for n in mats}
        new_bufs, updates = (
            _MUON_NESTEROV if group["nesterov"] else _MUON_PLAIN
        )(params, {n: grads[n] for n in mats}, {n: self.state[n]["muon_buf"] for n in mats}, momentum)

        buckets: dict[int, list[tuple[str, mx.array, tuple[int, ...]]]] = {}
        for name in mats:
            update = updates[name]
            cols = int(update[0].size)
            buckets.setdefault(cols, []).append((name, update.reshape(-1, cols), tuple(update.shape)))

        shaped: dict[str, mx.array] = {}
        for cols, items in buckets.items():
            height = max(u.shape[0] for _, u, _ in items)
            batch = mx.concatenate([mx.pad(u, [(0, height - u.shape[0]), (0, 0)]) for _, u, _ in items], axis=0)
            out = newton_schulz(batch)
            for i, (name, update, shape) in enumerate(items):
                block = out[i * height : i * height + update.shape[0]]
                # scale from the flattened matrix, as in the reference (rows / cols)
                scale = max(1.0, shape[0] / cols) ** 0.5
                shaped[name] = (block.reshape(shape) * mx.array(scale, mx.float32)).astype(self.params[name].dtype)

        new_params = _APPLY_MUON(
            params, shaped, {n: mx.array(1.0) for n in shaped}, mx.array(group["lr"], mx.float32),
            mx.array(self.muon, mx.float32),
        )
        for name, value in new_params.items():
            self.params[name][...] = value
        for name, value in new_bufs.items():
            self.state[name]["muon_buf"][...] = value

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
