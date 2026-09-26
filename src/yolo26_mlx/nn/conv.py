"""Convolution primitives: BatchNorm, Conv (conv+BN+act), DWConv, Conv2d."""

from __future__ import annotations

import math

import mlx.core as mx

from .module import Module, _copy_module

__all__ = ["BatchNorm", "Conv", "Conv2d", "DWConv", "autopad"]


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def autopad(k, p=None, d: int = 1):
    """'same'-style padding: output size = ceil(input / stride)."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class BatchNorm(Module):
    """BatchNorm over the last axis, with training-time batch statistics.

    Defaults follow Ultralytics' ``initialize_weights`` (eps 1e-3, momentum 0.03) rather than
    torch's, so statistics match the reference checkpoints. Running variance tracks the
    unbiased estimate, as in torch.
    """

    def __init__(self, num_features: int, eps: float = 1e-3, momentum: float = 0.03) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = mx.ones((num_features,))
        self.bias = mx.zeros((num_features,))
        self.register_buffer("running_mean", mx.zeros((num_features,)))
        self.register_buffer("running_var", mx.ones((num_features,)))

    def forward(self, x: mx.array) -> mx.array:
        axes = (0, 1, 2)
        if self.training:
            n = x.shape[0] * x.shape[1] * x.shape[2]
            mean = mx.mean(x, axis=axes)
            var = mx.var(x, axis=axes)  # one fused reduction; (x - mean)**2 costs two extra passes
            unbiased = var * n / max(n - 1, 1)  # running_var tracks the unbiased estimate
            self.running_mean[...] = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var[...] = (1 - self.momentum) * self.running_var + self.momentum * unbiased
        else:
            mean, var = self.running_mean, self.running_var
        # fold the normalisation into an affine map: one elementwise pass instead of three
        scale = self.weight * mx.rsqrt(var + self.eps)
        return x * scale + (self.bias - mean * scale)


class Conv2d(Module):
    """Bare convolution with bias (also the final 1x1 head output)."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p=None,
        g: int = 1,
        d: int = 1,
        bias: bool = True,
        act: bool = False,
    ) -> None:
        super().__init__()
        kh, kw = (k, k) if isinstance(k, int) else (k[0], k[1])
        self.act = act
        self.c1, self.c2, self.k, self.s, self.g, self.d = c1, c2, (kh, kw), s, g, d
        self.p = autopad((kh, kw), p, d)
        scale = math.sqrt(2.0 / (kh * kw * (c1 // g)))
        self.weight = mx.random.normal((c2, kh, kw, c1 // g)) * scale
        self.bias = mx.zeros((c2,)) if bias else None

    def forward(self, x: mx.array) -> mx.array:
        x = mx.conv2d(x, self.weight, stride=self.s, padding=self.p, dilation=self.d, groups=self.g)
        if self.bias is not None:
            x = x + self.bias
        return _silu(x) if self.act else x


class Conv(Module):
    """Standard Conv block: Conv2d (no bias) -> BatchNorm -> optional SiLU."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p=None,
        g: int = 1,
        d: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = Conv2d(c1, c2, k, s, p, g, d, bias=False)
        self.bn = BatchNorm(c2)
        self.act = act

    def forward(self, x: mx.array) -> mx.array:
        return _silu(self.bn(self.conv(x))) if self.act else self.bn(self.conv(x))

    def fuse_into(self) -> Conv2d:
        """Fold BatchNorm into the convolution weights (inference/export only).

        The activation travels with the layer, so the folded block is a drop-in replacement.
        """
        bn, conv = self.bn, self.conv
        inv_std = mx.rsqrt(bn.running_var + bn.eps)
        scale = bn.weight * inv_std
        fused = _copy_module(conv)
        fused.weight = conv.weight * scale.reshape(-1, 1, 1, 1)
        fused.bias = bn.bias - bn.running_mean * scale
        fused.act = self.act
        return fused


class DWConv(Conv):
    """Depthwise Conv block: a Conv whose group count equals the channel count."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, d: int = 1, act: bool = True) -> None:
        super().__init__(c1, c2, k, s, g=_gcd(c1, c2), d=d, act=act)
