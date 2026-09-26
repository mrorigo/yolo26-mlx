"""YOLO26 building blocks (MLX): Bottleneck, C3k, C2f/C3k2, SPPF, Attention, C2PSA."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .conv import Conv
from .module import Module, ModuleList, Sequential
from .ops import split_sizes

__all__ = [
    "C2PSA",
    "C3",
    "SPPF",
    "Attention",
    "Bottleneck",
    "C2f",
    "C3k",
    "C3k2",
    "Concat",
    "MaxPool",
    "PSABlock",
    "Upsample",
]


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class Bottleneck(Module):
    """Two 3x3 convs with an optional identity shortcut."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k=(3, 3),
        e: float = 0.5,
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: mx.array) -> mx.array:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3(Module):
    """CSP bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: mx.array) -> mx.array:
        return self.cv3(mx.concatenate([self.m(self.cv1(x)), self.cv2(x)], axis=-1))


class C3k(C3):
    """C3 with `k x k` convolutions in the bottleneck."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C2f(Module):
    """Faster CSP bottleneck: split cv1 into two halves, run n bottlenecks, concat."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = ModuleList(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: mx.array) -> mx.array:
        y = list(split_sizes(self.cv1(x), 2, axis=-1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(mx.concatenate(y, axis=-1))


class C3k2(C2f):
    """C2f whose bottlenecks are either Bottleneck, C3k, or Bottleneck+PSABlock."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ) -> None:
        super().__init__(c1, c2, n, shortcut, g, e)
        if attn:
            blocks = [Sequential(Bottleneck(self.c, self.c, shortcut, g), PSABlock(self.c, 0.5, max(self.c // 64, 1)))]
        elif c3k:
            blocks = [C3k(self.c, self.c, 2, shortcut, g)]
        else:
            blocks = [Bottleneck(self.c, self.c, shortcut, g)]
        self.m = ModuleList(*(blocks[0] for _ in range(n)))


class SPPF(Module):
    """Spatial Pyramid Pooling - Fast: equivalent to SPP(k=(5, 9, 13))."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False) -> None:
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.k, self.n = k, n
        self.add = shortcut and c1 == c2

    def forward(self, x: mx.array) -> mx.array:
        y = [self.cv1(x)]
        for _ in range(self.n):
            y.append(MaxPool(self.k, 1, self.k // 2)(y[-1]))
        y = self.cv2(mx.concatenate(y, axis=-1))
        return x + y if self.add else y


class Attention(Module):
    """Position-sensitive multi-head attention with a 3x3 depthwise positional encoding."""

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        self.qkv = Conv(dim, dim + nh_kd * 2, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        n = h * w
        # (b, h, w, heads, 2*key_dim + head_dim) -> (b, heads, 2*key_dim + head_dim, h*w)
        qkv = (
            self.qkv(x).reshape(b, h, w, self.num_heads, -1).transpose(0, 3, 4, 1, 2).reshape(b, self.num_heads, -1, n)
        )
        q, k, v = split_sizes(qkv, [self.key_dim, self.key_dim, self.head_dim], axis=2)
        attn = mx.softmax((q * self.scale).transpose(0, 1, 3, 2) @ k, axis=-1)
        # (b, heads, head_dim, n) -> (b, h, w, c) with channel = head * head_dim + dim
        to_nhwc = lambda t: t.transpose(0, 3, 1, 2).reshape(b, h, w, c)  # noqa: E731
        out = to_nhwc(v @ attn.transpose(0, 1, 3, 2))
        return self.proj(out + self.pe(to_nhwc(v)))


class PSABlock(Module):
    """Attention block followed by a pointwise feed-forward, both with shortcuts."""

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: mx.array) -> mx.array:
        x = x + self.attn(x) if self.add else self.attn(x)
        return x + self.ffn(x) if self.add else self.ffn(x)


class C2PSA(Module):
    """C2f variant with a stack of PSA blocks on the second half of the split."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5) -> None:
        super().__init__()
        assert c1 == c2, "C2PSA requires c1 == c2"
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = Sequential(*(PSABlock(self.c, 0.5, max(self.c // 64, 1)) for _ in range(n)))

    def forward(self, x: mx.array) -> mx.array:
        a, b = split_sizes(self.cv1(x), 2, axis=-1)
        return self.cv2(mx.concatenate([a, self.m(b)], axis=-1))


class Concat(Module):
    """Concatenate a list of inputs along the channel (last) axis."""

    def __init__(self, dimension: int = 1) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, xs):
        if isinstance(xs, mx.array):
            xs = [xs]
        axis = -1 if self.dimension == 1 else self.dimension - 1
        return mx.concatenate(list(xs), axis=axis)


class Upsample(Module):
    """Nearest-neighbour resize by an integer factor."""

    def __init__(self, size=None, scale_factor: int = 2, mode: str = "nearest") -> None:
        super().__init__()
        if mode != "nearest":
            raise ValueError(f"unsupported upsample mode {mode!r}")
        self.scale_factor = scale_factor

    def forward(self, x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        s = self.scale_factor
        x = mx.broadcast_to(x.reshape(b, h, 1, w, 1, c), (b, h, s, w, s, c))
        return x.reshape(b, h * s, w * s, c)


class MaxPool(Module):
    """Max pooling with matching padding (MLX has no `padding='same'`)."""

    def __init__(self, k: int = 2, s: int = 1, p: int = 0) -> None:
        super().__init__()
        self.k, self.s, self.p = k, s, p

    def forward(self, x: mx.array) -> mx.array:
        # MLX pools with -inf padding, which is exactly the "same" padding SPPF needs
        return nn.MaxPool2d(self.k, stride=self.s, padding=self.p)(x)


def bias_init_array(fan_in: int, out_features: int) -> mx.array:
    """Kaiming-uniform-ish normal init for 2D weights, NHWC."""
    return mx.random.normal((out_features, 1, 1, fan_in)) * math.sqrt(1.0 / fan_in)
