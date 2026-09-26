"""Model graph, layer shapes and parameter-count checks."""

from __future__ import annotations

import mlx.core as mx
import pytest

from yolo26_mlx import build_model
from yolo26_mlx.config import load_config, make_divisible, scale_config
from yolo26_mlx.nn.block import C2PSA, SPPF, Attention, C3k2
from yolo26_mlx.nn.conv import BatchNorm, Conv
from yolo26_mlx.tasks import DetectionModel

# Published parameter counts for the dual-head training model (Ultralytics YOLO26)
EXPECTED_PARAMS = {
    "n": 2_572_280,
    "s": 10_009_784,
    "m": 21_896_248,
    "l": 26_299_704,
    "x": 58_993_368,
}


@pytest.mark.parametrize("scale,expected", sorted(EXPECTED_PARAMS.items()))
def test_parameter_counts_match_published(scale, expected):
    """Every scale must have exactly the published parameter count: no layers missing or doubled."""
    model = DetectionModel("yolo26", scale, verbose=False)
    assert model.num_parameters() == expected


def test_scale_math():
    assert make_divisible(100) == 104
    assert make_divisible(8, 8) == 8
    specs, meta = scale_config(load_config("yolo26", "n"))
    assert meta == {"scale": "n", "nc": 80, "reg_max": 1, "end2end": True}
    # the C3k2 blocks of m/l/x switch to C3k kernels
    m_specs, _ = scale_config(load_config("yolo26", "m"))
    assert m_specs[2].args[3] is True
    assert specs[2].args[3] is False


def test_layer_shapes_and_strides():
    model = build_model("yolo26", "n", imgsz=64, verbose=False)
    assert model.stride == [8.0, 16.0, 32.0]
    model.train(False)
    out = model(mx.zeros((1, 64, 64, 3)))
    assert out.shape == (1, 84, 6)  # NMS-free top-k, capped by the 84 anchors of a 64px input


def test_training_mode_returns_both_branches():
    model = build_model("yolo26", "n", imgsz=64, verbose=False)
    model.train(True)
    preds = model(mx.zeros((2, 64, 64, 3)))
    assert set(preds) == {"one2many", "one2one"}
    assert preds["one2many"]["boxes"].shape == (2, 84, 4)  # 8x8 + 4x4 + 2x2 anchors
    assert preds["one2many"]["scores"].shape == (2, 84, 80)
    assert len(preds["one2many"]["feats"]) == 3


def test_p2_variant_builds():
    model = build_model("yolo26-p2", "n", imgsz=64, verbose=False)
    assert model.stride == [4.0, 8.0, 16.0, 32.0]
    assert model.head.nl == 4


def test_conv_and_batchnorm_fold():
    conv = Conv(4, 8, 3, act=True)
    conv.eval()
    fused = conv.fuse()
    x = mx.random.normal((2, 16, 16, 4))
    assert mx.allclose(fused(x), conv(x), atol=1e-5).item()
    # the folded layer is a plain conv: no BatchNorm left, and a bias appeared
    assert isinstance(fused, type(conv.conv))
    assert fused.bias is not None
    assert not any(isinstance(m, BatchNorm) for _, m in fused.named_modules())


def test_block_invariants():
    x = mx.random.normal((1, 8, 8, 32))
    assert SPPF(32, 32, 5, 3, True)(x).shape == (1, 8, 8, 32)
    assert C3k2(32, 64, 1, False, 0.25)(x).shape == (1, 8, 8, 64)
    assert C2PSA(32, 32, 1)(x).shape == (1, 8, 8, 32)
    # attention output keeps the input channel count regardless of head count
    attn = Attention(32, num_heads=2, attn_ratio=0.5)
    assert attn(x).shape == x.shape


def test_weight_roundtrip(tmp_path):
    model = build_model("yolo26", "n", imgsz=64, verbose=False)
    path = tmp_path / "w.safetensors"
    model.save_weights(str(path), {"epochs": "3"})
    other = build_model("yolo26", "n", imgsz=64, verbose=False)
    other.load_weights(str(path))
    x = mx.random.normal((1, 64, 64, 3))
    model.train(False)
    other.train(False)
    assert mx.allclose(model(x), other(x), atol=1e-6).item()
