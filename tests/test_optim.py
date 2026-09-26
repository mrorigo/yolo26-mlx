"""MuSGD / Newton-Schulz tests."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from yolo26_mlx.optim import MuSGD, muon_param_groups, newton_schulz


def test_newton_schulz_orthogonalises():
    """The iteration should flatten the spectrum: singular values land in the 0.5-1.5 band."""
    g = mx.array([[3.0, 0.0], [0.0, 1.0]])
    out = np.array(newton_schulz(g))
    singular = np.linalg.svd(out, compute_uv=False)
    assert singular.min() >= 0.5 and singular.max() <= 1.5
    assert singular.max() / singular.min() < 3.0  # the raw 3:1 spread is gone


def test_newton_schulz_handles_tall_and_batched():
    tall = newton_schulz(mx.array(np.random.normal(size=(6, 2))))
    assert tall.shape == (6, 2)
    batch = newton_schulz(mx.array(np.random.normal(size=(3, 4, 4))))
    assert batch.shape == (3, 4, 4)
    assert np.isfinite(np.array(batch)).all()


def test_sgd_group_moves_downhill():
    params = {"w": mx.ones((4, 3)), "b": mx.zeros((4,))}
    opt = MuSGD(params, [{"keys": ["w"]}, {"keys": ["b"]}], lr=0.1, momentum=0.9, weight_decay=0.0)
    opt.step({"w": mx.full((4, 3), 1.0), "b": mx.full((4,), 1.0)})
    assert np.all(np.array(params["w"]) < 1.0)
    assert np.all(np.array(params["b"]) < 0.0)


def test_nesterov_momentum_accelerates():
    """Nesterov should travel further than plain momentum on a constant gradient."""
    plain, nesterov = {"w": mx.ones((2, 2))}, {"w": mx.ones((2, 2))}

    def run(params, nesterov_flag):
        opt = MuSGD(params, [{"keys": ["w"], "nesterov": nesterov_flag}], lr=0.1, momentum=0.9, weight_decay=0.0)
        for _ in range(3):
            opt.step({"w": mx.ones((2, 2))})
        return float(params["w"][0, 0])

    assert run(nesterov, True) < run(plain, False)


def test_muon_group_flattens_the_update_spectrum():
    """Pure Muon: the update is orthogonalized, so it is not parallel to the raw gradient."""
    params = {"w": mx.zeros((4, 4))}
    opt = MuSGD(params, [{"keys": ["w"], "use_muon": True}], lr=0.1, momentum=0.0, weight_decay=0.0, muon=1.0, sgd=0.0)
    grad = np.diag([8.0, 4.0, 2.0, 1.0])  # strongly anisotropic
    before = np.array(params["w"]).copy()
    opt.step({"w": mx.array(grad)})
    delta = np.array(params["w"]) - before
    singular = np.linalg.svd(delta, compute_uv=False)
    # an SGD step would keep the 8:1 spread of the gradient; Muon flattens it towards 1
    assert singular.max() / singular.min() < 2.5
    assert abs(singular.max() / singular.min() - 1.0) < abs(8.0 - 1.0)


def test_hybrid_mode_applies_both_updates():
    """The MuSGD group gets the orthogonalized update *and* the SGD update."""
    grad = np.diag([8.0, 4.0, 2.0, 1.0])

    def run(muon, sgd):
        params = {"w": mx.zeros((4, 4))}
        opt = MuSGD(
            params,
            [{"keys": ["w"], "use_muon": True}],
            lr=0.1,
            momentum=0.0,
            weight_decay=0.0,
            muon=muon,
            sgd=sgd,
        )
        opt.step({"w": mx.array(grad)})
        return np.array(params["w"])

    only_muon, only_sgd, hybrid = run(1.0, 0.0), run(0.0, 1.0), run(1.0, 1.0)
    assert np.allclose(hybrid, only_muon + only_sgd, atol=1e-5)
    assert np.linalg.svd(only_sgd, compute_uv=False).max() / np.linalg.svd(only_sgd, compute_uv=False).min() > 4.0


def test_weight_decay_only_affects_its_group():
    params = {"w": mx.ones((2, 2)), "b": mx.ones((2,))}
    opt = MuSGD(
        params,
        [{"keys": ["w"], "weight_decay": 0.5}, {"keys": ["b"], "weight_decay": 0.0}],
        lr=0.1,
        momentum=0.0,
    )
    opt.step({"w": mx.zeros((2, 2)), "b": mx.zeros((2,))})
    # with a zero gradient, decay alone still pulls w down but leaves b alone
    assert float(params["w"][0, 0]) < 1.0
    assert float(params["b"][0]) == 1.0


def test_param_groups_split_by_rank_and_role():
    class Hyp:
        lr0 = 0.01
        weight_decay = 5e-4

    from yolo26_mlx import build_model

    model = build_model("yolo26", "n", nc=2, imgsz=64, verbose=False)
    groups = muon_param_groups(model, Hyp())
    keys = {g["param_group"]: set(g["keys"]) for g in groups}
    assert keys["muon"] and keys["norm"]
    # every trainable parameter lands in exactly one group
    assert keys["muon"] | keys["norm"] | keys["weight"] == set(model.parameters())
    assert not (keys["muon"] & keys["norm"])
    assert all(model.parameters()[k].ndim >= 2 for k in keys["muon"])
    assert all(model.parameters()[k].ndim < 2 for k in keys["norm"] | keys["weight"])
    assert all("bias" in k or ".bn." in k for k in keys["norm"])
