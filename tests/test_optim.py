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


def _eager_musgd(params, grads, muon, sgd, lr, beta, weight_decay, nesterov, use_muon, steps):
    """Straight transcription of the MuSGD specification, used to check the compiled implementation.

    Deliberately unoptimised: no grouping, no batching, plain float32 arithmetic.
    """
    import numpy as np

    from yolo26_mlx.optim.musgd import newton_schulz

    p = {k: np.array(v) for k, v in params.items()}
    muon_buf = {k: np.zeros_like(v) for k, v in p.items()}
    sgd_buf = {k: np.zeros_like(v) for k, v in p.items()}
    for _ in range(steps):
        for k, g0 in grads.items():
            g = np.array(g0)
            lr_eff = lr
            if use_muon and p[k].ndim >= 2:
                muon_buf[k] = beta * muon_buf[k] + (1 - beta) * g
                u = beta * muon_buf[k] + (1 - beta) * g if nesterov else muon_buf[k]
                # a conv weight is orthogonalized as one (rows, cols) matrix, not per spatial tap
                if u.ndim > 2:
                    u = u.reshape(u.shape[0], -1)
                u = np.array(newton_schulz(mx.array(u))).reshape(p[k].shape)
                scale = max(1.0, p[k].shape[-2] / p[k].shape[-1]) ** 0.5
                p[k] = p[k] - lr * muon * scale * u
                lr_eff = lr * sgd
            gp = g + weight_decay * p[k] if weight_decay else g
            sgd_buf[k] = beta * sgd_buf[k] + gp
            p[k] = p[k] - lr_eff * (gp + beta * sgd_buf[k] if nesterov else sgd_buf[k])
    return p, muon_buf, sgd_buf


def test_compiled_passes_match_the_specification():
    """The compiled SGD and Muon passes must agree with a literal transcription of the spec.

    This is the regression guard for batching the per-tensor work: the eager transcription is the
    oracle, and it encodes the two buffer recurrences and the post-Muon weight-decay ordering.
    """
    mx.random.seed(0)
    shapes = {"w2d": (16, 8), "w4d": (12, 4, 3, 3), "bias": (16,), "norm": (8,)}
    params = {k: mx.random.normal(s) for k, s in shapes.items()}
    opt = MuSGD(params, [{"keys": list(params), "use_muon": True}], lr=0.1, momentum=0.9,
                weight_decay=0.01, nesterov=True, muon=0.2, sgd=1.0)
    grads = {k: mx.random.normal(s) * 0.5 for k, s in shapes.items()}
    initial = {k: np.array(v) for k, v in params.items()}  # the eager oracle must start here
    for _ in range(3):
        opt.step(grads)
    ref_p, ref_m, ref_s = _eager_musgd(initial, grads, 0.2, 1.0, 0.1, 0.9, 0.01, True, True, steps=3)
    for k in params:
        assert np.allclose(np.array(params[k]), ref_p[k], atol=1e-5), f"parameter {k}"
    for k in params:
        if params[k].ndim >= 2:
            assert np.allclose(np.array(opt.state[k]["muon_buf"]), ref_m[k], atol=1e-6), f"muon buffer {k}"
    for k in params:
        assert np.allclose(np.array(opt.state[k]["buf"]), ref_s[k], atol=1e-5), f"sgd buffer {k}"


def test_plain_group_matches_the_specification():
    """Non-Nesterov and non-muon paths, including weight decay."""
    mx.random.seed(1)
    params = {"a": mx.random.normal((6, 3)), "b": mx.random.normal((5,))}
    opt = MuSGD(params, [{"keys": list(params)}], lr=0.05, momentum=0.8, weight_decay=0.1, nesterov=False)
    grads = {k: mx.random.normal(v.shape) for k, v in params.items()}
    initial = {k: np.array(v) for k, v in params.items()}
    for _ in range(2):
        opt.step(grads)
    ref_p, _, ref_s = _eager_musgd(initial, grads, 0.2, 1.0, 0.05, 0.8, 0.1, False, False, steps=2)
    for k in params:
        assert np.allclose(np.array(params[k]), ref_p[k], atol=1e-6), f"parameter {k}"
        assert np.allclose(np.array(opt.state[k]["buf"]), ref_s[k], atol=1e-6), f"buffer {k}"


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
