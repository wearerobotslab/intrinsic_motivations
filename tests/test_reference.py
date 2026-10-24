"""Checks of the SOX reference (tests/sox_reference.py) against the theory.

If these pass, the reference is a trustworthy oracle for the other tests:
  * L is the Jacobian of the sensor-motor loop x -> A g(Cx + h) + S x + b,
  * the controller update (harmony = 0) is gradient descent on the
    time-loop error E = v^T v (or log E with loga), v = L^+ xi,
  * the model update learns the world,
  * the full controller self-amplifies a weak closed loop.
"""
import numpy as np
import pytest

from sox_reference import SoxConfig, SoxReference, load_state, random_state, lpz_pseudo_inverse


def _ref_with_state(ns, nm, conf, seed=0, t=15):
    rng = np.random.default_rng(seed)
    ref = SoxReference(ns, nm, conf)
    state = random_state(rng, ns, nm, conf.buffersize)
    load_state(ref, state, t)
    return ref


def test_lpz_pseudo_inverse_matches_moore_penrose():
    rng = np.random.default_rng(1)
    for shape in [(4, 4), (5, 3), (3, 5)]:
        M = rng.standard_normal(shape)
        np.testing.assert_allclose(lpz_pseudo_inverse(M), np.linalg.pinv(M), atol=1e-6)


@pytest.mark.parametrize("ns,nm", [(3, 3), (4, 2)])
def test_L_is_jacobian_of_the_loop(ns, nm):
    ref = _ref_with_state(ns, nm, SoxConfig(epsA=0, epsC=0))
    A, C, S, b, h = ref.A, ref.C, ref.S, ref.b, ref.h
    ref.learn()
    x = ref.last["x"]

    def loop(xx):
        return A @ np.tanh(C @ xx + h) + S @ xx + b

    eps = 1e-6
    J = np.column_stack([(loop(x + eps * e) - loop(x - eps * e)) / (2 * eps) for e in np.eye(ns)])
    np.testing.assert_allclose(ref.L, J, atol=1e-8)


def _E(A, C, S, h, x, xi):
    z = C @ x + h
    L = A @ ((1 - np.tanh(z) ** 2)[:, None] * C) + S
    v = np.linalg.solve(L, xi)
    return v @ v


@pytest.mark.parametrize("loga", [False, True])
def test_controller_update_is_gradient_descent_on_time_loop_error(loga):
    ns = nm = 3
    epsC = 1e-4  # small enough that clipping never kicks in
    conf = SoxConfig(epsA=0, epsC=epsC, damping=0, harmony=0, sense=1, loga=loga, pinv="svd")
    ref = _ref_with_state(ns, nm, conf, seed=3)
    A, C0, S, h0 = ref.A.copy(), ref.C.copy(), ref.S.copy(), ref.h.copy()
    ref.learn()
    x, xi, v = ref.last["x"], ref.last["xi"], ref.last["v"]
    EE = 0.1 / (v @ v + 0.001) if loga else 1.0

    d = 1e-6
    gC = np.zeros_like(C0)
    for i in range(nm):
        for j in range(ns):
            Cp, Cm = C0.copy(), C0.copy()
            Cp[i, j] += d
            Cm[i, j] -= d
            gC[i, j] = (_E(A, Cp, S, h0, x, xi) - _E(A, Cm, S, h0, x, xi)) / (2 * d)
    gh = np.zeros_like(h0)
    for i in range(nm):
        hp, hm = h0.copy(), h0.copy()
        hp[i] += d
        hm[i] -= d
        gh[i] = (_E(A, C0, S, hp, x, xi) - _E(A, C0, S, hm, x, xi)) / (2 * d)

    # Delta C = -(epsC * EE / 2) * dE/dC, same for h
    np.testing.assert_allclose(ref.C - C0, -0.5 * epsC * EE * gC, rtol=1e-4, atol=1e-12)
    np.testing.assert_allclose(ref.h - h0, -0.5 * epsC * EE * gh, rtol=1e-4, atol=1e-12)


def test_model_learns_a_linear_world():
    """With the controller frozen, A, S and b recover x' = A_true y + S_true x + b_true.

    Samples are fed straight into learn() (iid x), because in a closed loop
    y ~ Cx makes A and S hard to tell apart and the test would be about
    excitation rather than about the update rule.
    """
    rng = np.random.default_rng(4)
    ns = nm = 3
    A_true = np.array([[0.8, 0.2, 0.0], [-0.1, 0.9, 0.1], [0.0, 0.3, 0.7]])
    S_true = 0.2 * rng.standard_normal((ns, ns))
    b_true = 0.1 * rng.standard_normal(ns)
    conf = SoxConfig(epsC=0, epsA=0.05, damping=0, causeaware=0.0)
    bs = conf.buffersize
    ref = SoxReference(ns, nm, conf)
    ref.C = 2 * rng.standard_normal((nm, ns))  # fixed, saturating controller
    errs = []
    for _ in range(3000):
        x = rng.uniform(-1.5, 1.5, ns)
        y = np.tanh(ref.C @ x + ref.h)
        ref.t = 2 * bs
        ref.x_buffer[(ref.t - 1) % bs] = x
        ref.y_buffer[(ref.t - 1) % bs] = y
        ref.x_buffer[ref.t % bs] = A_true @ y + S_true @ x + b_true
        ref.learn()
        errs.append(np.linalg.norm(ref.last["xi"]))
    assert np.mean(errs[-100:]) < 0.01 * np.mean(errs[:20])
    np.testing.assert_allclose(ref.A, A_true, atol=0.01)
    np.testing.assert_allclose(ref.S, S_true, atol=0.01)
    np.testing.assert_allclose(ref.b, b_true, atol=0.01)


def test_controller_amplifies_weak_feedback_loop():
    """Homeokinesis drives a weakly coupled loop (x' = y + noise) away from the trivial fixed point."""
    rng = np.random.default_rng(5)
    ns = nm = 2
    conf = SoxConfig(init_feedback_strength=0.2)
    ref = SoxReference(ns, nm, conf)
    x = np.zeros(ns)
    ys = []
    for _ in range(2000):
        y = ref.step(x)
        x = y + 0.05 * rng.standard_normal(ns)
        ys.append(y)
    gain0 = 0.2
    assert np.max(np.abs(np.linalg.eigvals(ref.C))) > 2 * gain0
    assert np.mean(np.abs(ys[-200:])) > 0.3  # sustained, non-trivial activity
