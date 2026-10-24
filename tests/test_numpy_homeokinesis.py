"""python/homeokinesis/numpy/homeokinesis.py against the SOX reference.

Every test runs the port and the reference with the port's own
hyperparameters, so only the maths is compared, not the tuning.
"""
import numpy as np
import pytest

from sox_reference import SoxConfig, SoxReference, load_state, random_state

ATOL = 1e-9


def make(numpy_impl, ns, nm, **overrides):
    hk = numpy_impl.Homeokinesis(ns, nm)
    for k, v in overrides.items():
        setattr(hk, k, v)
    conf = SoxConfig(
        init_feedback_strength=hk.init_feedback_strength,
        use_extended_model=hk.use_extended_model,
        steps4averaging=hk.steps_for_averaging,
        steps4delay=hk.steps_for_delay,
        factor_S=hk.factor_S, factor_b=hk.factor_b, factor_h=hk.factor_h,
        loga=hk.loga, epsC=hk.eps_C, epsA=hk.eps_A, sense=hk.sense,
        creativity=hk.creativity, damping=hk.damping, causeaware=hk.causeaware,
        harmony=hk.harmony, buffersize=hk.buffer_size,
        clip_A=hk.clip_limit, clip_A_damp=hk.clip_limit, clip_S=hk.clip_limit,
        clip_b=hk.clip_limit, clip_C=hk.clip_limit, clip_C_damp=hk.clip_limit,
        clip_h=hk.clip_limit, pinv="svd",
    )
    return hk, SoxReference(ns, nm, conf)


def load_impl_state(hk, state, t):
    hk.A, hk.C, hk.S = state["A"].copy(), state["C"].copy(), state["S"].copy()
    hk.b, hk.h, hk.v_avg = state["b"].copy(), state["h"].copy(), state["v_avg"].copy()
    hk.x_buffer = np.stack(state["x_buffer"], axis=1).astype(float)
    hk.y_buffer = np.stack(state["y_buffer"], axis=1).astype(float)
    hk.t = t


def assert_same_params(hk, ref):
    for name in ("A", "S", "b", "C", "h", "v_avg"):
        np.testing.assert_allclose(getattr(hk, name), getattr(ref, name), atol=ATOL, err_msg=name)


def test_initial_matrix_shapes(numpy_impl):
    ns, nm = 5, 3
    hk = numpy_impl.Homeokinesis(ns, nm)
    assert hk.A.shape == (ns, nm)
    assert hk.C.shape == (nm, ns)
    assert hk.S.shape == (ns, ns), "S maps sensors to sensors"


# Big learning rates and clip limits so that every term of the update shows up.
LEARN_CASES = {
    "default": {},
    "sox_rates": dict(eps_A=0.1, eps_C=0.1, clip_limit=0.1),
    "harmony_loga": dict(eps_A=0.1, eps_C=0.1, clip_limit=0.1, harmony=0.05, loga=True),
}


@pytest.mark.parametrize("case", LEARN_CASES)
@pytest.mark.parametrize("ns,nm", [(3, 3), (4, 2)])
def test_single_learn_step_matches_sox(numpy_impl, case, ns, nm):
    try:
        hk, ref = make(numpy_impl, ns, nm, **LEARN_CASES[case])
    except Exception as e:  # pragma: no cover - reported as failure
        pytest.fail(f"construction failed: {e!r}")
    state = random_state(np.random.default_rng(10), ns, nm, hk.buffer_size)
    t = 17
    load_impl_state(hk, state, t)
    load_state(ref, state, t)
    ref.learn()
    hk.learn()
    assert_same_params(hk, ref)


def test_step_returns_motor_values_while_buffer_fills(numpy_impl):
    """Sox::step always writes motor values; it only skips *learning* while t <= buffersize."""
    hk, ref = make(numpy_impl, 3, 3)
    for k in range(hk.buffer_size + 2):
        x = np.full(3, 0.1 * k)
        y = hk.step(x)
        y_ref = ref.step(x)
        assert y is not None, f"step {k} returned None"
        np.testing.assert_allclose(y, y_ref, atol=ATOL)


@pytest.mark.parametrize("ns,nm", [(3, 3), (4, 2)])
def test_trajectory_matches_sox(numpy_impl, ns, nm):
    """Feed the same sensor stream to both and compare motor outputs and parameters."""
    try:
        hk, ref = make(numpy_impl, ns, nm, eps_A=0.1, eps_C=0.1, clip_limit=0.1)
    except Exception as e:
        pytest.fail(f"construction failed: {e!r}")
    rng = np.random.default_rng(11)
    for k in range(60):
        x = np.sin(0.3 * k + np.arange(ns)) * 0.8 + 0.05 * rng.standard_normal(ns)
        y_ref = ref.step(x)
        y = hk.step(x)
        if k > hk.buffer_size:
            np.testing.assert_allclose(y, y_ref, atol=1e-7, err_msg=f"step {k}")
    assert_same_params(hk, ref)
