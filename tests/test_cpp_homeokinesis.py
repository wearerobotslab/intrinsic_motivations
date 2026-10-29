"""homeokinesis/cpp/eigen against the SOX reference.

The controller is compiled with stub ARE headers (tests/cpp/stubs) and
driven by tests/cpp/driver.cpp. Two run modes:
  * "update": the public Homeokinesis::update(), as used in ARE,
  * "sox":    step()/learn() called in the same order as the original Sox::step.
The single-learn-step tests check learn() in isolation.
"""
import subprocess

import numpy as np
import pytest

from sox_reference import SoxConfig, SoxReference, load_state, random_state

ATOL = 1e-9

# Defaults from Homeokinesis::_set_default_config and the member initialisers.
CPP_DEFAULTS = dict(
    initFeedbackStrength=1.1, useExtendedModel=0, steps4Averaging=10, steps4Delay=10,
    epsA=0.1, epsC=0.05, creativity=0.1, harmony=0.1, causeaware=0.01, sense=1.0,
    damping=0.00001, loga=1,
)
SOX_DEFAULTS = dict(
    initFeedbackStrength=1.0, useExtendedModel=1, steps4Averaging=1, steps4Delay=1,
    epsA=0.1, epsC=0.1, creativity=0.0, harmony=0.0, causeaware=0.01, sense=1.0,
    damping=0.00001, loga=0,
)
CONFIGS = {"cpp_defaults": CPP_DEFAULTS, "sox_defaults": SOX_DEFAULTS,
           "sox_ext_harmony": dict(SOX_DEFAULTS, harmony=0.05, loga=1, steps4Delay=3)}


def ref_conf(c):
    return SoxConfig(
        init_feedback_strength=c["initFeedbackStrength"], use_extended_model=bool(c["useExtendedModel"]),
        steps4averaging=c["steps4Averaging"], steps4delay=c["steps4Delay"],
        loga=bool(c["loga"]), epsC=c["epsC"], epsA=c["epsA"], sense=c["sense"],
        creativity=c["creativity"], damping=c["damping"], causeaware=c["causeaware"],
        harmony=c["harmony"], buffersize=10, pinv="svd",
    )


def _num(v):
    """Plain text for the C++ drivers (numpy 2 would print np.float64(...))."""
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return repr(float(v))


def run(driver, mode, ns, nm, conf, body):
    head = [mode, ns, nm, len(conf)] + [f"{k} {_num(v)}" for k, v in conf.items()]
    inp = " ".join(map(str, head)) + "\n" + " ".join(_num(v) for v in body) + "\n"
    res = subprocess.run([str(driver)], input=inp, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    return [np.array(line.split(), dtype=float) for line in res.stdout.strip().splitlines()]


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("ns,nm", [(3, 3), (4, 2)])
def test_single_learn_step_matches_sox(cpp_driver, cfg, ns, nm):
    conf = CONFIGS[cfg]
    state = random_state(np.random.default_rng(30), ns, nm, 10)
    t = 17
    body = [t]
    for name in ("A", "C", "S", "b", "h", "v_avg"):
        body += list(np.ravel(state[name]))
    for v in state["x_buffer"] + state["y_buffer"]:
        body += list(v)
    A, C, S, b, h, v_avg = run(cpp_driver, "learn", ns, nm, conf, body)

    ref = SoxReference(ns, nm, ref_conf(conf))
    load_state(ref, state, t)
    ref.learn()
    for name, got in zip(("A", "C", "S", "b", "h", "v_avg"), (A, C, S, b, h, v_avg)):
        np.testing.assert_allclose(got, np.ravel(getattr(ref, name)), atol=ATOL, err_msg=name)


def sensor_stream(ns, T=60, seed=31):
    rng = np.random.default_rng(seed)
    return [np.sin(0.3 * k + np.arange(ns)) * 0.8 + 0.05 * rng.standard_normal(ns) for k in range(T)]


@pytest.mark.parametrize("mode", ["sox", "update"])
@pytest.mark.parametrize("cfg", CONFIGS)
def test_trajectory_matches_sox(cpp_driver, mode, cfg):
    """'sox' validates step()+learn(); 'update' validates the public update() that ARE calls."""
    ns, nm = 3, 3
    conf = CONFIGS[cfg]
    xs = sensor_stream(ns)
    ys = run(cpp_driver, mode, ns, nm, conf, [len(xs)] + [v for x in xs for v in x])
    ref = SoxReference(ns, nm, ref_conf(conf))
    for k, x in enumerate(xs):
        np.testing.assert_allclose(ys[k], ref.step(x), atol=1e-7, err_msg=f"step {k}")


def test_add_noise_is_scaled_by_e(cpp_driver):
    """add_noise(e) must change every parameter by at most e (Eigen's Random is in [-1, 1])."""
    e, *changes = run(cpp_driver, "noise", 3, 3, {}, [0.01])[0]
    for name, d in zip(("A", "S", "C", "b", "h"), changes):
        assert d <= e, f"{name} changed by {d:.3g} > e = {e}"


def test_clone_continues_identically(cpp_driver):
    """A clone taken mid-run must produce the same outputs as the original from then on."""
    xs = sensor_stream(3, T=67)  # clone after 33 steps: t % buffersize != 0
    for line in run(cpp_driver, "clone", 3, 3, SOX_DEFAULTS, [len(xs)] + [v for x in xs for v in x]):
        np.testing.assert_array_equal(line[:3], line[3:])
