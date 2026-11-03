"""homeokinesis/cpp/portable against the SOX reference.

The header-only controller is driven by tests/cpp/portable_driver.cpp, which
takes the same input as the Eigen driver. In double it must equal the
reference; in float (as on the ESP32) it must stay close to it.
"""
import numpy as np
import pytest

from sox_reference import SoxReference, load_state, random_state
from test_cpp_homeokinesis import CONFIGS, ref_conf, run, sensor_stream

ATOL = 1e-9


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("ns,nm", [(3, 3), (4, 2)])
def test_portable_single_learn_step_matches_sox(portable_driver, cfg, ns, nm):
    conf = CONFIGS[cfg]
    state = random_state(np.random.default_rng(30), ns, nm, 10)
    t = 17
    body = [t]
    for name in ("A", "C", "S", "b", "h", "v_avg"):
        body += list(np.ravel(state[name]))
    for v in state["x_buffer"] + state["y_buffer"]:
        body += list(v)
    got = run(portable_driver, "learn", ns, nm, conf, body)

    ref = SoxReference(ns, nm, ref_conf(conf))
    load_state(ref, state, t)
    ref.learn()
    for name, values in zip(("A", "C", "S", "b", "h", "v_avg"), got):
        np.testing.assert_allclose(values, np.ravel(getattr(ref, name)), atol=ATOL, err_msg=name)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_portable_trajectory_matches_sox(portable_driver, cfg):
    ns, nm = 3, 3
    conf = CONFIGS[cfg]
    xs = sensor_stream(ns)
    ys = run(portable_driver, "sox", ns, nm, conf, [len(xs)] + [v for x in xs for v in x])
    ref = SoxReference(ns, nm, ref_conf(conf))
    for k, x in enumerate(xs):
        np.testing.assert_allclose(ys[k], ref.step(x), atol=1e-7, err_msg=f"step {k}")


@pytest.mark.parametrize("cfg", CONFIGS)
def test_portable_float_stays_close_to_sox(portable_driver, cfg):
    """The float version (as run on the ESP32) follows the reference within float rounding."""
    ns, nm = 3, 3
    conf = CONFIGS[cfg]
    xs = sensor_stream(ns)
    ys = run(portable_driver, "sox_float", ns, nm, conf, [len(xs)] + [v for x in xs for v in x])
    ref = SoxReference(ns, nm, ref_conf(conf))
    for k, x in enumerate(xs):
        np.testing.assert_allclose(ys[k], ref.step(x), atol=1e-3, err_msg=f"step {k}")


def test_portable_add_noise_is_scaled_by_e(portable_driver):
    """add_noise(e) must change every parameter by at most e (Eigen's Random is in [-1, 1])."""
    e, *changes = run(portable_driver, "noise", 3, 3, {}, [0.01])[0]
    for name, d in zip(("A", "S", "C", "b", "h"), changes):
        assert d <= e, f"{name} changed by {d:.3g} > e = {e}"
