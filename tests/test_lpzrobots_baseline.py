"""Numerical comparison with the ORIGINAL lpzrobots Sox (selforg/controller/sox.cpp).

Builds the real Sox from a local lpzrobots checkout, found in
external/lpzrobots (or wherever LPZROBOTS_SELFORG points); skipped otherwise:

    git clone --depth 1 https://github.com/georgmartius/lpzrobots external/lpzrobots
    pytest tests

1. The Python reference (sox_reference.py) must reproduce Sox to machine
   precision. This is what makes it a valid oracle for the other test files.
2. Each implementation in this repo is also run directly against Sox.
   Learning rates are kept small there so that clip limits, which are
   hard-coded and differ between ports, never become active, and only the
   maths is compared.
"""
import subprocess

import numpy as np
import pytest

from sox_reference import SoxConfig, SoxReference, load_state, random_state

SOX_DEFAULTS = dict(initFeedbackStrength=1.0, useExtendedModel=1, steps4Averaging=1, steps4Delay=1,
                    epsA=0.1, epsC=0.1, creativity=0.0, harmony=0.0, causeaware=0.01, sense=1.0,
                    damping=0.00001, loga=0)
CONFIGS = {
    "sox_defaults": SOX_DEFAULTS,
    "all_terms": dict(SOX_DEFAULTS, steps4Averaging=3, steps4Delay=2, creativity=0.2, harmony=0.05,
                      loga=1, damping=1e-4, sense=1.5),
    "no_extended_model": dict(SOX_DEFAULTS, useExtendedModel=0, causeaware=0.0, initFeedbackStrength=1.1),
}
# Small rates: no clip limit is reached in any port, so only the maths is compared.
SMALL_RATES = dict(SOX_DEFAULTS, epsA=0.005, epsC=0.005, steps4Averaging=2, steps4Delay=2,
                   creativity=0.1, harmony=0.02, damping=1e-4)
SHAPES = [(3, 3), (4, 2), (2, 4)]


def ref_conf(c):
    return SoxConfig(
        init_feedback_strength=c["initFeedbackStrength"], use_extended_model=bool(c["useExtendedModel"]),
        steps4averaging=c["steps4Averaging"], steps4delay=c["steps4Delay"], loga=bool(c["loga"]),
        epsC=c["epsC"], epsA=c["epsA"], sense=c["sense"], creativity=c["creativity"],
        damping=c["damping"], causeaware=c["causeaware"], harmony=c["harmony"], pinv="lpz",
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
    res = subprocess.run([str(driver)], input=inp, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    return [np.array(line.split(), dtype=float) for line in res.stdout.strip().splitlines()]


def sensor_stream(ns, T, seed=41):
    rng = np.random.default_rng(seed)
    return [np.sin(0.3 * k + np.arange(ns)) * 0.8 + 0.05 * rng.standard_normal(ns) for k in range(T)]


def sox_outputs(driver, ns, nm, conf, xs):
    return np.array(run(driver, "sox", ns, nm, conf, [len(xs)] + [v for x in xs for v in x]))


# -- 1. reference == lpzrobots Sox ------------------------------------------------

@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("ns,nm", SHAPES)
def test_reference_learn_step_equals_lpzrobots(lpz_sox_driver, cfg, ns, nm):
    conf = CONFIGS[cfg]
    state = random_state(np.random.default_rng(40), ns, nm, 10)
    t = 17
    body = [t]
    for name in ("A", "C", "S", "b", "h", "v_avg"):
        body += list(np.ravel(state[name]))
    for v in state["x_buffer"] + state["y_buffer"]:
        body += list(v)
    out = dict(zip(("A", "C", "S", "b", "h", "v_avg", "L"), run(lpz_sox_driver, "learn", ns, nm, conf, body)))
    ref = SoxReference(ns, nm, ref_conf(conf))
    load_state(ref, state, t)
    ref.learn()
    for name, got in out.items():
        np.testing.assert_allclose(np.ravel(getattr(ref, name)), got, rtol=0, atol=1e-12, err_msg=name)


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("ns,nm", SHAPES)
def test_reference_trajectory_equals_lpzrobots(lpz_sox_driver, cfg, ns, nm):
    conf = CONFIGS[cfg]
    xs = sensor_stream(ns, 300)
    ys = sox_outputs(lpz_sox_driver, ns, nm, conf, xs)
    ref = SoxReference(ns, nm, ref_conf(conf))
    yr = np.array([ref.step(x) for x in xs])
    np.testing.assert_allclose(yr, ys, rtol=0, atol=1e-9)


# -- 2. each implementation == lpzrobots Sox -------------------------------------

def numpy_outputs(numpy_impl, ns, nm, c, xs):
    hk = numpy_impl.Homeokinesis(ns, nm)
    hk.init_feedback_strength = c["initFeedbackStrength"]
    hk.C = np.eye(nm, ns) * c["initFeedbackStrength"]
    hk.use_extended_model = bool(c["useExtendedModel"])
    hk.steps_for_averaging, hk.steps_for_delay = c["steps4Averaging"], c["steps4Delay"]
    hk.eps_A, hk.eps_C = c["epsA"], c["epsC"]
    hk.creativity, hk.harmony, hk.causeaware = c["creativity"], c["harmony"], c["causeaware"]
    hk.sense, hk.damping, hk.loga = c["sense"], c["damping"], bool(c["loga"])
    hk.clip_limit = 0.05
    out = []
    for x in xs:
        y = hk.step(np.array(x))
        out.append(np.full(nm, np.nan) if y is None else y)
    return np.array(out)


def torch_outputs(torch_impl, ns, nm, c, xs, monkeypatch):
    torch = torch_impl.torch
    monkeypatch.setattr(torch, "normal", lambda mean, std, size, device=None, **kw: torch.zeros(size, device=device))
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        conf = dict(device="cpu", init_feedback_strength=c["initFeedbackStrength"],
                    use_extended_model=bool(c["useExtendedModel"]), buffersize=10,
                    steps_for_averaging=c["steps4Averaging"], steps_for_delay=c["steps4Delay"],
                    creativity=c["creativity"], cause_aware=c["causeaware"], sense=c["sense"],
                    harmony=c["harmony"], loga=bool(c["loga"]), epsilon_A=c["epsA"], epsilon_C=c["epsC"],
                    factor_S=1.0, factor_b=1.0, factor_h=1.0, damping=c["damping"])
        hk = torch_impl.Homeokinesis(dict(config=conf, num_inputs=ns, num_outputs=nm, num_envs=1))
        return np.array([hk.step(torch.tensor(np.array(x)[None, :])).reshape(-1).numpy() for x in xs])
    finally:
        torch.set_default_dtype(old)


@pytest.mark.parametrize("impl", ["numpy", "torch", "cpp_update", "cpp_sox_order"])
@pytest.mark.parametrize("ns,nm", SHAPES)
def test_implementation_trajectory_equals_lpzrobots(request, lpz_sox_driver, impl, ns, nm, monkeypatch):
    c = SMALL_RATES
    xs = sensor_stream(ns, 200)
    ys = sox_outputs(lpz_sox_driver, ns, nm, c, xs)
    if impl == "numpy":
        try:
            got = numpy_outputs(request.getfixturevalue("numpy_impl"), ns, nm, c, xs)
        except Exception as e:
            pytest.fail(f"numpy port raised {e!r}")
    elif impl == "torch":
        got = torch_outputs(request.getfixturevalue("torch_impl"), ns, nm, c, xs, monkeypatch)
    else:
        mode = "update" if impl == "cpp_update" else "sox"
        got = np.array(run(request.getfixturevalue("cpp_driver"), mode, ns, nm, c,
                           [len(xs)] + [v for x in xs for v in x]))
    # 1e-7: the ports use SVD/QR pseudo-inverses, Sox uses the normal equations, so for
    # non-square A the rounding differs at ~1e-9. Real discrepancies are >= 1e-3.
    tol = 1e-7
    diff = np.abs(got - ys)
    first_bad = int(np.argmax(np.any(~(diff <= tol), axis=1))) if np.any(~(diff <= tol)) else None
    assert first_bad is None, (f"motor outputs differ from lpzrobots Sox from step {first_bad} on; "
                               f"max |diff| = {np.nanmax(diff):.3g}, NaN/None steps = {int(np.isnan(diff).any(1).sum())}")
