"""homeokinesis/python/homeokinesis_torch.py against the SOX reference.

The torch port runs a batch of independent controllers (num_envs); every
env is checked against its own reference instance. Tests run in float64
and, unless stated otherwise, with the port's sensor/motor noise disabled.
"""
import numpy as np
import pytest

from sox_reference import SoxConfig, SoxReference, load_state, random_state

torch = pytest.importorskip("torch")

ATOL = 1e-9

BASE_CONF = dict(
    device="cpu",
    init_feedback_strength=1.0,
    use_extended_model=True,
    buffersize=10,
    steps_for_averaging=1,
    steps_for_delay=1,
    creativity=0.0,
    cause_aware=0.01,
    sense=1.0,
    harmony=0.0,
    loga=False,
    epsilon_A=0.1,
    epsilon_C=0.1,
    factor_S=1.0,
    factor_b=1.0,
    factor_h=1.0,
    damping=0.00001,
)

# clip limits hard-coded in the torch port's learn()
TORCH_CLIPS = dict(clip_A=0.01, clip_A_damp=0.1, clip_S=0.1, clip_b=0.1,
                   clip_C=0.05, clip_C_damp=0.05, clip_h=0.05)


@pytest.fixture(autouse=True)
def float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


@pytest.fixture
def no_noise(torch_impl, monkeypatch):
    def zeros(mean, std, size, device=None, **kw):
        return torch.zeros(size, device=device)
    monkeypatch.setattr(torch_impl.torch, "normal", zeros)


def ref_conf(c, clips=TORCH_CLIPS):
    return SoxConfig(
        init_feedback_strength=c["init_feedback_strength"],
        use_extended_model=c["use_extended_model"],
        steps4averaging=c["steps_for_averaging"], steps4delay=c["steps_for_delay"],
        factor_S=c["factor_S"], factor_b=c["factor_b"], factor_h=c["factor_h"],
        loga=c["loga"], epsC=c["epsilon_C"], epsA=c["epsilon_A"], sense=c["sense"],
        creativity=c["creativity"], damping=c["damping"], causeaware=c["cause_aware"],
        harmony=c["harmony"], buffersize=c["buffersize"], pinv="svd", **clips,
    )


def make(torch_impl, ns, nm, num_envs=3, **overrides):
    c = dict(BASE_CONF, **overrides)
    hk = torch_impl.Homeokinesis(dict(config=c, num_inputs=ns, num_outputs=nm, num_envs=num_envs))
    refs = [SoxReference(ns, nm, ref_conf(c)) for _ in range(num_envs)]
    return hk, refs


def t2n(v):
    return v.detach().cpu().numpy()


def load_impl_state(hk, states, t):
    def col(a):
        return torch.tensor(np.asarray(a))[..., None]
    hk.A = torch.tensor(np.stack([s["A"] for s in states]))
    hk.C = torch.tensor(np.stack([s["C"] for s in states]))
    hk.S = torch.tensor(np.stack([s["S"] for s in states]))
    hk.b = col(np.stack([s["b"] for s in states]))
    hk.h = col(np.stack([s["h"] for s in states]))
    hk.v_avg = col(np.stack([s["v_avg"] for s in states]))
    hk.x_buffer = col(np.stack([np.stack(s["x_buffer"]) for s in states]))
    hk.y_buffer = col(np.stack([np.stack(s["y_buffer"]) for s in states]))
    hk.t = t


def assert_same_params(hk, refs, atol=ATOL):
    for i, ref in enumerate(refs):
        for name in ("A", "S", "C"):
            np.testing.assert_allclose(t2n(getattr(hk, name))[i], getattr(ref, name), atol=atol,
                                       err_msg=f"env {i}: {name}")
        for name in ("b", "h", "v_avg"):
            np.testing.assert_allclose(t2n(getattr(hk, name))[i, :, 0], getattr(ref, name), atol=atol,
                                       err_msg=f"env {i}: {name}")


LEARN_CASES = {
    "default": {},
    "harmony": dict(harmony=0.05),
    "loga": dict(loga=True),
    "delay_avg_creativity": dict(steps_for_delay=3, steps_for_averaging=4, creativity=0.3),
}


@pytest.mark.parametrize("case", LEARN_CASES)
@pytest.mark.parametrize("ns,nm", [(3, 3), (4, 2)])
def test_single_learn_step_matches_sox(torch_impl, case, ns, nm):
    hk, refs = make(torch_impl, ns, nm, **LEARN_CASES[case])
    rng = np.random.default_rng(20)
    states = [random_state(rng, ns, nm, 10) for _ in refs]
    t = 17
    load_impl_state(hk, states, t)
    for ref, s in zip(refs, states):
        load_state(ref, s, t)
        ref.learn()
    hk.learn()
    assert_same_params(hk, refs)


def test_step_does_not_modify_callers_tensor(torch_impl):
    hk, _ = make(torch_impl, 3, 3, num_envs=2)
    x = torch.full((2, 3), 0.3)
    x_before = x.clone()
    hk.step(x)
    assert torch.equal(x, x_before), "step() added sensor noise in place to the caller's observation"


@pytest.mark.parametrize("ns,nm", [(3, 3), (4, 2)])
def test_trajectory_matches_sox(torch_impl, no_noise, ns, nm):
    """Same sensor stream into both; compare motor outputs and parameters."""
    num_envs = 2
    hk, refs = make(torch_impl, ns, nm, num_envs=num_envs)
    rng = np.random.default_rng(21)
    for k in range(60):
        xs = [np.sin(0.3 * k + np.arange(ns) + e) * 0.8 + 0.05 * rng.standard_normal(ns) for e in range(num_envs)]
        y = t2n(hk.step(torch.tensor(np.stack(xs))))
        y_ref = np.stack([r.step(x) for r, x in zip(refs, xs)])
        np.testing.assert_allclose(y, y_ref, atol=1e-7, err_msg=f"step {k}")
    assert_same_params(hk, refs, atol=1e-7)
