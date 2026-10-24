"""Reference port of the SOX controller (lpzrobots, Der & Martius).

This is a deliberately literal, unoptimised translation of
lpzrobots/selforg/controller/sox.cpp (Sox::step, Sox::stepNoLearning,
Sox::learn, Sox::pseudoInvL with pseudo == 0) and of the lpzrobots
matrix helpers it relies on. It is used as the test oracle for the
implementations in this repository.

Notation follows sox.cpp. Vectors are 1-D numpy arrays.
lpzrobots' `M & v` (row-wise multiplication) is written as `v[:, None] * M`,
which equals diag(v) @ M.

The only things exposed as options (with the SOX defaults) are the
hyperparameters, and the clip limits so that ports with different clip
limits can still be checked for the rest of the maths.
"""
from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class SoxConfig:
    # SoxConf (Sox::getDefaultConf)
    init_feedback_strength: float = 1.0
    use_extended_model: bool = True
    steps4averaging: int = 1
    steps4delay: int = 1
    factor_S: float = 1.0
    factor_b: float = 1.0
    factor_h: float = 1.0
    # parameters declared in Sox::constructor
    loga: bool = False
    epsC: float = 0.1
    epsA: float = 0.1
    sense: float = 1.0
    creativity: float = 0.0
    damping: float = 0.00001
    causeaware: float = None  # None -> 0.01 if use_extended_model else 0
    harmony: float = 0.0
    # Sox::buffersize
    buffersize: int = 10
    # clip limits used in Sox::learn (mapP(limit, clip))
    clip_A: float = 0.1
    clip_A_damp: float = 0.1
    clip_S: float = 0.1
    clip_b: float = 0.1
    clip_C: float = 0.05
    clip_C_damp: float = 0.05
    clip_h: float = 0.05
    # pseudo-inverse used for A+ and L+: "lpz" = lpzrobots Matrix::pseudoInverse
    # (normal equations, regularised only if singular), "svd" = Moore-Penrose via SVD.
    pinv: str = "lpz"

    def __post_init__(self):
        if self.causeaware is None:
            self.causeaware = 0.01 if self.use_extended_model else 0.0

    def but(self, **kw):
        return replace(self, **kw)


def lpz_pseudo_inverse(M, lam=1e-8):
    """lpzrobots matrix::Matrix::pseudoInverse.

    Wide: A^T (A A^T)^-1, tall: (A^T A)^-1 A^T. lambda is only added to the
    diagonal when the plain inverse has non-finite entries.
    """
    m, n = M.shape
    R = M.T @ M if m > n else M @ M.T
    with np.errstate(all="ignore"):
        try:
            Rinv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            Rinv = np.full_like(R, np.nan)
    if not np.all(np.isfinite(Rinv)):
        Rinv = np.linalg.inv(R + lam * np.eye(R.shape[0]))
    return Rinv @ M.T if m > n else M.T @ Rinv


def g(z):
    return np.tanh(z)


def g_s(z):
    """Derivative of tanh (Sox::g_s)."""
    k = np.tanh(z)
    return 1.0 - k * k


class SoxReference:
    def __init__(self, number_sensors, number_motors, config=None):
        self.conf = config if config is not None else SoxConfig()
        ns, nm = number_sensors, number_motors
        self.number_sensors, self.number_motors = ns, nm
        self.t = 0
        self.A = np.eye(ns, nm)
        self.C = np.eye(nm, ns) * self.conf.init_feedback_strength
        self.S = np.eye(ns, ns) * 0.05
        self.b = np.zeros(ns)
        self.h = np.zeros(nm)
        self.L = np.zeros((ns, ns))
        self.R = np.zeros((ns, ns))
        self.v_avg = np.zeros(ns)
        self.A_native = np.eye(ns, nm)
        self.C_native = np.eye(nm, ns) * 1.2
        self.x = np.zeros(ns)
        self.x_smooth = np.zeros(ns)
        bs = self.conf.buffersize
        self.x_buffer = [np.zeros(ns) for _ in range(bs)]
        self.y_buffer = [np.zeros(nm) for _ in range(bs)]

    # -- helpers -----------------------------------------------------------
    def _pinv(self, M):
        if self.conf.pinv == "svd":
            return np.linalg.pinv(M)
        return lpz_pseudo_inverse(M)

    @staticmethod
    def _clip(M, limit):
        return np.clip(M, -limit, limit)

    # -- Sox::step ---------------------------------------------------------
    def step(self, x_):
        y = self.step_no_learning(x_)
        if self.t <= self.conf.buffersize:
            return y
        self.t -= 1
        if self.conf.epsC != 0 or self.conf.epsA != 0:
            self.learn()
        self.t += 1
        return y

    # -- Sox::stepNoLearning -----------------------------------------------
    def step_no_learning(self, x_):
        c = self.conf
        bs = c.buffersize
        self.x = np.array(x_, dtype=float)
        c.steps4averaging = int(np.clip(c.steps4averaging, 1, bs - 1))
        if c.steps4averaging > 1:
            self.x_smooth = self.x_smooth + (self.x - self.x_smooth) * (1.0 / c.steps4averaging)
        else:
            self.x_smooth = self.x.copy()
        self.x_buffer[self.t % bs] = self.x_smooth.copy()
        y = g(self.C @ (self.x_smooth + self.v_avg * c.creativity) + self.h)
        self.y_buffer[self.t % bs] = y.copy()
        self.t += 1
        return y

    # -- Sox::learn --------------------------------------------------------
    def learn(self):
        c = self.conf
        bs = c.buffersize
        A, C, S, b, h = self.A, self.C, self.S, self.b, self.h

        s4delay = int(np.clip(c.steps4delay, 1, bs - 1))
        x = self.x_buffer[(self.t - max(s4delay, 1) + bs) % bs]
        y_creat = self.y_buffer[(self.t - max(s4delay, 1) + bs) % bs]
        x_fut = self.x_buffer[self.t % bs]

        xi = x_fut - (A @ y_creat + b + S @ x)

        z = C @ x + h
        y = g(z)
        g_prime = g_s(z)

        self.L = L = A @ (g_prime[:, None] * C) + S
        self.R = A @ C + S

        eta = self._pinv(A) @ xi
        y_hat = y + eta * c.causeaware

        Lplus = self._pinv(L)
        v = Lplus @ xi
        chi = Lplus.T @ v

        mu = (g_prime[:, None] * A.T) @ chi
        epsrel = (mu * (C @ v)) * (c.sense * 2)

        v_hat = v + x * c.harmony

        self.v_avg = self.v_avg + (v - self.v_avg) * 0.1

        EE = 1.0
        if c.loga:
            EE = 0.1 / (v @ v + 0.001)

        if c.epsA > 0:
            epsS = c.epsA * c.factor_S
            epsb = c.epsA * c.factor_b
            A = A + self._clip(np.outer(xi, y_hat) * c.epsA, c.clip_A)
            if c.damping:
                A = A + self._clip(((self.A_native - A) ** 3) * c.damping, c.clip_A_damp)
            if c.use_extended_model:
                S = S + self._clip(np.outer(xi, x) * epsS + S * (-c.damping * 10), c.clip_S)
            b = b + self._clip(xi * epsb + b * (-c.damping), c.clip_b)
        if c.epsC > 0:
            C = C + self._clip((np.outer(mu, v_hat) - np.outer(epsrel * y, x)) * (EE * c.epsC), c.clip_C)
            if c.damping:
                C = C + self._clip(((self.C_native - C) ** 3) * c.damping, c.clip_C_damp)
            h = h + self._clip((mu * c.harmony - epsrel * y) * (EE * c.epsC * c.factor_h), c.clip_h)

        self.A, self.C, self.S, self.b, self.h = A, C, S, b, h
        # expose intermediates for diagnostics
        self.last = dict(x=x, y_creat=y_creat, x_fut=x_fut, xi=xi, z=z, y=y,
                         g_prime=g_prime, eta=eta, y_hat=y_hat, Lplus=Lplus,
                         v=v, chi=chi, mu=mu, epsrel=epsrel, EE=EE)


def random_state(rng, ns, nm, bs=10, scale=0.5):
    """A random, generic controller state (for single-learn-step comparisons)."""
    return dict(
        A=np.eye(ns, nm) + scale * rng.standard_normal((ns, nm)),
        C=np.eye(nm, ns) + scale * rng.standard_normal((nm, ns)),
        S=0.05 * np.eye(ns) + 0.1 * scale * rng.standard_normal((ns, ns)),
        b=0.1 * rng.standard_normal(ns),
        h=0.1 * rng.standard_normal(nm),
        v_avg=0.1 * rng.standard_normal(ns),
        x_buffer=[rng.uniform(-1, 1, ns) for _ in range(bs)],
        y_buffer=[rng.uniform(-1, 1, nm) for _ in range(bs)],
    )


def load_state(ref, state, t):
    ref.A, ref.C, ref.S = state["A"].copy(), state["C"].copy(), state["S"].copy()
    ref.b, ref.h, ref.v_avg = state["b"].copy(), state["h"].copy(), state["v_avg"].copy()
    ref.x_buffer = [v.copy() for v in state["x_buffer"]]
    ref.y_buffer = [v.copy() for v in state["y_buffer"]]
    ref.t = t
