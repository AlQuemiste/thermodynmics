"""Symplectic integration of harmonic / beta-FPUT chains (m = kappa = a = 1).

V(r) = r^2/2 + beta r^4/4 for each bond elongation r. Arrays carry a leading
ensemble axis so that many trajectories are advanced at once.
"""
import numpy as np

# 4th-order Yoshida coefficients
_w1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_w0 = -(2.0 ** (1.0 / 3.0)) * _w1
YOSHIDA_C = (_w1 / 2, (_w0 + _w1) / 2, (_w0 + _w1) / 2, _w1 / 2)
YOSHIDA_D = (_w1, _w0, _w1)


def force_fixed(x, beta):
    """Forces for fixed ends x_0 = x_{N+1} = 0; x has shape (R, N), beta (R, 1)."""
    xp = np.pad(x, ((0, 0), (1, 1)))
    r = np.diff(xp, axis=1)                       # N+1 bond elongations
    f = r + beta * r**3                           # V'(r)
    return f[:, 1:] - f[:, :-1]


def yoshida_fixed(x, p, beta, dt, nsteps):
    """Advance fixed-end chain by nsteps with the 4th-order Yoshida scheme."""
    for _ in range(nsteps):
        for c, d in zip(YOSHIDA_C[:3], YOSHIDA_D):
            x += c * dt * p
            p += d * dt * force_fixed(x, beta)
        x += YOSHIDA_C[3] * dt * p
    return x, p


def energy_fixed(x, p, beta):
    xp = np.pad(x, ((0, 0), (1, 1)))
    r = np.diff(xp, axis=1)
    return 0.5 * (p**2).sum(1) + (0.5 * r**2 + 0.25 * beta * r**4).sum(1)


def modes_fixed(N):
    """Orthonormal sine modes S[k, j] and frequencies of the fixed-end chain."""
    k = np.arange(1, N + 1)
    j = np.arange(1, N + 1)
    S = np.sqrt(2.0 / (N + 1)) * np.sin(np.pi * np.outer(k, j) / (N + 1))
    w = 2.0 * np.sin(np.pi * k / (2 * (N + 1)))
    return S, w


def mode_energies(x, p, S, w):
    """Harmonic energies E_k = (P_k^2 + w_k^2 Q_k^2)/2 of each normal mode."""
    Q = x @ S.T
    P = p @ S.T
    return 0.5 * (P**2 + (w * Q) ** 2)
