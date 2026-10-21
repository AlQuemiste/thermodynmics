"""Closed chain, energy initially in the four longest-wavelength modes.
Harmonic (beta = 0) versus beta-FPUT (beta = 1), same initial conditions.

Diagnostics: effective mode number n_eff = exp(S_spec)/N, nearest-neighbour
velocity correlation c_1, and the late-time mode-energy spectrum.

Writes figs/chain.pdf (harmonic chain only, Sec. 3) and figs/fput.pdf
(harmonic versus beta-FPUT, appendix). The simulation is cached in
figs/fput_data.npz (delete it to rerun, ~3 min).
"""
import os
import numpy as np
from chain import yoshida_fixed, energy_fixed, modes_fixed, mode_energies
from common import plt, C_HARM, C_FPUT, C_REF, panel_label

N, R, eps, dt = 32, 16, 0.1, 0.05
DATA = "figs/fput_data.npz"


def simulate():
    kex = np.arange(4)                            # excited modes k = 1..4
    S, w = modes_fixed(N)
    rng = np.random.default_rng(7)
    ph = rng.uniform(0.0, 2 * np.pi, (R, kex.size))
    Ek = eps * N / kex.size
    Q = np.zeros((R, N)); P = np.zeros((R, N))
    Q[:, kex] = np.sqrt(2 * Ek) / w[kex] * np.cos(ph)
    P[:, kex] = -np.sqrt(2 * Ek) * np.sin(ph)
    x = np.vstack([Q @ S, Q @ S])                 # rows 0..R-1 harmonic, R..2R-1 FPUT
    p = np.vstack([P @ S, P @ S])
    beta = np.vstack([np.zeros((R, 1)), np.ones((R, 1))])
    E0 = energy_fixed(x, p, beta)

    t_out = np.unique(np.round(np.logspace(0, np.log10(2e4), 90) / dt).astype(int)) * dt
    neff, c1, tt = [], [], []
    Ek_late = np.zeros((2 * R, N)); n_late = 0
    step_done = 0
    for tk in t_out:
        n = int(round(tk / dt)) - step_done
        x, p = yoshida_fixed(x, p, beta, dt, n)
        step_done += n
        E = mode_energies(x, p, S, w)
        wq = E / E.sum(1, keepdims=True)
        neff.append(np.exp(-(wq * np.log(wq + 1e-300)).sum(1)) / N)
        c1.append((p[:, :-1] * p[:, 1:]).sum(1) / (p**2).sum(1))
        tt.append(tk)
        if tk > 1e4:
            Ek_late += E; n_late += 1
    drift = np.abs(energy_fixed(x, p, beta) / E0 - 1).max()
    np.savez(DATA, t=np.array(tt), neff=np.array(neff), c1=np.array(c1),
             Ek_late=Ek_late / n_late, drift=drift)


if not os.path.exists(DATA):
    simulate()
d = np.load(DATA)
tt, neff, c1, Ek_late = d["t"], d["neff"], d["c1"], d["Ek_late"]
print("max relative energy drift:", float(d["drift"]))
print("late n_eff harm/FPUT:", neff[tt > 1e4, :R].mean(), neff[tt > 1e4, R:].mean())
print("late c1 harm/FPUT:", c1[tt > 1e4, :R].mean(), c1[tt > 1e4, R:].mean())

neq = np.exp(np.euler_gamma - 1.0)            # equipartition value of n_eff
k = np.arange(1, N + 1)
series = {"harmonic": (slice(0, R), C_HARM, "o"), r"$\beta$-FPUT": (slice(R, 2 * R), C_FPUT, "s")}


def plot(names, fname, log_spectrum):
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 2.9))
    ax = axs[0]
    for nm in names:
        sl, col, _ = series[nm]
        ax.semilogx(tt, neff[:, sl].mean(1), color=col, label=nm)
    ax.axhline(neq, color=C_REF, ls="--", lw=1.0)
    ax.text(1.3, neq + 0.02, "equipartition", color=C_REF, fontsize=8.5)
    ax.set_xlabel(r"$\omega_0 t$"); ax.set_ylabel(r"$n_{\mathrm{eff}}$")
    ax.set_ylim(0, 0.8)
    if len(names) > 1:
        ax.legend(loc="center right")
    panel_label(ax, "(a)")

    ax = axs[1]
    for nm in names:
        sl, col, _ = series[nm]
        ax.semilogx(tt, c1[:, sl].mean(1), color=col)
    ax.axhline(0.0, color=C_REF, ls="--", lw=1.0)
    ax.text(1.3, 0.04, "Gibbs", color=C_REF, fontsize=8.5)
    ax.set_xlabel(r"$\omega_0 t$"); ax.set_ylabel(r"$c_1$")
    ax.set_ylim(-0.2, 1.05)
    panel_label(ax, "(b)")

    ax = axs[2]
    for nm in names:
        sl, col, mk = series[nm]
        y = Ek_late[sl].mean(0) / eps
        (ax.semilogy if log_spectrum else ax.plot)(k, y, mk, ms=3.5, color=col)
    ax.axhline(1.0, color=C_REF, ls="--", lw=1.0)
    ax.set_xlabel(r"mode index $k$"); ax.set_ylabel(r"$\overline{E_k}/\varepsilon$")
    if log_spectrum:
        ax.set_ylim(1e-3, 30)
    panel_label(ax, "(c)")
    fig.tight_layout()
    fig.savefig(fname)
    print("wrote", fname)


plot(["harmonic"], "figs/chain.pdf", log_spectrum=False)
plot(["harmonic", r"$\beta$-FPUT"], "figs/fput.pdf", log_spectrum=True)
