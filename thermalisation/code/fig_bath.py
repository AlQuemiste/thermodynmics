"""Two coupled oscillators weakly coupled (through x_1) to a large harmonic bath.

H = sum_{i=1,2} p_i^2/2 + x_1^2/2 + x_2^2/2 + kappa (x_1 - x_2)^2/2
    + sum_k [ p_k^2/2 + omega_k^2/2 (q_k - c_k x_1/omega_k^2)^2 ]      (Caldeira-Leggett form)

Ohmic spectral density J(w) = eta w for w < w_c, discretized as
c_k^2 = (2/pi) eta w_k^2 dw. Units: m = omega_0 = k_B = 1.
The full quadratic problem is solved exactly through its normal modes, and the
energies of the two system normal modes are compared with the weak-coupling
prediction E_a(t) = T + (E_a(0) - T) exp(-gamma_a t), gamma_a = u_{1a}^2 eta.
"""
import numpy as np
from common import plt, C_HARM, C_FPUT, C_REF, C_ALT, panel_label

KAPPA, ETA, WC, T = 0.3, 0.04, 5.0, 1.0
KS = np.array([[1 + KAPPA, -KAPPA], [-KAPPA, 1 + KAPPA]])
OM2, US = np.linalg.eigh(KS)                     # system normal modes
OM = np.sqrt(OM2)                                # Omega_+ = 1, Omega_- = sqrt(1 + 2 kappa)
GAMMA = US[0] ** 2 * ETA                         # predicted energy damping rates


def build(nb):
    """Bath frequencies, couplings and the full mass-weighted stiffness matrix."""
    dw = WC / nb
    w = (np.arange(nb) + 0.5) * dw
    c = np.sqrt(2.0 / np.pi * ETA * w**2 * dw)
    n = 2 + nb
    Kf = np.zeros((n, n))
    Kf[:2, :2] = KS
    Kf[0, 0] += (c**2 / w**2).sum()              # counter-term keeps H positive
    Kf[0, 2:] = Kf[2:, 0] = -c
    Kf[2:, 2:] = np.diag(w**2)
    return w, c, Kf


def initial(w, c, R, E0, rng):
    """System: mode energies E0 with random phases; bath: Gibbs at temperature T,
    centred on the displaced equilibrium q_k = c_k x_1 / w_k^2."""
    ph = rng.uniform(0, 2 * np.pi, (2, R))
    Q = np.sqrt(2 * E0)[:, None] / OM[:, None] * np.cos(ph)
    P = -np.sqrt(2 * E0)[:, None] * np.sin(ph)
    xs, ps = US @ Q, US @ P
    q = c[:, None] * xs[0] / w[:, None] ** 2 + np.sqrt(T) / w[:, None] * rng.normal(size=(len(w), R))
    p = np.sqrt(T) * rng.normal(size=(len(w), R))
    return np.vstack([xs, q]), np.vstack([ps, p])


def evolve_system_modes(Kf, x0, p0, times):
    """Exact evolution; returns system normal-mode energies E_a(t) for each trajectory."""
    lam, V = np.linalg.eigh(Kf)
    om = np.sqrt(lam)
    X0, P0 = V.T @ x0, V.T @ p0
    Vs = V[:2]
    out = np.empty((len(times), 2, x0.shape[1]))
    for i, t in enumerate(times):
        co, si = np.cos(om * t)[:, None], np.sin(om * t)[:, None]
        xs = Vs @ (co * X0 + si / om[:, None] * P0)
        ps = Vs @ (-om[:, None] * si * X0 + co * P0)
        Qa, Pa = US.T @ xs, US.T @ ps
        out[i] = 0.5 * (Pa**2 + OM2[:, None] * Qa**2)
    return out


if __name__ == "__main__":
    import sys
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    rng = np.random.default_rng(3)
    E0 = np.array([4.0, 0.0])                        # energy only in the in-phase mode
    times = np.linspace(0, 300, 301)
    print("Omega =", OM, " gamma_a =", GAMMA, " secular ratio gamma/dOmega =", GAMMA.max() / np.diff(OM)[0])

    # (a, b) large bath: nb = 2000, dw = 2.5e-3 << gamma = 2e-2
    w, c, Kf = build(2000)
    x0, p0 = initial(w, c, R, E0, rng)
    Ea = evolve_system_modes(Kf, x0, p0, times)

    # (c) small bath: nb = 100, dw = 0.05 > gamma
    times_d = np.linspace(0, 600, 601)
    ws, cs, Ks = build(100)
    x0, p0 = initial(ws, cs, R, E0, rng)
    Ed = evolve_system_modes(Ks, x0, p0, times_d)

    theory = lambda a, t: T + (E0[a] - T) * np.exp(-GAMMA[a] * t)
    late = times > 200
    Elate = Ea[late].reshape(-1, 2, R).transpose(1, 0, 2).reshape(2, -1)
    print("late <E_a> thermal bath:", Elate.mean(1), " var/mean^2:", Elate.var(1) / Elate.mean(1) ** 2)
    print("max |<E_a> - theory|:",
          max(np.abs(Ea[:, a].mean(1) - theory(a, times)).max() for a in range(2)))
    np.savez("figs/bath_data.npz", t=times, Ea=Ea.mean(2), td=times_d, Ed=Ed.mean(2))

    cols = (C_FPUT, C_HARM)
    lab = (r"$\Omega_+ = %.2f$" % OM[0], r"$\Omega_- = %.2f$" % OM[1])
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 2.9))
    ax = axs[0]
    for a in range(2):
        ax.plot(times, Ea[:, a].mean(1), color=cols[a], label=lab[a])
        ax.plot(times, theory(a, times), color="k", ls="--", lw=0.9)
    ax.axhline(T, color=C_REF, lw=0.8, ls=":")
    ax.set_xlabel(r"$\omega_0 t$"); ax.set_ylabel(r"$\langle E_a\rangle / k_B T$")
    ax.legend(loc="upper right"); panel_label(ax, "(a)")

    ax = axs[1]
    bins = np.linspace(0, 6, 31)
    bc = 0.5 * (bins[1:] + bins[:-1])
    for a, mk in zip(range(2), "os"):
        hst, _ = np.histogram(Elate[a] / T, bins=bins, density=True)
        ax.plot(bc, hst, mk, ms=3.5, color=cols[a])
    xx = np.linspace(0, 6, 200)
    ax.semilogy(xx, np.exp(-xx), color="k", ls="--", lw=0.9, label=r"$e^{-E/k_BT}$")
    ax.set_xlabel(r"$E_a / k_B T$"); ax.set_ylabel("probability density")
    ax.set_ylim(2e-3, 1.5); ax.legend(loc="upper right"); panel_label(ax, "(b)")

    ax = axs[2]
    for a in range(2):
        ax.plot(times_d, Ed[:, a].mean(1), color=cols[a], lw=1.0)
    ax.axhline(T, color=C_REF, lw=0.8, ls=":")
    ax.set_xlabel(r"$\omega_0 t$"); ax.set_ylabel(r"$\langle E_a\rangle / k_B T$")
    ax.set_title(r"small bath, $N_B = 100$", fontsize=10); panel_label(ax, "(c)")

    fig.tight_layout()
    fig.savefig("figs/bath.pdf")
    print("wrote figs/bath.pdf")
