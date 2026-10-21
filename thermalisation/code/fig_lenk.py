"""Isolated Bose gas of Lenk et al. (M = 5 levels, Delta = 10, U = 1, U' = 0.1, J = 1).

Figure lenk_mz:     Mori-Zwanzig analysis of n_sys = n_3 + n_4 + n_5 in equilibrium
                    (Kubo relaxation function with the energy projected out, memory
                    kernel, finite-size plateau) for several N, and the relaxation
                    spectrum of a set of candidate slow variables (N = 10, 15).
Figure lenk_quench: quench from the Fock state n_1 = N: occupations, entanglement and
                    diagonal entropies of the levels i >= 3, and the conserved energy
                    distribution of the whole system compared with the Gibbs weights.

usage: python code/fig_lenk.py [N_list, default 10,15,20]; the largest N is used for
the quench. Derived data are cached in figs/lenk_N*.npz (delete to recompute).
"""
import os, sys, time
import numpy as np
from lenk import (Model, Partition, entropies, canonical_beta, kubo_relaxation,
                  volterra_kernel, kubo_matrix, relaxation_spectrum)
from common import plt, C_HARM, C_FPUT, C_REF, C_ALT, panel_label

Ns = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [10, 15, 20]
SYS = [2, 3, 4]                                   # levels i = 3, 4, 5 (0-based)
TMAX = 12.0


def analyse(N, quench):
    fn = f"figs/lenk_N{N}.npz"
    if os.path.exists(fn):
        d = dict(np.load(fn))
        if not quench or "tq" in d:
            return d
    t0 = time.time()
    mod = Model(N)
    E, V = mod.eigh()
    s = np.diff(E); r = np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])
    c = V.T @ mod.state([N, 0, 0, 0, 0])
    Eq = (c**2) @ E                                # energy of the initial Fock state
    beta = canonical_beta(E, Eq)
    nsys = mod.basis[:, SYS].sum(1).astype(float)
    A = V.T @ (nsys[:, None] * V)
    t, C, dC, ddC, info = kubo_relaxation(E, A, beta, TMAX)
    K = volterra_kernel(t, C, dC, ddC)
    d = dict(N=N, D=mod.D, r=r[mod.D // 5:4 * mod.D // 5].mean(), beta=beta, Eq=Eq,
             sigE=np.sqrt((c**2) @ E**2 - Eq**2), t=t, C=C / C[0], K=K,
             mazur=info["mazur_H"], plateau=info["plateau"])
    print(f"N={N} D={mod.D} <r>={d['r']:.3f} T={1/beta:.1f} E={Eq:.0f} sigma_E={d['sigE']:.1f} "
          f"Mazur(H)={d['mazur']:.3f} plateau={d['plateau']:.3f} K(0)={K[0]:.1f} ({time.time()-t0:.0f}s)")
    if quench:
        part = Partition(mod, SYS)
        p = np.exp(-beta * (E - E[0])); p /= p.sum()
        nE = np.einsum("ka,ki->ai", V**2, mod.basis.astype(float))   # <a|n_i|a>
        d["n_gibbs"], d["n_diag"] = p @ nE, (c**2) @ nE
        # thermal reference for a pure state: canonical-typical states
        # |psi> = sum_a sqrt(p_a) exp(i phi_a) |a> with random phases phi_a
        rng = np.random.default_rng(1)
        typ = np.array([entropies(part.rho_blocks(V @ (np.sqrt(p) * np.exp(2j * np.pi * rng.random(len(E))))))
                        for _ in range(8)])
        d["S_typ"], d["S_typ_err"] = typ.mean(0), typ.std(0)
        tq = np.r_[0.0, np.logspace(-3, 2, 250)]
        occ, Svn, Sdg = [], [], []
        for tt in tq:
            psi = V @ (c * np.exp(-1j * E * tt))
            occ.append((np.abs(psi) ** 2) @ mod.basis)
            a, b = entropies(part.rho_blocks(psi)); Svn.append(a); Sdg.append(b)
        d.update(tq=tq, occ=np.array(occ), Svn=np.array(Svn), Sdg=np.array(Sdg), E=E, pE=c**2)
        print(f"   quench: n_diag={np.round(d['n_diag'], 2)} n_gibbs={np.round(d['n_gibbs'], 2)}")
        np.savez(fn, **d)
        late = tq > 20
        Svn, Sdg = np.array(Svn), np.array(Sdg)
        print(f"   late S_S={np.mean(Svn[late]):.3f}+-{np.std(Svn[late]):.3f}  S_diag={np.mean(Sdg[late]):.3f}  "
              f"typical: S_S={d['S_typ'][0]:.3f}+-{d['S_typ_err'][0]:.3f}  S_diag={d['S_typ'][1]:.3f}  "
              f"({time.time()-t0:.0f}s)")
    np.savez(fn, **d)
    return d


def spectrum(N):
    """Relaxation spectrum of candidate slow variables: occupations of S, intra-level
    interaction energies of S and R, uniform-mode occupation n_0 = K/M and its
    interaction energy n_0(n_0 - 1), and the level energy L = sum_i (i-1) n_i.
    Their span contains H, so one eigenvalue is exactly 1."""
    fn = f"figs/lenk_spec_N{N}.npz"
    if os.path.exists(fn):
        return dict(np.load(fn))
    mod = Model(N)
    E, V = mod.eigh()
    beta = canonical_beta(E, (V.T @ mod.state([N, 0, 0, 0, 0])) ** 2 @ E)
    n = mod.basis.astype(float)
    n0 = mod.K / mod.M
    diag = lambda v: V.T @ (v[:, None] * V)
    ops = [diag(n[:, i]) for i in SYS]
    ops += [diag((n[:, 2:] * (n[:, 2:] - 1)).sum(1)), diag((n[:, :2] * (n[:, :2] - 1)).sum(1)),
            V.T @ (n0 @ V), V.T @ ((n0 @ n0 - n0) @ V), diag((n * np.arange(mod.M)).sum(1))]
    tau = np.r_[0.0, np.logspace(-2, 1, 60)]
    tau, Ct = kubo_matrix(E, ops, beta, tau)
    lam = relaxation_spectrum(Ct[1:], Ct[0])
    d = dict(tau=tau[1:], lam=lam)
    np.savez(fn, **d)
    print(f"N={N}: relaxation spectrum at tau=1:", np.round(lam[np.argmin(abs(tau[1:] - 1))], 3))
    return d


res = {N: analyse(N, quench=(N == max(Ns))) for N in Ns}
spec = {N: spectrum(N) for N in (10, 15)}
Nq = max(Ns); q = res[Nq]
cols = {10: C_ALT, 15: C_HARM, 20: C_FPUT, 25: "k"}

# ---------------------------------------------------------------- Mori-Zwanzig figure
fig, axs = plt.subplots(2, 2, figsize=(8.6, 6.0))
axs = axs.ravel()
ax = axs[0]
for N in Ns:
    ax.plot(res[N]["t"], res[N]["C"], color=cols[N], lw=1.2, label=rf"$N={N}$")
ax.axhline(0, color=C_REF, lw=0.6)
ax.set_xlim(0, TMAX); ax.set_xlabel(r"$Jt$"); ax.set_ylabel(r"$C_\perp(t)/C_\perp(0)$")
ax.legend(loc="upper right"); panel_label(ax, "(a)")

ax = axs[1]
d = res[Nq]; dt = d["t"][1] - d["t"][0]
ax.plot(d["t"], d["K"] / d["K"][0], color=cols[Nq], lw=0.9, label=r"$K(t)/K(0)$")
G = np.cumsum(d["K"]) * dt
ax2 = ax.twinx()
ax2.plot(d["t"], G, color="k", lw=1.2, label=r"$\int_0^t K\,\mathrm{d}s$")
ax2.set_ylabel(r"$\int_0^t K(s)\,\mathrm{d}s \;[J]$")
ax2.spines["right"].set_visible(True)
ax.set_xlim(0, 3); ax.set_xlabel(r"$Jt$"); ax.set_ylabel(r"$K(t)/K(0)$")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper right")
panel_label(ax, "(b)")

ax = axs[2]
D = np.array([res[N]["D"] for N in Ns]); pl = np.array([res[N]["plateau"] for N in Ns])
ax.loglog(D, pl, "o", color=C_HARM, label="plateau")
ax.loglog(D, pl[0] * np.sqrt(D[0] / D), color=C_REF, ls="--", lw=0.9, label=r"$\propto D^{-1/2}$")
ax.set_xlabel(r"Hilbert-space dimension $D$"); ax.set_ylabel(r"$\overline{C_\perp}/C_\perp(0)$")
ax.legend(loc="upper right"); panel_label(ax, "(c)")

ax = axs[3]
for N, ls in ((10, ":"), (15, "-")):
    lam = spec[N]["lam"]
    for k in range(lam.shape[1]):
        ax.semilogx(spec[N]["tau"], lam[:, k], color=cols[N], ls=ls, lw=1.0,
                    label=rf"$N={N}$" if k == 0 else None)
ax.set_xlabel(r"lag $J\tau$"); ax.set_ylabel(r"eigenvalues of $\chi^{-1/2}C(\tau)\chi^{-1/2}$")
ax.set_ylim(-0.4, 1.05); ax.legend(loc="lower left")
panel_label(ax, "(d)")
fig.tight_layout(); fig.savefig("figs/lenk_mz.pdf"); print("wrote figs/lenk_mz.pdf")

# ---------------------------------------------------------------- quench figure
fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.0))
ax = axs[0]
tq = q["tq"][1:]
lc = [C_FPUT, C_HARM, C_ALT, "#CC79A7", "#999999"]
for i in range(5):
    ax.semilogx(tq, q["occ"][1:, i], color=lc[i], lw=1.0, label=rf"$n_{i+1}$")
    ax.axhline(q["n_gibbs"][i], color=lc[i], ls="--", lw=0.8)
ax.set_xlabel(r"$Jt$"); ax.set_ylabel(r"$\langle n_i\rangle(t)$"); ax.set_xlim(1e-3, 100)
ax.legend(loc="upper right", ncol=2); panel_label(ax, "(a)")

ax = axs[1]
ax.semilogx(tq, q["Svn"][1:], color=C_HARM, label=r"$S_{\mathrm{S}}$ (von Neumann)")
ax.semilogx(tq, q["Sdg"][1:], color=C_FPUT, label=r"$S_{\mathrm{diag}}$ (Fock diagonal)")
ax.axhline(q["S_typ"][0], color=C_HARM, ls="--", lw=0.8)
ax.axhline(q["S_typ"][1], color=C_FPUT, ls="--", lw=0.8)
ax.set_xlabel(r"$Jt$"); ax.set_ylabel("entropy of levels 3-5"); ax.set_xlim(1e-3, 100)
ax.legend(loc="lower right"); panel_label(ax, "(b)")

ax = axs[2]
E, pE = q["E"], q["pE"]
bins = np.linspace(E[0], E[-1], 80); bc = 0.5 * (bins[1:] + bins[:-1])
hq, _ = np.histogram(E, bins=bins, weights=pE)
pg = np.exp(-q["beta"] * (E - E[0])); pg /= pg.sum()
hg, _ = np.histogram(E, bins=bins, weights=pg)
ax.plot(bc, hq, color=C_HARM, drawstyle="steps-mid", label=r"initial state, $|c_E|^2$")
ax.plot(bc, hg, color=C_REF, ls="--", drawstyle="steps-mid", label="Gibbs, same $\\langle H\\rangle$")
ax.set_xlabel(r"$E/J$"); ax.set_ylabel("weight per bin")
ax.legend(loc="upper right"); panel_label(ax, "(c)")
fig.tight_layout(); fig.savefig("figs/lenk_quench.pdf"); print("wrote figs/lenk_quench.pdf")
