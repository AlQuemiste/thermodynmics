"""Quench observables of the Lenk et al. model at a single N, for comparison with
their published numbers (N = 25): late-time entanglement entropy of levels 3-5,
its canonical-typical pure-state reference and the Schmidt-rank bound, the
canonical temperature at the quench energy, and the long-time occupations.

usage: python code/lenk_check.py N   (N = 25: D = 23751, ~15 min, ~15 GB RAM)
"""
import sys, time
import numpy as np
from lenk import Model, Partition, entropies, canonical_beta

N = int(sys.argv[1])
t0 = time.time()
mod = Model(N)
E, V = mod.eigh()
print(f"N={N} D={mod.D}: diagonalized in {time.time() - t0:.0f} s")
s = np.diff(E); r = np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])
c = V.T @ mod.state([N, 0, 0, 0, 0])
pq = c**2
Eq = pq @ E
beta = canonical_beta(E, Eq)
p = np.exp(-beta * (E - E[0])); p /= p.sum()
part = Partition(mod, [2, 3, 4])
rank = sum(min(a, b) for *_, a, b in part.blocks)

# long-time occupations: diagonal ensemble vs canonical
nE = np.einsum("ka,ki->ai", V**2, mod.basis.astype(float))
n_diag, n_can = pq @ nE, p @ nE
del nE

late = np.linspace(20, 100, 41)
S_late = np.array([entropies(part.rho_blocks(V @ (c * np.exp(-1j * E * t))))[0] for t in late])
early = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
S_early = np.array([entropies(part.rho_blocks(V @ (c * np.exp(-1j * E * t))))[0] for t in early])
rng = np.random.default_rng(1)
S_typ = np.array([entropies(part.rho_blocks(V @ (np.sqrt(p) * np.exp(2j * np.pi * rng.random(len(E))))))[0]
                  for _ in range(8)])

res = dict(N=N, D=mod.D, r=r[mod.D // 5:4 * mod.D // 5].mean(), Eq=Eq, sigE=np.sqrt(pq @ E**2 - Eq**2),
           T=1 / beta, sigE_can=np.sqrt(p @ E**2 - (p @ E) ** 2), lnrank=np.log(rank),
           S_late=S_late.mean(), S_late_sd=S_late.std(), S_typ=S_typ.mean(), S_typ_sd=S_typ.std(),
           early=early, S_early=S_early, n_diag=n_diag, n_can=n_can)
np.savez(f"figs/lenk_check_N{N}.npz", **res)
for k, v in res.items():
    print(f"  {k}: {np.round(v, 4)}")
print(f"total {time.time() - t0:.0f} s")
