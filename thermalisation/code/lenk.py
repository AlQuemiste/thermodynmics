"""Bose gas of N particles in M single-particle levels (Lenk et al., Entropy 27, 636 (2025)).

H = Delta sum_i (i-1) n_i + J sum_{i!=j} b_i^+ b_j + U sum_i b_i^+ b_i^+ b_i b_i
    + U' sum_{ijlm, not i=j=l=m} b_i^+ b_j^+ b_l b_m .

With B = sum_i b_i and K = B^+ B, the all-to-all terms are
    J sum_{i!=j} b_i^+ b_j = J (K - N),
    U' sum_{not i=j=l=m} ... = U' (B^+ B^+ B B - sum_i n_i(n_i-1)) = U' (K^2 - M K - sum_i n_i(n_i-1)),
using [B, B^+] = M. Units: J = 1, hbar = k_B = 1.
"""
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from scipy.optimize import brentq
from itertools import combinations


def fock_basis(N, M):
    """All occupation tuples (n_1..n_M) with sum N, as an int array (D, M)."""
    # stars and bars: choose M-1 bar positions among N+M-1 slots
    out = []
    for bars in combinations(range(N + M - 1), M - 1):
        b = (-1,) + bars + (N + M - 1,)
        out.append([b[k + 1] - b[k] - 1 for k in range(M)])
    return np.array(out, dtype=np.int64)


class Model:
    def __init__(self, N, M=5, Delta=10.0, J=1.0, U=1.0, Up=0.1, eps=None):
        self.N, self.M = N, M
        self.basis = fock_basis(N, M)
        self.D = len(self.basis)
        self.index = {tuple(s): k for k, s in enumerate(self.basis)}
        n = self.basis
        # single-particle level energies; default: equidistant, eps_i = Delta (i-1)
        eps = Delta * np.arange(M) if eps is None else np.asarray(eps, float)
        # diagonal part in the Fock basis
        self.diag_single = (n * eps).sum(1)
        self.diag_int = (n * (n - 1)).sum(1)
        K = self._K()
        I = sp.identity(self.D, format="csr")
        self.K = K
        self.H0 = sp.diags(self.diag_single + (U - Up) * self.diag_int)
        self.Hhop = J * (K - N * I)
        self.Hpair = Up * (K @ K - M * K)
        self.H = (self.H0 + self.Hhop + self.Hpair).tocsr()

    def _K(self):
        """K = sum_{ij} b_i^+ b_j as a sparse matrix in the N-particle sector."""
        rows, cols, vals = [], [], []
        for k, s in enumerate(self.basis):
            for j in range(self.M):
                if s[j] == 0:
                    continue
                for i in range(self.M):
                    t = s.copy()
                    amp = np.sqrt(t[j]); t[j] -= 1
                    amp *= np.sqrt(t[i] + 1); t[i] += 1
                    rows.append(self.index[tuple(t)]); cols.append(k); vals.append(amp)
        return sp.csr_matrix((vals, (rows, cols)), shape=(self.D, self.D))

    def state(self, occ):
        v = np.zeros(self.D)
        v[self.index[tuple(occ)]] = 1.0
        return v

    def eigh(self):
        """Full spectrum and eigenvectors (dense LAPACK, divide and conquer)."""
        return la.eigh(self.H.toarray(), overwrite_a=True, driver="evd", check_finite=False)


class Partition:
    """Split of the levels into an observed subsystem S and a remainder R.

    The N-particle Hilbert space is a direct sum over n_S of products of
    n_S-particle states of S and (N - n_S)-particle states of R, so every reduced
    density matrix of S is block diagonal in n_S. Index maps are built once.
    """

    def __init__(self, model, sys_levels):
        sl = list(sys_levels)
        rl = [i for i in range(model.M) if i not in sl]
        ns, nr = model.basis[:, sl], model.basis[:, rl]
        nsys = ns.sum(1)
        self.blocks = []
        for m in np.unique(nsys):
            sel = np.where(nsys == m)[0]
            su, si = np.unique(ns[sel], axis=0, return_inverse=True)
            ru, ri = np.unique(nr[sel], axis=0, return_inverse=True)
            self.blocks.append((sel, si.ravel(), ri.ravel(), len(su), len(ru)))

    def rho_blocks(self, psi):
        """Blocks of rho_S = Tr_R |psi><psi|."""
        out = []
        for sel, si, ri, a, b in self.blocks:
            X = np.zeros((a, b), dtype=complex)
            X[si, ri] = psi[sel]
            out.append(X @ X.conj().T)
        return out

    def rho_blocks_mixture(self, V, p):
        """Blocks of Tr_R sum_a p_a |a><a| for eigenvectors V[:, a] with weights p_a."""
        out = []
        sq = np.sqrt(p)
        for sel, si, ri, a, b in self.blocks:
            rho = np.zeros((a, a))
            Vs = V[sel] * sq                      # rows of this block, weighted
            for r in range(b):
                rows = ri == r
                Y = np.zeros((a, V.shape[1]))
                Y[si[rows]] = Vs[rows]
                rho += Y @ Y.T
            out.append(rho)
        return out


def entropies(blocks):
    """von Neumann entropy of a block-diagonal state and Shannon entropy of its diagonal."""
    S_vn = S_diag = 0.0
    for rho in blocks:
        ev = np.linalg.eigvalsh(rho)
        ev = ev[ev > 1e-15]
        S_vn -= (ev * np.log(ev)).sum()
        d = np.real(np.diag(rho))
        d = d[d > 1e-15]
        S_diag -= (d * np.log(d)).sum()
    return S_vn, S_diag


def canonical_beta(E, Emean):
    """Inverse temperature of the Gibbs state with <H> = Emean."""
    f = lambda b: (np.exp(-b * (E - E[0])) @ E) / np.exp(-b * (E - E[0])).sum() - Emean
    return brentq(f, 1e-8, 10.0)


def kubo_relaxation(E, A, beta, tmax, dw=0.005, project_energy=True):
    """Kubo relaxation function C(t) = (dA, dA(t)), with its first and second time
    derivatives, for an observable A given as a matrix in the eigenbasis.

    (A, B) = int_0^beta dl <exp(lH) A^+ exp(-lH) B> / beta - <A^+><B>, so that
    C(t)/C(0) is the exact linear-response relaxation of <A> after a small static
    field conjugate to A is switched off. With project_energy, the component of A
    along the conserved H is removed (relevant set {A, H}). The spectral weights are
    binned on a frequency grid of spacing dw and Fourier transformed by FFT.
    """
    p = np.exp(-beta * (E - E[0])); p /= p.sum()
    w = E[None, :] - E[:, None]                      # omega_ab = E_b - E_a
    with np.errstate(divide="ignore", invalid="ignore"):
        kub = np.where(np.abs(w) > 1e-9, (p[:, None] - p[None, :]) / (beta * w), p[:, None])
    wmax = 1.05 * (E[-1] - E[0])
    nb = int(2 * wmax / dw); nb += nb % 2
    idx = (np.rint(w / dw).astype(np.int64) % nb).ravel()
    h = np.bincount(idx, weights=(kub * A * A.T).ravel(), minlength=nb)
    diagA = np.diag(A); mean = p @ diagA
    h[0] -= mean**2
    info = {"C0_full": h.sum()}
    if project_energy:
        dH = E - p @ E
        maz = (p @ ((diagA - mean) * dH)) ** 2 / (p @ dH**2)
        h[0] -= maz
        info["mazur_H"] = maz / info["C0_full"]
    info["plateau"] = (p @ diagA**2 - mean**2 - info.get("mazur_H", 0) * info["C0_full"])
    wg = np.r_[np.arange(nb // 2), np.arange(-nb // 2, 0)] * dw
    dt = 2 * np.pi / (nb * dw)
    nt = int(tmax / dt) + 1
    C = np.real(np.fft.fft(h))[:nt]
    dC = np.real(np.fft.fft(-1j * wg * h))[:nt]
    ddC = np.real(np.fft.fft(-wg**2 * h))[:nt]
    info["plateau"] /= C[0]
    return np.arange(nt) * dt, C, dC, ddC, info


def kubo_matrix(E, ops, beta, times, dw=0.005):
    """Kubo correlation matrix C_ij(t) = (dA_i, dA_j(t)) of Hermitian operators
    (matrices in the eigenbasis) at the requested times; returns (len(times), n, n)."""
    p = np.exp(-beta * (E - E[0])); p /= p.sum()
    w = E[None, :] - E[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        kub = np.where(np.abs(w) > 1e-9, (p[:, None] - p[None, :]) / (beta * w), p[:, None])
    wmax = 1.05 * (E[-1] - E[0])
    nb = int(2 * wmax / dw); nb += nb % 2
    idx = (np.rint(w / dw).astype(np.int64) % nb).ravel()
    dt = 2 * np.pi / (nb * dw)
    ti = np.rint(np.asarray(times) / dt).astype(int)
    means = [p @ np.diag(A) for A in ops]
    n = len(ops)
    out = np.zeros((len(ti), n, n))
    for a in range(n):
        for b in range(a, n):
            h = np.bincount(idx, weights=(kub * ops[a] * ops[b].T).ravel(), minlength=nb)
            h[0] -= means[a] * means[b]
            out[:, a, b] = out[:, b, a] = np.real(np.fft.fft(h))[ti]
    return ti * dt, out


def relaxation_spectrum(Ct, C0):
    """Eigenvalues of the normalized Mori propagator C0^-1/2 C(t) C0^-1/2, sorted
    in decreasing order; eigenvalues near 1 mark slow directions, near 0 fast ones."""
    ev, U = np.linalg.eigh(C0)
    W = U @ np.diag(ev**-0.5) @ U.T
    return np.array([np.sort(np.linalg.eigvalsh(W @ C @ W))[::-1] for C in Ct])


def volterra_kernel(t, C, dC, ddC):
    """Mori memory kernel from dC/dt = -int_0^t K(s) C(t-s) ds (single variable, Omega = 0).

    Differentiating once gives the Volterra equation of the second kind
    K(t) C(0) = -C''(t) - int_0^t K(s) C'(t-s) ds, solved by the trapezoidal rule.
    """
    dt = t[1] - t[0]
    K = np.empty(len(t))
    K[0] = -ddC[0] / C[0]
    for i in range(1, len(t)):
        s = 0.5 * K[0] * dC[i] + np.dot(K[1:i], dC[i - 1:0:-1])   # C'(0) = 0
        K[i] = (-ddC[i] - dt * s) / C[0]
    return K
