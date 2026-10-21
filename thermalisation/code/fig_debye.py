"""Debye-Waller exponent W(T) and the entanglement entropy of a probe branching
into two momentum transfers, for a Debye phonon bath.

W(q, T) = 2 W_0 Phi(T/Theta_D),  Phi(x) = int_0^1 du u coth(u / 2x),
W_0 = 3 hbar^2 q^2 / (8 M k_B Theta_D) (zero-point value).
Probe visibility |s| = exp(-W); entropy S = -sum_{+-} l ln l, l = (1 +- |s|)/2.
"""
import numpy as np
from scipy.integrate import quad
from common import plt, C_HARM, C_FPUT, C_REF, C_ALT, panel_label


def phi(x):
    if x == 0:
        return 0.5
    return quad(lambda u: u / np.tanh(u / (2 * x)) if u > 0 else 2 * x, 0, 1)[0]


def s_ent(W):
    s = np.exp(-W)
    lp, lm = 0.5 * (1 + s), 0.5 * (1 - s)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -(lp * np.log(lp) + np.where(lm > 0, lm * np.log(lm), 0.0))


x = np.linspace(0, 2, 201)
Phi = np.array([phi(v) for v in x])
print("Phi(0) =", Phi[0], " high-T check Phi(2)/(2*2) =", Phi[-1] / 4)

fig, axs = plt.subplots(1, 2, figsize=(8.0, 2.9))
ax = axs[0]
ax.plot(x, 2 * Phi, color=C_HARM, label="exact")
xl = np.linspace(0, 0.35, 50)
ax.plot(xl, 1 + 2 * np.pi**2 / 3 * xl**2, color=C_REF, ls="--", lw=0.9, label=r"$1 + \frac{2\pi^2}{3}(T/\Theta_D)^2$")
ax.plot(x, 4 * x, color=C_REF, ls=":", lw=1.2, label=r"$4\,T/\Theta_D$")
ax.set_xlabel(r"$T/\Theta_D$"); ax.set_ylabel(r"$W(T)/W_0$")
ax.set_ylim(0, 8.5); ax.legend(loc="upper left")
panel_label(ax, "(a)")

ax = axs[1]
for W0, col in zip((0.02, 0.1, 0.5), (C_ALT, C_HARM, C_FPUT)):
    ax.plot(x, s_ent(2 * W0 * Phi), color=col, label=rf"$W_0 = {W0}$")
ax.axhline(np.log(2), color=C_REF, ls="--", lw=0.9)
ax.text(0.02, np.log(2) + 0.02, r"$\ln 2$", color=C_REF, fontsize=8.5)
ax.set_xlabel(r"$T/\Theta_D$"); ax.set_ylabel(r"$S_{\mathrm{ent}}$")
ax.set_ylim(0, 0.8); ax.legend(loc="lower right")
panel_label(ax, "(b)")

fig.tight_layout()
fig.savefig("figs/debye.pdf")
print("wrote figs/debye.pdf")
