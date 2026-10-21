"""Shared plotting style and helpers for the thermalization figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# colour-blind-safe palette (Okabe-Ito subset)
C_HARM = "#0072B2"   # harmonic / integrable
C_FPUT = "#D55E00"   # anharmonic / non-integrable
C_REF = "#555555"    # analytic reference curves
C_ALT = "#009E73"    # secondary series

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.4,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def panel_label(ax, s):
    ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontweight="bold",
            va="bottom", ha="left")
