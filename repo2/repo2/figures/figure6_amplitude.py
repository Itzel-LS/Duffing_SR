"""
regenerate_figure4.py (v2 -- mismo formato que figure3_free_curves.py)
==========================================================================
Regenera Figure_6_amplitude.png (Fig. 4 del texto: omega(A) a alpha=1,
beta=0.15 fijos) con el mismo formato visual que Figure_3_free_curves.py:
fuente serif, grid punteado, sombreado bajo la curva de error, colores
C0/C1/C3.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, brentq

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 10,
})

ALPHA, BETA = 1.0, 0.15
eps = 1e-20
M = 5


def _hb_residual_projections(c, omega, alpha, beta, M, ngrid=2048):
    th = np.arange(ngrid) * 2 * np.pi / ngrid
    k = np.arange(M + 1)
    C = np.cos(np.outer(k, th))
    x = c @ C
    xdd = -(omega ** 2) * ((k ** 2 * c) @ C)
    R = xdd + alpha * x + beta * x ** 4
    proj = np.empty(M + 1)
    proj[0] = R.mean()
    for kk in range(1, M + 1):
        proj[kk] = 2.0 * (R * np.cos(kk * th)).mean()
    return proj


def solve_quartic_hb_corrected(alpha, beta, A, n_harmonics=M, u0=None):
    def eqs(u):
        omega, c = u[0], u[1:]
        r = _hb_residual_projections(c, omega, alpha, beta, n_harmonics)
        return np.concatenate((r, [c.sum() - A]))
    if u0 is None:
        c0 = np.zeros(n_harmonics + 1)
        c0[1] = A
        u0 = np.concatenate(([np.sqrt(alpha)], c0))
    sol = fsolve(eqs, u0, maxfev=2000)
    return sol[0]


def escape_amplitude(alpha, beta):
    V = lambda x: 0.5 * alpha * x ** 2 + beta * x ** 5 / 5.0
    xc = -(alpha / beta) ** (1.0 / 3.0)
    Vc = V(xc)
    return brentq(lambda a: V(a) - Vc, 1e-6, 50.0)


def sr_cubic_A(alpha, beta, A):
    return np.sqrt(alpha + 0.75 * A ** 2 * beta)


def sr_quartic_A(alpha, beta, A):
    return np.sqrt(alpha - ((A * ((((A * A) * beta) *
           (A * ((A * (3.92628 * (A * np.sqrt(beta)))) - alpha))) *
           (beta * 1.5603521))) * (A / alpha)))


def sr_quintic_A(alpha, beta, A):
    return np.sqrt(alpha + 0.625 * A ** 4 * beta)


A_esc = escape_amplitude(ALPHA, BETA)
A_range = np.linspace(0.25, min(2.0, A_esc * 0.985), 100)

ref_cubic = sr_cubic_A(ALPHA, BETA, A_range)
pred_cubic = ref_cubic.copy()

ref_quartic = np.array([solve_quartic_hb_corrected(ALPHA, BETA, A) for A in A_range])
pred_quartic = sr_quartic_A(ALPHA, BETA, A_range)

ref_quintic = sr_quintic_A(ALPHA, BETA, A_range)
pred_quintic = ref_quintic.copy()

panels = [
    (ref_cubic, pred_cubic, r"(a) $x^3$", ""),
    (ref_quartic, pred_quartic, r"(b) $x^4$", ""),
    (ref_quintic, pred_quintic, r"(c) $x^5$", "First-harm. balance"),
]

fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
for i, (ref, pred, title, method) in enumerate(panels):
    ax_top = axes[0, i]
    ax_top.plot(A_range, ref, linewidth=2.2, label="Reference")
    ax_top.plot(A_range, pred, linewidth=2.0, linestyle="--", label="SR")
    ax_top.set_xlabel("Amplitude $A$")
    ax_top.set_title(f"{title}\n{method}" if method else title)
    ax_top.grid(True, linestyle=":", alpha=0.3)
    ax_top.legend(frameon=False)
    if i == 0:
        ax_top.set_ylabel(r"Frequency $\omega$")

    ax_bot = axes[1, i]
    error = np.abs(ref - pred) + eps
    ax_bot.plot(A_range, error, linewidth=1.8, color="C3")
    ax_bot.fill_between(A_range, error, alpha=0.15, color="C3")
    ax_bot.set_yscale("log")
    ax_bot.set_xlabel("Amplitude $A$")
    ax_bot.grid(True, linestyle=":", alpha=0.3)
    ax_bot.text(0.5, 0.85, f"Max = {error.max():.2e}",
                transform=ax_bot.transAxes, ha="center", fontsize=9)
    if i == 0:
        ax_bot.set_ylabel("Absolute error (log scale)")

plt.tight_layout()
plt.savefig("Figure_6_amplitude.png", dpi=300, bbox_inches="tight")
print("Guardada: Figure_6_amplitude.png")
for (ref, pred, title, method) in panels:
    print(f"{title}: max error = {np.max(np.abs(ref-pred)):.3e}")
