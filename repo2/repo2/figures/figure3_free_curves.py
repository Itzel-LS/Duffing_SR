"""
regenerate_figure3.py (v2 -- mismo formato que figure3_free_curves.py original)
=================================================================================
Regenera Figure_3_free_curves.png respetando el formato visual del script
original de Itzel (fuente serif, grid punteado, sombreado bajo la curva de
error, colores C0/C1/C3), pero con:
  - la ecuacion SR cuartica corregida (A1, ronda 2)
  - la referencia cuartica recalculada con la convencion de amplitud de
    punto de suelta (sum_k c_k = A), en vez de la vieja c1=A
  - la etiqueta del panel quintico actualizada de "HAM-P" a
    "First-harm. balance" (correccion B2)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 10,
})

beta_fixed = 0.15
alpha_range = np.linspace(0.5, 2.0, 100)
eps = 1e-20

# ======================================================================
# Balance armonico cuartico, convencion corregida (punto de suelta)
# ======================================================================
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

def solve_quartic_hb_corrected(alpha, beta, A=1.0, n_harmonics=M, u0=None):
    def eqs(u):
        omega, c = u[0], u[1:]
        r = _hb_residual_projections(c, omega, alpha, beta, n_harmonics)
        return np.concatenate((r, [c.sum() - A]))
    if u0 is None:
        c0 = np.zeros(n_harmonics + 1)
        c0[1] = A
        u0 = np.concatenate(([np.sqrt(alpha)], c0))
    sol = fsolve(eqs, u0, maxfev=2000)
    return sol[0]  # omega

# ======================================================================
# Expresiones SR finales (Tabla II, ronda 2)
# ======================================================================
def sr_cubic(alpha, beta):
    return np.sqrt(alpha + 0.7500 * beta)

def sr_quartic(alpha, beta):
    # corregida en A1 (ronda 2)
    return np.sqrt(alpha - beta * (
        (beta / (alpha - 0.4739 * (beta * alpha) / (alpha - 1.9888085 * beta)))
        - 0.015744861
    ))

def sr_quintic_first_harm(alpha, beta):
    return np.sqrt(alpha + 0.6250 * beta)

# ======================================================================
# Referencias
# ======================================================================
ref_cubic = np.sqrt(alpha_range + 0.75 * beta_fixed)
ref_quintic = np.sqrt(alpha_range + 0.625 * beta_fixed)

ref_quartic = []
prev = None  # no warm-start entre puntos (ver bug de ronda 2, punto A1)
for a in alpha_range:
    omega = solve_quartic_hb_corrected(a, beta_fixed, A=1.0)
    ref_quartic.append(omega)
ref_quartic = np.array(ref_quartic)

pred_cubic = sr_cubic(alpha_range, beta_fixed)
pred_quartic = sr_quartic(alpha_range, beta_fixed)
pred_quintic = sr_quintic_first_harm(alpha_range, beta_fixed)

# ======================================================================
# Figura: 2 filas x 3 columnas
# ======================================================================
fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))

panels = [
    (ref_cubic, pred_cubic, r"(a) $x^3$", "LP reference"),
    (ref_quartic, pred_quartic, r"(b) $x^4$", "Multi-harm. balance"),
    (ref_quintic, pred_quintic, r"(c) $x^5$", "First-harm. balance"),
]

for i, (ref, pred, title, method) in enumerate(panels):
    ax_top = axes[0, i]
    ax_top.plot(alpha_range, ref, linewidth=2.2, label="Reference")
    ax_top.plot(alpha_range, pred, linewidth=2.0, linestyle="--", label="SR")
    ax_top.set_xlabel(r"Parameter $\alpha$")
    ax_top.set_title(f"{title}\n{method}")
    ax_top.grid(True, linestyle=":", alpha=0.3)
    ax_top.legend(frameon=False)
    if i == 0:
        ax_top.set_ylabel(r"Frequency $\omega$")

    ax_bot = axes[1, i]
    error = np.abs(ref - pred) + eps
    ax_bot.plot(alpha_range, error, linewidth=1.8, color="C3")
    ax_bot.fill_between(alpha_range, error, alpha=0.15, color="C3")
    ax_bot.set_yscale("log")
    ax_bot.set_xlabel(r"Parameter $\alpha$")
    ax_bot.grid(True, linestyle=":", alpha=0.3)
    ax_bot.text(0.5, 0.85, f"Max = {error.max():.2e}",
                transform=ax_bot.transAxes, ha="center", fontsize=9)
    if i == 0:
        ax_bot.set_ylabel("Absolute error (log scale)")

plt.tight_layout()
plt.savefig("Figure_3_free_curves.png", dpi=300, bbox_inches="tight")
print("Guardada: Figure_3_free_curves.png")
for (ref, pred, title, method) in panels:
    print(f"{title}: max error = {np.max(np.abs(ref-pred)):.3e}")
