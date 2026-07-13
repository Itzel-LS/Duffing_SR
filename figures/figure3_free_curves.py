"""
figure3_free_curves.py
========================

Regenera la Figura 3: omega(alpha) a beta=0.15 fijo, para los tres casos
con referencia de forma cerrada o semi-analitica (cubico-LP, cuartico
balance multi-armonico, quintico-HAM-P). El quintico-RK45 se excluye,
igual que en la version original, porque RK45 no da una funcion continua
omega(alpha).

NOTA: este script asume que quartic_harmonic_balance.py ya fue ejecutado
antes en el notebook (define solve_quartic_hb).
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 10,
})

beta_fixed = 0.15
alpha_range = np.linspace(0.5, 2.0, 100)
eps = 1e-20

# ======================================================================
# Expresiones SR finales (Tabla II)
# ======================================================================
def sr_cubic(alpha, beta):
    return np.sqrt(alpha + 0.7500 * beta)

def sr_quartic(alpha, beta):
    return np.sqrt(alpha - beta * (0.7941304 / (
        (alpha / beta) - ((np.sqrt(beta) / ((alpha / beta) - 1.0)) / 1.1552787)
    )))

def sr_quintic_hamp(alpha, beta):
    return np.sqrt(alpha + 0.6250 * beta)

# ======================================================================
# Referencias
# ======================================================================
ref_cubic = np.sqrt(alpha_range + 0.75 * beta_fixed)
ref_quintic = np.sqrt(alpha_range + 0.625 * beta_fixed)

ref_quartic = []
prev = None
for a in alpha_range:
    guess = {"omega": prev["omega"], "c0": prev["c0"]} if prev else None
    sol = solve_quartic_hb(a, beta_fixed, A=1.0, n_harmonics=5, initial_guess=guess)
    prev = sol
    ref_quartic.append(sol["omega"])
ref_quartic = np.array(ref_quartic)

pred_cubic = sr_cubic(alpha_range, beta_fixed)
pred_quartic = sr_quartic(alpha_range, beta_fixed)
pred_quintic = sr_quintic_hamp(alpha_range, beta_fixed)

# ======================================================================
# Figura: 2 filas x 3 columnas
# ======================================================================
fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))

panels = [
    (ref_cubic, pred_cubic, r"(a) $x^3$", "LP reference"),
    (ref_quartic, pred_quartic, r"(b) $x^4$", "Multi-harm. balance"),
    (ref_quintic, pred_quintic := pred_quintic, r"(c) $x^5$", "HAM-P reference"),
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
