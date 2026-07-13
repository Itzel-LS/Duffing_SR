"""
figure5_forced_amplitude.py
==============================

Bloque C - regenera la Figura 5: ahora que el objetivo es la amplitud de
respuesta A (no omega), la figura fisicamente relevante es la curva de
resonancia clasica: A vs. omega_ext, a alpha/beta/delta/F0 fijos,
comparando la referencia (Ec. 13 cerrada para cubico/quintico, balance
multi-armonico para cuartico) contra la expresion descubierta por SR.

NOTA: este script asume que forced_resonance.py ya fue ejecutado antes en
el notebook (define solve_forced_closed_form, pick_branch, solve_forced_hb).
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# ======================================================================
# Parametros fijos representativos (dentro del rango de entrenamiento)
# ======================================================================
alpha_fixed = 1.0
beta_fixed = 0.15
delta_fixed = 0.10
F0_fixed = 0.5
omega_ext_range = np.linspace(0.8, 1.5, 250)  # mismo rango de muestreo (Ec. de sampling)
eps = 1e-18

# ======================================================================
# Expresiones SR descubiertas (Bloque C, resumen final)
# ======================================================================
def A_sr_cubic(alpha, beta, F0, omega_ext):
    return (np.sqrt(F0) * 2.7030935) - np.cbrt(
        (F0 - (omega_ext**2 - alpha)) *
        (((omega_ext + (omega_ext - alpha)) - np.cbrt(beta / 0.05756853)) / -0.68097097)
    )

def A_sr_quartic(alpha, beta, F0, omega_ext):
    return np.sqrt(F0 / np.cbrt(beta)) - (np.cbrt(omega_ext - 0.977861) * (omega_ext - alpha))

def A_sr_quintic(alpha, beta, F0, omega_ext):
    return np.sqrt(
        np.cbrt((0.8809793 - (omega_ext - alpha)) / beta) *
        (omega_ext - ((alpha - 0.35662863) - omega_ext)) * F0
    )

# ======================================================================
# Referencias (barrido en omega_ext, con continuidad de rama)
# ======================================================================
def reference_curve_cubic_or_quintic(n, alpha, beta, delta, F0, omega_ext_range):
    A_vals = []
    prev_u = None
    for we in omega_ext_range:
        u_roots = solve_forced_closed_form(alpha, beta, delta, F0, we, n)
        if len(u_roots) == 0:
            A_vals.append(np.nan)
            prev_u = None
            continue
        u_sel = pick_branch(u_roots, prev_u)
        prev_u = u_sel
        A_vals.append(np.sqrt(u_sel))
    return np.array(A_vals)

def reference_curve_quartic(alpha, beta, delta, F0, omega_ext_range):
    """
    Barrido en dos direcciones desde el punto mas cercano a la resonancia
    (donde la convergencia es mas facil), usando continuidad hacia afuera
    en ambos sentidos -- mucho mas robusto que un barrido de izquierda a
    derecha con semillas por defecto en cada punto.
    """
    n = len(omega_ext_range)
    A_vals = np.full(n, np.nan)
    omega_nat = np.sqrt(alpha)
    start_idx = int(np.argmin(np.abs(omega_ext_range - omega_nat)))

    def solve_one(we, guess):
        try:
            hb = solve_forced_hb(alpha, beta, delta, F0, we, n=4, n_harmonics=5,
                                   initial_guess=guess)
            return hb["A"], hb["params"]
        except RuntimeError:
            try:
                hb = solve_forced_hb(alpha, beta, delta, F0, we, n=4, n_harmonics=5)
                return hb["A"], hb["params"]
            except RuntimeError:
                return np.nan, None

    A_vals[start_idx], guess = solve_one(omega_ext_range[start_idx], None)

    g = guess
    for i in range(start_idx + 1, n):
        A_vals[i], g = solve_one(omega_ext_range[i], g)
        if g is None:
            g = guess  # reintentar desde el punto de partida si se pierde la continuidad

    g = guess
    for i in range(start_idx - 1, -1, -1):
        A_vals[i], g = solve_one(omega_ext_range[i], g)
        if g is None:
            g = guess

    return A_vals

# ======================================================================
# Calcular las tres curvas
# ======================================================================
A_ref_cubic = reference_curve_cubic_or_quintic(3, alpha_fixed, beta_fixed, delta_fixed,
                                                  F0_fixed, omega_ext_range)
A_ref_quintic = reference_curve_cubic_or_quintic(5, alpha_fixed, beta_fixed, delta_fixed,
                                                    F0_fixed, omega_ext_range)
A_ref_quartic = reference_curve_quartic(alpha_fixed, beta_fixed, delta_fixed, F0_fixed,
                                          omega_ext_range)

# El cuartico forzado no converge para todo el rango de omega_ext (region sin
# solucion periodica acotada, analogo al hallazgo del caso libre) -- se
# restringe el panel a la subregion valida, documentado en el pie de figura.
quartic_valid = ~np.isnan(A_ref_quartic)
omega_ext_quartic_valid = omega_ext_range[quartic_valid]
A_ref_quartic_valid = A_ref_quartic[quartic_valid]

A_sr_cubic_vals = A_sr_cubic(alpha_fixed, beta_fixed, F0_fixed, omega_ext_range)
A_sr_quartic_vals = A_sr_quartic(alpha_fixed, beta_fixed, F0_fixed, omega_ext_quartic_valid)
A_sr_quintic_vals = A_sr_quintic(alpha_fixed, beta_fixed, F0_fixed, omega_ext_range)

# ======================================================================
# Figura: 2 filas x 3 columnas
# ======================================================================
fig, axes = plt.subplots(2, 3, figsize=(13, 7))

cases = [
    (omega_ext_range, A_ref_cubic, A_sr_cubic_vals, r'(a) $x^3$ forced', 'Single-harm. balance'),
    (omega_ext_quartic_valid, A_ref_quartic_valid, A_sr_quartic_vals, r'(b) $x^4$ forced',
     'Multi-harm. balance\n(bounded-orbit region only)'),
    (omega_ext_range, A_ref_quintic, A_sr_quintic_vals, r'(c) $x^5$ forced', 'Single-harm. balance'),
]

for i, (we_vals, y_ref, y_sr, title, ref_label) in enumerate(cases):
    ax_top = axes[0, i]
    ax_top.plot(we_vals, y_ref, linewidth=2.2, label='Reference')
    ax_top.plot(we_vals, y_sr, linewidth=2.0, linestyle='--', label='SR')
    ax_top.set_xlabel(r'$\omega_{\mathrm{ext}}$')
    ax_top.set_title(f'{title}\n{ref_label}')
    ax_top.grid(True, linestyle=':', alpha=0.3)
    ax_top.legend(frameon=False)
    if i == 0:
        ax_top.set_ylabel(r'Response amplitude $A$')

    ax_bot = axes[1, i]
    error = np.abs(y_ref - y_sr) + eps
    ax_bot.plot(we_vals, error, linewidth=1.8, color='C3')
    ax_bot.fill_between(we_vals, error, alpha=0.15, color='C3')
    ax_bot.set_yscale('log')
    ax_bot.set_xlabel(r'$\omega_{\mathrm{ext}}$')
    ax_bot.grid(True, linestyle=':', alpha=0.3)
    finite_error = error[np.isfinite(error)]
    if len(finite_error):
        ax_bot.text(0.5, 0.85, f'Max = {np.nanmax(finite_error):.2e}',
                    transform=ax_bot.transAxes, ha='center', fontsize=9)
    if i == 0:
        ax_bot.set_ylabel('Absolute error (log scale)')

plt.tight_layout()
plt.savefig('Figure_5_forced_amplitude.png', dpi=300, bbox_inches='tight')
print("Guardada: Figure_5_forced_amplitude.png")
