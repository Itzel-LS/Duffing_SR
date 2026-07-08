"""
figure6_amplitude.py
======================
Genera la Figura 6 (nueva): omega vs A para los tres casos libres, con
alpha y beta fijos, comparando la referencia y la expresion SR
descubierta (Tabla IV) -- analoga en estilo a la Fig. 3 del manuscrito
(curvas arriba, error absoluto en escala log abajo).
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

alpha_fixed = 1.0
beta_fixed = 0.15
A_range = np.linspace(0.3, 2.0, 200)
eps = 1e-18

# --- referencias y SR (cubico, quintico: forma cerrada) ---
def omega_ref_cubic(A):
    return np.sqrt(alpha_fixed + 0.75 * beta_fixed * A**2)

def omega_sr_cubic(A):
    return np.sqrt(alpha_fixed + 0.7500 * beta_fixed * A**2)

def omega_ref_quintic(A):
    return np.sqrt(alpha_fixed + (5.0/8.0) * beta_fixed * A**4)

def omega_sr_quintic(A):
    return np.sqrt(alpha_fixed + 0.6250 * beta_fixed * A**4)

# --- cuartico: usar el balance armonico como "referencia", y filtrar
#     la region no acotada (barrera de potencial) ---
def quartic_potential(x, alpha, beta):
    return 0.5*alpha*x**2 + (beta/5.0)*x**5

def is_bounded(alpha, beta, A):
    xc = -(alpha/beta)**(1/3)
    return quartic_potential(A, alpha, beta) < quartic_potential(xc, alpha, beta)

# Nota: para no depender de scipy.optimize aqui, usamos la forma
# perturbativa validada (n_harmonics=5) via una funcion ligera reimplementada
import sys
sys.path.insert(0, '/mnt/user-data/outputs')
from quartic_harmonic_balance import solve_quartic_hb

omega_ref_quartic = []
A_bounded_quartic = []
prev = None
for A in A_range:
    if not is_bounded(alpha_fixed, beta_fixed, A):
        prev = None
        continue
    guess = {"omega": prev["omega"], "c0": prev["c0"]} if prev else None
    try:
        sol = solve_quartic_hb(alpha_fixed, beta_fixed, A, n_harmonics=5, initial_guess=guess)
        prev = sol
        omega_ref_quartic.append(sol["omega"])
        A_bounded_quartic.append(A)
    except RuntimeError:
        prev = None
        continue

A_bounded_quartic = np.array(A_bounded_quartic)
omega_ref_quartic = np.array(omega_ref_quartic)

# expresion SR cuartica (simplificada para graficar; ver Tabla IV)
def omega_sr_quartic(A, alpha=alpha_fixed, beta=beta_fixed):
    return np.sqrt(alpha + A*(np.sqrt(beta/alpha)*A)*(A*beta)*(A/-2.6582658))

omega_sr_quartic_vals = omega_sr_quartic(A_bounded_quartic)

# ======================================================================
# FIGURA: 2 filas x 3 columnas (curvas arriba, error abajo)
# ======================================================================
fig, axes = plt.subplots(2, 3, figsize=(13, 7))

cases = [
    (A_range, omega_ref_cubic(A_range), omega_sr_cubic(A_range),
     r'(a) $x^3$', r'LP reference'),
    (A_bounded_quartic, omega_ref_quartic, omega_sr_quartic_vals,
     r'(b) $x^4$', r'Multi-harmonic balance'),
    (A_range, omega_ref_quintic(A_range), omega_sr_quintic(A_range),
     r'(c) $x^5$', r'HAM-P reference'),
]

for i, (A_vals, y_ref, y_sr, title, ref_label) in enumerate(cases):
    ax_top = axes[0, i]
    ax_top.plot(A_vals, y_ref, linewidth=2.2, label='Reference')
    ax_top.plot(A_vals, y_sr, linewidth=2.0, linestyle='--', label='SR')
    ax_top.set_xlabel(r'Amplitude $A$')
    ax_top.set_title(f'{title}\n{ref_label}')
    ax_top.grid(True, linestyle=':', alpha=0.3)
    ax_top.legend(frameon=False)
    if i == 0:
        ax_top.set_ylabel(r'Frequency $\omega$')

    ax_bot = axes[1, i]
    error = np.abs(y_ref - y_sr) + eps
    ax_bot.plot(A_vals, error, linewidth=1.8, color='C3')
    ax_bot.fill_between(A_vals, error, alpha=0.15, color='C3')
    ax_bot.set_yscale('log')
    ax_bot.set_xlabel(r'Amplitude $A$')
    ax_bot.grid(True, linestyle=':', alpha=0.3)
    ax_bot.text(0.5, 0.85, f'Max = {error.max():.1e}',
                transform=ax_bot.transAxes, ha='center', fontsize=9)
    if i == 0:
        ax_bot.set_ylabel('Absolute error (log scale)')

plt.tight_layout()
plt.savefig('Figure_6_amplitude.png', dpi=300, bbox_inches='tight')
print("Guardada: Figure_6_amplitude.png")
