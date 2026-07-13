"""
figure2_free_scatter.py
=========================

Regenera la Figura 2 (predicho vs. referencia, 4 paneles) con los datos
corregidos: cuartico via balance multi-armonico (ya no la Ec. 5 retirada),
y quintico RK45 con la correccion de unidades omega=2*pi*f ya aplicada.

Panel (a): cubico, referencia LP.
Panel (b): cuartico, referencia balance multi-armonico.
Panel (c): quintico, referencia RK45+FFT.
Panel (d): quintico, referencia HAM-P.

Requiere: quartic_free_dataset.csv (generado en el Bloque A) y
quintic_free_rk45_test200.csv (200 muestras de prueba, generadas aparte).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 10,
})

SEED = 42

# ======================================================================
# Expresiones SR finales (ya confirmadas, Tabla II)
# ======================================================================
def sr_cubic(alpha, beta):
    return np.sqrt(alpha + 0.7500 * beta)

def sr_quartic(alpha, beta):
    # expresion real descubierta por SR (Tabla II, R^2=0.999999)
    return np.sqrt(alpha - beta * (0.7941304 / (
        (alpha / beta) - ((np.sqrt(beta) / ((alpha / beta) - 1.0)) / 1.1552787)
    )))

def sr_quintic_rk45(alpha, beta):
    return np.sqrt(alpha + 0.6178 * beta)

def sr_quintic_hamp(alpha, beta):
    return np.sqrt(alpha + 0.6250 * beta)

# ======================================================================
# Panel (a): cubico, LP
# ======================================================================
rng = np.random.default_rng(SEED)
alpha_a = rng.uniform(0.5, 2.0, 1000)
beta_a = rng.uniform(0.01, 0.3, 1000)
omega_ref_a = np.sqrt(alpha_a + 0.75 * beta_a)
_, alpha_a_test, _, beta_a_test = train_test_split(
    alpha_a, beta_a, test_size=0.2, random_state=SEED)
_, omega_a_test = train_test_split(omega_ref_a, test_size=0.2, random_state=SEED)
pred_a = sr_cubic(alpha_a_test, beta_a_test)

# ======================================================================
# Panel (b): cuartico, balance multi-armonico
# ======================================================================
df_b = pd.read_csv("quartic_free_dataset.csv")
_, df_b_test = train_test_split(df_b, test_size=0.2, random_state=SEED)
ref_b = df_b_test["omega_hb"].values
pred_b = sr_quartic(df_b_test["alpha"].values, df_b_test["beta"].values)

# ======================================================================
# Panel (c): quintico, RK45+FFT (dataset completo real, split 80/20 oficial)
# ======================================================================
df_c_full = pd.read_csv("quintic_free_rk45_dataset.csv")
_, df_c_test = train_test_split(df_c_full, test_size=0.2, random_state=SEED)
ref_c = df_c_test["omega"].values
pred_c = sr_quintic_rk45(df_c_test["alpha"].values, df_c_test["beta"].values)

# ======================================================================
# Panel (d): quintico, HAM-P
# ======================================================================
rng2 = np.random.default_rng(SEED)
alpha_d = rng2.uniform(0.5, 2.0, 1000)
beta_d = rng2.uniform(0.01, 0.3, 1000)
omega_ref_d = np.sqrt(alpha_d + 0.625 * beta_d)
_, alpha_d_test, _, beta_d_test = train_test_split(
    alpha_d, beta_d, test_size=0.2, random_state=SEED)
_, omega_d_test = train_test_split(omega_ref_d, test_size=0.2, random_state=SEED)
pred_d = sr_quintic_hamp(alpha_d_test, beta_d_test)

# ======================================================================
# Figura: 1x4 paneles
# ======================================================================
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

panels = [
    (omega_a_test, pred_a, "(a) $x^3$ Free Vibration", "LP"),
    (ref_b, pred_b, "(b) $x^4$ Free Vibration", "Multi-harm. balance"),
    (ref_c, pred_c, "(c) $x^5$ Free Vibration", "RK45+FFT"),
    (omega_d_test, pred_d, "(d) $x^5$ Free Vibration", "HAM-P"),
]

for ax, (ref, pred, title, method) in zip(axes, panels):
    r2 = r2_score(ref, pred)
    lims = [min(ref.min(), pred.min()), max(ref.max(), pred.max())]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.6)
    ax.scatter(ref, pred, s=10, alpha=0.6, color='C0')
    ax.set_xlabel(f"Reference $\\omega$ ({method})")
    ax.set_ylabel("Predicted $\\omega$")
    ax.set_title(f"{title}\n$R^2$ = {r2:.4f}")
    ax.grid(True, linestyle=':', alpha=0.3)

plt.tight_layout()
plt.savefig("Figure_2_free_scatter.png", dpi=300, bbox_inches="tight")
print("Guardada: Figure_2_free_scatter.png")
for (ref, pred, title, method) in panels:
    print(f"{title}: R^2={r2_score(ref,pred):.6f}, MSE={mean_squared_error(ref,pred):.3e}")
