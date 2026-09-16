"""
regenerate_residuals_free.py (v2, con datos reales)
=======================================================
Regenera residuals_free.png con datos REALES en las 3 filas:
  - cubico y quintico: cubic_amplitude_dataset.csv / quintic_amplitude_dataset.csv
    (1000 muestras cada uno, A en {0.5,1.0,1.5,2.0}), residuo contra la formula
    exacta que SR reproduce (~1e-16, tal como el manuscrito ya afirma).
  - cuartico: quartic_free_variableA.npz (662 muestras, dataset corregido),
    residuo contra la ecuacion SR nueva.

Uso:
    python regenerate_residuals_free.py
Salida:
    residuals_free.png
"""
import numpy as np
import matplotlib.pyplot as plt
import csv


def load_csv(path, cols):
    data = {c: [] for c in cols}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            for c in cols:
                data[c].append(float(row[c]))
    return {c: np.array(v) for c, v in data.items()}


# ---------- fila cubica: datos reales ----------
d_top = load_csv("cubic_amplitude_dataset.csv", ["alpha", "beta", "A", "omega"])
ref_top = np.sqrt(d_top["alpha"] + 0.75 * d_top["A"] ** 2 * d_top["beta"])
r_top = d_top["omega"] - ref_top

# ---------- fila cuartica: dataset corregido ----------
d_mid = load_csv("quartic_amplitude_dataset.csv", ["alpha", "beta", "A", "omega"])
alpha_m, beta_m, A_m, omega_m = d_mid["alpha"], d_mid["beta"], d_mid["A"], d_mid["omega"]
sr_m = np.sqrt(alpha_m - ((A_m * ((((A_m * A_m) * beta_m) *
       (A_m * ((A_m * (3.92628 * (A_m * np.sqrt(beta_m)))) - alpha_m))) *
       (beta_m * 1.5603521))) * (A_m / alpha_m)))
r_mid = omega_m - sr_m

# ---------- fila quintica: datos reales ----------
d_bot = load_csv("quintic_amplitude_dataset.csv", ["alpha", "beta", "A", "omega"])
ref_bot = np.sqrt(d_bot["alpha"] + 0.625 * d_bot["A"] ** 4 * d_bot["beta"])
r_bot = d_bot["omega"] - ref_bot

rows = [
    ("$x^3$ free\nResidual ($\\omega$)", d_top["alpha"], d_top["beta"], d_top["A"], r_top),
    ("$x^4$ free\nResidual ($\\omega$)", alpha_m, beta_m, A_m, r_mid),
    ("$x^5$ free\nResidual ($\\omega$)", d_bot["alpha"], d_bot["beta"], d_bot["A"], r_bot),
]
col_labels = [r"$\alpha$", r"$\beta$", r"$A$"]

fig, axes = plt.subplots(3, 3, figsize=(13, 10))
for row_i, (ylabel, alpha_r, beta_r, A_r, resid_r) in enumerate(rows):
    xs = [alpha_r, beta_r, A_r]
    for col_i, xlabel in enumerate(col_labels):
        ax = axes[row_i, col_i]
        sc = ax.scatter(xs[col_i], resid_r, c=A_r, cmap="viridis", s=14)
        ax.axhline(0, color="red", ls="--", lw=0.7)
        if col_i == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        if row_i == 2:
            ax.set_xlabel(xlabel)
        if row_i == 0 and col_i == 2:
            plt.colorbar(sc, ax=ax, label="$A$")

plt.tight_layout()
plt.savefig("residuals_free.png", dpi=200)
print("Guardado: residuals_free.png (datos reales en las 3 filas)")
print(f"  cubico:  max|resid|={np.abs(r_top).max():.3e}")
print(f"  cuartico: max|resid|={np.abs(r_mid).max():.3e}")
print(f"  quintico: max|resid|={np.abs(r_bot).max():.3e}")
