"""
residual_analysis.py
======================

Bloque de validacion adicional, Pieza A (Rev1 #9, Rev4 #9): residuales de
prediccion en funcion de los parametros del sistema, en vez de un unico
R^2 global. No requiere reentrenar PySR -- usa las expresiones ya
descubiertas y los datasets ya generados.

Genera:
  - residuals_free.png    : residuales del caso libre (amplitud variable)
                             vs. alpha, beta, A
  - residuals_forced.png  : residuales del caso forzado vs. alpha, beta,
                             F0, y el desajuste (detuning) omega_ext - sqrt(alpha)

Requiere los CSV ya generados: cubic_amplitude_dataset.csv,
quartic_amplitude_dataset.csv, quintic_amplitude_dataset.csv,
cubic_forced_dataset.csv, quartic_forced_dataset.csv,
quintic_forced_dataset.csv (en la carpeta actual).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

# ======================================================================
# Expresiones SR ya descubiertas (Bloque B: amplitud variable)
# ======================================================================
def sr_cubic_free(alpha, beta, A):
    return np.sqrt(alpha + 0.7500 * beta * A**2)

def sr_quartic_free(alpha, beta, A):
    return np.sqrt(alpha + A * (np.sqrt(beta / alpha) * A) * (A * beta) * (A / -2.6582658))

def sr_quintic_free(alpha, beta, A):
    return np.sqrt(alpha + 0.6250 * beta * A**4)

# Expresiones SR ya descubiertas (Bloque C: forzado)
def sr_cubic_forced(alpha, beta, F0, omega_ext):
    return (np.sqrt(F0) * 2.7030935) - np.cbrt(
        (F0 - (omega_ext**2 - alpha)) *
        (((omega_ext + (omega_ext - alpha)) - np.cbrt(beta / 0.05756853)) / -0.68097097)
    )

def sr_quartic_forced(alpha, beta, F0, omega_ext):
    return np.sqrt(F0 / np.cbrt(beta)) - (np.cbrt(omega_ext - 0.977861) * (omega_ext - alpha))

def sr_quintic_forced(alpha, beta, F0, omega_ext):
    return np.sqrt(
        np.cbrt((0.8809793 - (omega_ext - alpha)) / beta) *
        (omega_ext - ((alpha - 0.35662863) - omega_ext)) * F0
    )


# ======================================================================
# FIGURA 1: Residuales del caso libre vs. alpha, beta, A
# ======================================================================
def plot_free_residuals():
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    cases = [
        ("cubic_amplitude_dataset.csv", sr_cubic_free, "omega", r"$x^3$ free"),
        ("quartic_amplitude_dataset.csv", sr_quartic_free, "omega", r"$x^4$ free"),
        ("quintic_amplitude_dataset.csv", sr_quintic_free, "omega", r"$x^5$ free"),
    ]

    for row, (csv_path, sr_func, target_col, title) in enumerate(cases):
        df = pd.read_csv(csv_path)
        y_true = df[target_col].values
        y_pred = sr_func(df["alpha"].values, df["beta"].values, df["A"].values)
        residual = y_true - y_pred

        for col, (param, xlabel) in enumerate([
            ("alpha", r"$\alpha$"), ("beta", r"$\beta$"), ("A", r"$A$")
        ]):
            ax = axes[row, col]
            sc = ax.scatter(df[param], residual, c=df["A"], cmap="viridis",
                             s=8, alpha=0.6)
            ax.axhline(0, color="red", linewidth=0.8, linestyle="--")
            ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel(f"{title}\nResidual ($\\omega$)")
            ax.grid(True, linestyle=":", alpha=0.3)
            if row == 0 and col == 2:
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label("A", fontsize=8)

    plt.tight_layout()
    plt.savefig("residuals_free.png", dpi=250, bbox_inches="tight")
    print("Guardada: residuals_free.png")


# ======================================================================
# FIGURA 2: Residuales del caso forzado vs. alpha, beta, F0, detuning
# ======================================================================
def plot_forced_residuals():
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))

    cases = [
        ("cubic_forced_dataset.csv", sr_cubic_forced, r"$x^3$ forced"),
        ("quartic_forced_dataset.csv", sr_quartic_forced, r"$x^4$ forced"),
        ("quintic_forced_dataset.csv", sr_quintic_forced, r"$x^5$ forced"),
    ]

    for row, (csv_path, sr_func, title) in enumerate(cases):
        df = pd.read_csv(csv_path)
        y_true = df["A"].values
        y_pred = sr_func(df["alpha"].values, df["beta"].values,
                          df["F0"].values, df["omega_ext"].values)
        residual = y_true - y_pred
        detuning = df["omega_ext"].values - np.sqrt(df["alpha"].values)

        params = [
            ("alpha", r"$\alpha$", df["alpha"]),
            ("beta", r"$\beta$", df["beta"]),
            ("F0", r"$F_0$", df["F0"]),
            (None, r"$\omega_{\mathrm{ext}} - \sqrt{\alpha}$ (detuning)", detuning),
        ]

        for col, (param, xlabel, xvals) in enumerate(params):
            ax = axes[row, col]
            sc = ax.scatter(xvals, residual, c=df["beta"], cmap="plasma",
                             s=8, alpha=0.6)
            ax.axhline(0, color="red", linewidth=0.8, linestyle="--")
            ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel(f"{title}\nResidual ($A$)")
            ax.grid(True, linestyle=":", alpha=0.3)
            if row == 0 and col == 3:
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label(r"$\beta$", fontsize=8)

    plt.tight_layout()
    plt.savefig("residuals_forced.png", dpi=250, bbox_inches="tight")
    print("Guardada: residuals_forced.png")


if __name__ == "__main__":
    plot_free_residuals()
    plot_forced_residuals()
