"""
generate_quintic_forced_rk45_dataset.py
==========================================

Genera el dataset RK45 forzado quintico completo (1000 muestras),
completando la fila pendiente de la Tabla IV.

NOTA: este script asume que rk45_forced_reference.py ya fue ejecutado
antes en el notebook (define rk45_forced_amplitude).

Mismos rangos de muestreo que el resto del caso forzado:
    alpha    ~ U(0.5, 2.0)
    beta     ~ U(0.01, 0.3)
    delta    ~ U(0.01, 0.2)
    F0       ~ U(0.1, 1.5)
    omega_ext ~ U(0.8, 1.5)

Salida: quintic_forced_rk45_dataset.csv con columnas
        alpha, beta, delta, F0, omega_ext, A, fit_residual
"""

import csv
import time

import numpy as np

N_SAMPLES = 1000
SEED = 42
ALPHA_RANGE = (0.5, 2.0)
BETA_RANGE = (0.01, 0.3)
DELTA_RANGE = (0.01, 0.2)
F0_RANGE = (0.1, 1.5)
OMEGA_EXT_RANGE = (0.8, 1.5)

OUTPUT_CSV = "quintic_forced_rk45_dataset.csv"


def generate_dataset():
    rng = np.random.default_rng(SEED)
    alphas = rng.uniform(*ALPHA_RANGE, size=N_SAMPLES)
    betas = rng.uniform(*BETA_RANGE, size=N_SAMPLES)
    deltas = rng.uniform(*DELTA_RANGE, size=N_SAMPLES)
    F0s = rng.uniform(*F0_RANGE, size=N_SAMPLES)
    omega_exts = rng.uniform(*OMEGA_EXT_RANGE, size=N_SAMPLES)

    rows = []
    n_failed = 0

    t0 = time.time()
    for i, (a, b, d, F0, we) in enumerate(zip(alphas, betas, deltas, F0s, omega_exts)):
        try:
            res = rk45_forced_amplitude(a, b, d, F0, we)
            rows.append({
                "alpha": a, "beta": b, "delta": d, "F0": F0, "omega_ext": we,
                "A": res["A"], "fit_residual": res["fit_residual"],
            })
        except RuntimeError:
            n_failed += 1
            continue

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{N_SAMPLES} muestras procesadas...")

    elapsed = time.time() - t0
    print(f"\nCompletado en {elapsed/60:.1f} min. "
          f"{len(rows)} muestras exitosas, {n_failed} fallidas/no acotadas.")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "beta", "delta", "F0",
                                                  "omega_ext", "A", "fit_residual"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset guardado en: {OUTPUT_CSV}")

    residuals = np.array([r["fit_residual"] for r in rows])
    print(f"\nDiagnostico de calidad (residuo de ajuste relativo al fundamental):")
    print(f"  mediana: {np.median(residuals):.4e}")
    print(f"  maximo:  {residuals.max():.4e}")
    print(f"  muestras con residuo > 0.10 (posible contenido armonico fuerte): "
          f"{(residuals > 0.10).sum()} ({100*(residuals>0.10).mean():.1f}%)")

    return rows


if __name__ == "__main__":
    generate_dataset()
