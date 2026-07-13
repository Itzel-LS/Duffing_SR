"""
generate_quintic_free_rk45_dataset.py
========================================

Cierra el hueco pendiente de la Tabla II: genera el dataset RK45+FFT
quintico libre completo (1000 muestras), usando rk45_fft_reference() de
rk45_reference.py -- que ya tiene la correccion de unidades omega=2*pi*f
aplicada (Rev2 #3 / Rev3 #3).

NOTA: este script asume que rk45_reference.py ya fue ejecutado antes en
el notebook (define rk45_fft_reference).

Salida: quintic_free_rk45_dataset.csv con columnas alpha, beta, omega
"""

import csv
import time

import numpy as np

N_SAMPLES = 1000
SEED = 42
ALPHA_RANGE = (0.5, 2.0)
BETA_RANGE = (0.01, 0.3)

OUTPUT_CSV = "quintic_free_rk45_dataset.csv"


def generate_dataset():
    rng = np.random.default_rng(SEED)
    alphas = rng.uniform(*ALPHA_RANGE, size=N_SAMPLES)
    betas = rng.uniform(*BETA_RANGE, size=N_SAMPLES)

    rows = []
    n_failed = 0

    t0 = time.time()
    for i, (a, b) in enumerate(zip(alphas, betas)):
        try:
            res = rk45_fft_reference(alpha=a, beta=b, n=5, delta=0.0, A0=1.0)
            rows.append({"alpha": a, "beta": b, "omega": res["omega_rad_s"]})
        except RuntimeError:
            n_failed += 1
            continue

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{N_SAMPLES} muestras procesadas...")

    elapsed = time.time() - t0
    print(f"\nCompletado en {elapsed:.1f} s. "
          f"{len(rows)} muestras exitosas, {n_failed} fallidas.")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "beta", "omega"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset guardado en: {OUTPUT_CSV}")
    return rows


if __name__ == "__main__":
    generate_dataset()
