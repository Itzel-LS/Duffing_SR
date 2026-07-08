"""
generate_quartic_dataset.py
=============================

Cierra el Bloque A: genera el dataset completo (1000 muestras) para el
caso cuartico libre, usando la referencia corregida por balance armonico
multi-armonico (quartic_harmonic_balance.py) en vez de la Ec. (5) retirada.

Reemplaza, en la practica, la linea del pipeline original que hacia:

    omega_ref_2L = np.sqrt(ALPHA + (3*BETA**2*(np.sqrt(ALPHA))**6)/(16*ALPHA))

por una evaluacion del balance armonico convergido (M=5 armonicos) para
cada combinacion muestreada de (alpha, beta), con A=1 fijo (normalizacion
acordada en la Observacion 2 de Rev1 / Rev4).

Salida
------
    quartic_free_dataset.csv   con columnas: alpha, beta, omega_hb, max_residual

Ese CSV es la entrada que debes alimentar a PySR en tu entorno local
(reemplazando el dataset anterior generado con la formula retirada), con
las mismas particiones train/test (80/20, seed=42) que usa el resto del
estudio para mantener consistencia metodologica.

Uso
---
    python generate_quartic_dataset.py
"""

from __future__ import annotations

import csv
import time

import numpy as np

from quartic_harmonic_balance import solve_quartic_hb

# ======================================================================
# PROTOCOLO DE MUESTREO (identico al resto del estudio, Ecs. 8-9 del
# manuscrito revisado, para no introducir una inconsistencia nueva)
# ======================================================================
N_SAMPLES = 1000
SEED = 42
ALPHA_RANGE = (0.5, 2.0)
BETA_RANGE = (0.01, 0.3)
A_FIXED = 1.0          # normalizacion de amplitud acordada (Observacion 2)
N_HARMONICS = 5        # nivel de convergencia validado (residuo ~1.6e-4)

OUTPUT_CSV = "quartic_free_dataset.csv"


def generate_dataset():
    rng = np.random.default_rng(SEED)
    alphas = rng.uniform(*ALPHA_RANGE, size=N_SAMPLES)
    betas = rng.uniform(*BETA_RANGE, size=N_SAMPLES)

    rows = []
    n_failed = 0
    prev_solution = None  # usar la solucion previa como initial_guess (continuidad)

    t0 = time.time()
    for i, (a, b) in enumerate(zip(alphas, betas)):
        guess = None
        if prev_solution is not None:
            guess = {"omega": prev_solution["omega"], "c0": prev_solution["c0"]}
        try:
            sol = solve_quartic_hb(a, b, A_FIXED, n_harmonics=N_HARMONICS,
                                     initial_guess=guess)
            prev_solution = sol
        except RuntimeError:
            # reintentar desde una semilla neutra si la continuidad falla
            try:
                sol = solve_quartic_hb(a, b, A_FIXED, n_harmonics=N_HARMONICS)
                prev_solution = sol
            except RuntimeError:
                n_failed += 1
                continue

        rows.append({
            "alpha": a,
            "beta": b,
            "omega_hb": sol["omega"],
            "max_residual": sol["max_residual"],
        })

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{N_SAMPLES} muestras procesadas...")

    elapsed = time.time() - t0
    print(f"\nCompletado en {elapsed:.1f} s. "
          f"{len(rows)} muestras exitosas, {n_failed} fallidas.")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "beta", "omega_hb", "max_residual"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset guardado en: {OUTPUT_CSV}")

    residuals = np.array([r["max_residual"] for r in rows])
    print(f"\nDiagnostico de calidad del dataset:")
    print(f"  residuo maximo (peor caso)   : {residuals.max():.2e}")
    print(f"  residuo maximo (mediana)     : {np.median(residuals):.2e}")
    print(f"  muestras con residuo > 1e-2  : {(residuals > 1e-2).sum()} "
          f"({100 * (residuals > 1e-2).mean():.1f}%)")

    return rows


if __name__ == "__main__":
    generate_dataset()
