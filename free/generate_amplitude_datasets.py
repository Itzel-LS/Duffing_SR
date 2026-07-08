"""
generate_amplitude_datasets.py
================================

Bloque B - punto #3 de la tabla de modificaciones (Rev1 #2, Rev4 #2).

El Revisor 1 propuso una prueba concreta: generar datos para varias
amplitudes (A = 0.5, 1.0, 1.5, 2.0) y verificar si SR recupera la
dependencia A^2 esperada en el caso cubico y A^4 en el caso quintico.

Este script genera, para cada uno de los tres casos libres (cubico,
cuartico, quintico), un dataset donde A ya NO esta fijo en 1, sino que es
una variable de entrada mas, muestreada sobre la rejilla que sugirio el
revisor. alpha y beta se siguen muestreando de forma continua, igual que
en el resto del estudio, para que SR vea variacion simultanea de los tres
parametros (alpha, beta, A) y pueda separar sus efectos.

Formulas de referencia usadas (con A explicita, sin fijar en 1):

    Cubico  (LP):     omega = sqrt(alpha + 0.75*beta*A^2)
    Quintico(HAM-P):  omega = sqrt(alpha + (5/8)*beta*A^4)
    Cuartico (HB multi-armonico, ver quartic_harmonic_balance.py):
                       omega = solve_quartic_hb(alpha, beta, A, n_harmonics=5)

NOTA IMPORTANTE: este script asume que la celda con `solve_quartic_hb`
(de quartic_harmonic_balance.py) ya fue ejecutada antes en el notebook.

Salida
------
    cubic_amplitude_dataset.csv     columnas: alpha, beta, A, omega
    quartic_amplitude_dataset.csv   columnas: alpha, beta, A, omega, max_residual
    quintic_amplitude_dataset.csv   columnas: alpha, beta, A, omega

Cada archivo contiene N_PER_AMPLITUDE * len(AMPLITUDE_GRID) filas en
total, con A tomando cada valor de la rejilla en una fraccion igual del
dataset. Esto es lo que alimentas a PySR, con [alpha, beta, A] como
variables de entrada y omega como objetivo -- si SR recupera A^2 o A^4
explicitamente en la expresion resultante, eso confirma que la relacion
amplitud-frecuencia si es aprendible con este protocolo.
"""

import csv
import time

import numpy as np

# ======================================================================
# PROTOCOLO DE MUESTREO
# ======================================================================
AMPLITUDE_GRID = [0.5, 1.0, 1.5, 2.0]   # sugerido explicitamente por Rev1
N_PER_AMPLITUDE = 250                    # 250 * 4 amplitudes = 1000 muestras totales
SEED = 42
ALPHA_RANGE = (0.5, 2.0)
BETA_RANGE = (0.01, 0.3)
N_HARMONICS_QUARTIC = 5


def _sample_alpha_beta(rng, n):
    alphas = rng.uniform(*ALPHA_RANGE, size=n)
    betas = rng.uniform(*BETA_RANGE, size=n)
    return alphas, betas


def generate_cubic_amplitude_dataset(filename="cubic_amplitude_dataset.csv"):
    rng = np.random.default_rng(SEED)
    rows = []
    for A in AMPLITUDE_GRID:
        alphas, betas = _sample_alpha_beta(rng, N_PER_AMPLITUDE)
        omegas = np.sqrt(alphas + 0.75 * betas * A**2)
        for a, b, w in zip(alphas, betas, omegas):
            rows.append({"alpha": a, "beta": b, "A": A, "omega": w})

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "beta", "A", "omega"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[cubico]  {len(rows)} muestras -> {filename}")
    return rows


def generate_quintic_amplitude_dataset(filename="quintic_amplitude_dataset.csv"):
    rng = np.random.default_rng(SEED + 1)  # semilla distinta para no correlacionar datasets
    rows = []
    for A in AMPLITUDE_GRID:
        alphas, betas = _sample_alpha_beta(rng, N_PER_AMPLITUDE)
        omegas = np.sqrt(alphas + (5.0 / 8.0) * betas * A**4)
        for a, b, w in zip(alphas, betas, omegas):
            rows.append({"alpha": a, "beta": b, "A": A, "omega": w})

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "beta", "A", "omega"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[quintico] {len(rows)} muestras -> {filename}")
    return rows


def quartic_potential(x, alpha, beta):
    """V(x) = 0.5*alpha*x^2 + (beta/5)*x^5, potencial del cuartico libre."""
    return 0.5 * alpha * x**2 + (beta / 5.0) * x**5


def is_bounded_orbit(alpha, beta, A):
    """
    Para  x'' + alpha*x + beta*x^4 = 0 (beta > 0), el potencial
    V(x) = 0.5*alpha*x^2 + (beta/5)*x^5 NO esta acotado inferiormente: tiene
    un maximo local (barrera) en x_c = -(alpha/beta)^(1/3), mas alla del
    cual V -> -infinito. Una orbita que parte del reposo en x=A solo es
    periodica (acotada) si su energia V(A) es menor que la altura de esa
    barrera, V(x_c). Si V(A) >= V(x_c), el sistema escapa hacia -infinito
    en vez de oscilar: no existe una "frecuencia de oscilacion" para ese
    (alpha, beta, A).

    Este criterio fue descubierto al diagnosticar fallas de convergencia
    del balance armonico a A grande: se confirmo que el 100% de los casos
    fisicamente acotados convergen (0 falsos negativos), pero el solver
    puede converger a una solucion ESPURIA en casos no acotados (falsos
    positivos) si no se filtra explicitamente con este criterio.
    """
    x_barrier = -(alpha / beta) ** (1 / 3)
    return quartic_potential(A, alpha, beta) < quartic_potential(x_barrier, alpha, beta)


def generate_quartic_amplitude_dataset(filename="quartic_amplitude_dataset.csv"):
    rng = np.random.default_rng(SEED + 2)
    rows = []
    n_unbounded_skipped = 0
    n_solver_failed = 0
    prev_solution = None

    t0 = time.time()
    for A in AMPLITUDE_GRID:
        alphas, betas = _sample_alpha_beta(rng, N_PER_AMPLITUDE)
        for a, b in zip(alphas, betas):
            # filtro fisico OBLIGATORIO, antes de intentar resolver
            if not is_bounded_orbit(a, b, A):
                n_unbounded_skipped += 1
                prev_solution = None  # no continuar la continuacion desde un caso invalido
                continue

            guess = None
            if prev_solution is not None:
                guess = {"omega": prev_solution["omega"], "c0": prev_solution["c0"]}
            try:
                sol = solve_quartic_hb(a, b, A, n_harmonics=N_HARMONICS_QUARTIC,
                                         initial_guess=guess)
                prev_solution = sol
            except RuntimeError:
                try:
                    sol = solve_quartic_hb(a, b, A, n_harmonics=N_HARMONICS_QUARTIC)
                    prev_solution = sol
                except RuntimeError:
                    n_solver_failed += 1
                    continue

            rows.append({
                "alpha": a, "beta": b, "A": A,
                "omega": sol["omega"], "max_residual": sol["max_residual"],
            })

    elapsed = time.time() - t0
    print(f"[cuartico] {len(rows)} muestras validas, "
          f"{n_unbounded_skipped} descartadas por orbita no acotada (fisico), "
          f"{n_solver_failed} fallidas por el solver (numerico) "
          f"({elapsed:.1f} s) -> {filename}")

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "beta", "A", "omega", "max_residual"])
        writer.writeheader()
        writer.writerows(rows)

    return rows


def sanity_check_scaling():
    """
    Verifica que las formulas cerradas (cubico, quintico) efectivamente
    escalan como se espera al variar A, con alpha y beta fijos -- esto es
    solo una verificacion rapida antes de generar los datasets completos.
    """
    alpha, beta = 1.0, 0.15
    print("=== Verificacion de escalamiento con A (alpha=1.0, beta=0.15 fijos) ===\n")
    print(f"{'A':>6} | {'omega_cubico':>14} | {'omega_quintico':>16}")
    print("-" * 42)
    for A in AMPLITUDE_GRID:
        w_cubic = np.sqrt(alpha + 0.75 * beta * A**2)
        w_quintic = np.sqrt(alpha + (5.0 / 8.0) * beta * A**4)
        print(f"{A:>6.2f} | {w_cubic:>14.6f} | {w_quintic:>16.6f}")


if __name__ == "__main__":
    sanity_check_scaling()
    print()
    generate_cubic_amplitude_dataset()
    generate_quintic_amplitude_dataset()
    generate_quartic_amplitude_dataset()
