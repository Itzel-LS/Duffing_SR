"""
generate_forced_datasets.py
=============================

Bloque C - genera los tres datasets forzados usando la metodologia
aprobada:
  - Cubico y quintico: forma cerrada (Ec. 13-tipo, balance de un armonico),
    con seleccion de rama por continuidad cerca de la resonancia.
  - Cuartico: balance armonico multi-armonico numerico (unico camino
    valido, por la asimetria del potencial).

Variable objetivo: la AMPLITUD DE RESPUESTA A (no "omega", que en estado
estacionario es simplemente omega_ext -- ver discusion en Sec. II.A del
manuscrito revisado). F0 se incluye como variable de entrada explicita
(Rev1 #3c/Rev4 #3c pedian esto).

Rangos de muestreo (identicos a los ya usados en el resto del estudio):
    alpha    ~ U(0.5, 2.0)
    beta     ~ U(0.01, 0.3)
    delta    ~ U(0.01, 0.2)
    F0       ~ U(0.1, 1.5)
    omega_ext ~ U(0.8, 1.5)   (mismo rango del script original)

NOTA: este script asume que forced_resonance.py ya fue ejecutado antes en
el notebook (define solve_forced_closed_form, pick_branch, solve_forced_hb).
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
N_HARMONICS_QUARTIC = 5


def _sample_forced_params(rng, n):
    alphas = rng.uniform(*ALPHA_RANGE, size=n)
    betas = rng.uniform(*BETA_RANGE, size=n)
    deltas = rng.uniform(*DELTA_RANGE, size=n)
    F0s = rng.uniform(*F0_RANGE, size=n)
    omega_exts = rng.uniform(*OMEGA_EXT_RANGE, size=n)
    return alphas, betas, deltas, F0s, omega_exts


def generate_forced_closed_form_dataset(n, filename, single_branch_only=True):
    """Cubico (n=3) o quintico (n=5), forma cerrada.

    single_branch_only=True (recomendado): descarta las muestras donde el
    polinomio en A^2 tiene mas de una raiz real positiva (region de
    histeresis/salto cerca de la resonancia). Sin un criterio de
    estabilidad explicito, la eleccion de rama en esos casos es arbitraria
    y actua como ruido de etiqueta para SR; se documenta como muestras
    excluidas, igual que se hizo con la barrera de potencial del cuartico
    libre.
    """
    rng = np.random.default_rng(SEED + n)
    alphas, betas, deltas, F0s, omega_exts = _sample_forced_params(rng, N_SAMPLES)

    rows = []
    n_no_real_root = 0
    n_multivalued_excluded = 0

    for a, b, d, F0, we in zip(alphas, betas, deltas, F0s, omega_exts):
        u_roots = solve_forced_closed_form(a, b, d, F0, we, n)
        if len(u_roots) == 0:
            n_no_real_root += 1
            continue
        if len(u_roots) > 1:
            n_multivalued_excluded += 1
            if single_branch_only:
                continue
            u_sel = u_roots[-1]  # rama de mayor amplitud, si se decide conservar
        else:
            u_sel = u_roots[0]
        A = np.sqrt(u_sel)
        rows.append({"alpha": a, "beta": b, "delta": d, "F0": F0,
                      "omega_ext": we, "A": A, "n_branches": len(u_roots)})

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "beta", "delta", "F0",
                                                  "omega_ext", "A", "n_branches"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[n={n}, forma cerrada] {len(rows)} muestras validas, "
          f"{n_no_real_root} sin raiz real positiva, "
          f"{n_multivalued_excluded} con multiples ramas "
          f"({'excluidas' if single_branch_only else 'incluidas, rama mayor'}) "
          f"-> {filename}")
    return rows


def generate_forced_quartic_dataset(filename="quartic_forced_dataset.csv"):
    rng = np.random.default_rng(SEED + 4)
    alphas, betas, deltas, F0s, omega_exts = _sample_forced_params(rng, N_SAMPLES)

    rows = []
    n_failed = 0
    prev_guess = None

    t0 = time.time()
    for a, b, d, F0, we in zip(alphas, betas, deltas, F0s, omega_exts):
        try:
            hb = solve_forced_hb(a, b, d, F0, we, n=4,
                                   n_harmonics=N_HARMONICS_QUARTIC,
                                   initial_guess=prev_guess)
            prev_guess = hb["params"]
        except RuntimeError:
            try:
                hb = solve_forced_hb(a, b, d, F0, we, n=4,
                                       n_harmonics=N_HARMONICS_QUARTIC)
                prev_guess = hb["params"]
            except RuntimeError:
                n_failed += 1
                prev_guess = None
                continue

        rows.append({"alpha": a, "beta": b, "delta": d, "F0": F0,
                      "omega_ext": we, "A": hb["A"],
                      "max_residual": hb["max_residual"]})

    elapsed = time.time() - t0
    print(f"[n=4, balance multi-armonico] {len(rows)} muestras exitosas, "
          f"{n_failed} fallidas ({elapsed:.1f} s) -> {filename}")

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "beta", "delta", "F0",
                                                  "omega_ext", "A", "max_residual"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    generate_forced_closed_form_dataset(3, "cubic_forced_dataset.csv")
    generate_forced_quartic_dataset("quartic_forced_dataset.csv")
    generate_forced_closed_form_dataset(5, "quintic_forced_dataset.csv")
