"""
generate_validation_datasets.py
==================================

Bloque de validacion adicional -- Piezas B (extrapolacion) y C (ruido),
sobre el caso piloto: cubico libre, A=1 fijo (formula cerrada exacta
omega = sqrt(alpha + 0.75*beta), para poder comparar contra la verdad
conocida en cada prueba).

Genera:

  Extrapolacion (Pieza B):
    cubic_free_train_lowbeta.csv   -- beta in [0.01, 0.15] (entrenamiento)
    cubic_free_test_highbeta.csv   -- beta in [0.15, 0.30] (evaluacion,
                                        rango nunca visto en entrenamiento)

  Ruido (Pieza C):
    cubic_free_noisy_1pct.csv      -- ruido relativo gaussiano del 1%
    cubic_free_noisy_2pct.csv      -- ruido relativo gaussiano del 2%
    cubic_free_noisy_5pct.csv      -- ruido relativo gaussiano del 5%
    (cada uno con particion train/test 80/20 ya aplicada, columna 'split')
"""

import csv

import numpy as np

SEED = 42
N_SAMPLES = 1000
ALPHA_RANGE = (0.5, 2.0)
A_FIXED = 1.0  # misma normalizacion que el resto del caso libre A=1


def omega_true(alpha, beta):
    """Formula cerrada exacta LP, A=1: omega = sqrt(alpha + 0.75*beta)."""
    return np.sqrt(alpha + 0.75 * beta)


def save_csv(filename, alphas, betas, omegas, extra_col=None, extra_name=None):
    fieldnames = ["alpha", "beta", "omega"]
    if extra_col is not None:
        fieldnames.append(extra_name)
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(alphas)):
            row = {"alpha": alphas[i], "beta": betas[i], "omega": omegas[i]}
            if extra_col is not None:
                row[extra_name] = extra_col[i]
            writer.writerow(row)
    print(f"  {len(alphas)} muestras -> {filename}")


# ======================================================================
# PIEZA B: Extrapolacion -- entrenar en beta bajo, evaluar en beta alto
# ======================================================================
def generate_extrapolation_datasets():
    print("=== Pieza B: datasets de extrapolacion ===")
    rng = np.random.default_rng(SEED)

    # Entrenamiento: beta en la mitad INFERIOR del rango original [0.01, 0.30]
    alphas_train = rng.uniform(*ALPHA_RANGE, size=N_SAMPLES)
    betas_train = rng.uniform(0.01, 0.15, size=N_SAMPLES)
    omegas_train = omega_true(alphas_train, betas_train)
    save_csv("cubic_free_train_lowbeta.csv", alphas_train, betas_train, omegas_train)

    # Evaluacion: beta en la mitad SUPERIOR, nunca vista en entrenamiento
    rng2 = np.random.default_rng(SEED + 1)
    alphas_test = rng2.uniform(*ALPHA_RANGE, size=int(N_SAMPLES * 0.25))
    betas_test = rng2.uniform(0.15, 0.30, size=int(N_SAMPLES * 0.25))
    omegas_test = omega_true(alphas_test, betas_test)
    save_csv("cubic_free_test_highbeta.csv", alphas_test, betas_test, omegas_test)
    print()


# ======================================================================
# PIEZA C: Robustez ante ruido -- perturbar omega con ruido relativo
# ======================================================================
def generate_noise_datasets():
    print("=== Pieza C: datasets con ruido ===")
    rng = np.random.default_rng(SEED)
    alphas = rng.uniform(*ALPHA_RANGE, size=N_SAMPLES)
    betas = rng.uniform(0.01, 0.30, size=N_SAMPLES)
    omega_clean = omega_true(alphas, betas)

    # particion train/test 80/20 (mismos indices para las 3 versiones,
    # para que sean directamente comparables)
    rng_split = np.random.default_rng(SEED)
    idx = rng_split.permutation(N_SAMPLES)
    n_train = int(0.8 * N_SAMPLES)
    split = np.array(["train"] * N_SAMPLES)
    split[idx[n_train:]] = "test"

    for noise_level, label in [(0.01, "1pct"), (0.02, "2pct"), (0.05, "5pct")]:
        rng_noise = np.random.default_rng(SEED + hash(label) % 1000)
        noise = rng_noise.normal(0, noise_level, size=N_SAMPLES) * omega_clean
        omega_noisy = omega_clean + noise
        save_csv(f"cubic_free_noisy_{label}.csv", alphas, betas, omega_noisy,
                  extra_col=split, extra_name="split")
    print()


if __name__ == "__main__":
    generate_extrapolation_datasets()
    generate_noise_datasets()
    print("Listo. Verifica los CSV generados antes de entrenar PySR sobre ellos.")
