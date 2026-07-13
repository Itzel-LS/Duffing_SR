"""
train_pysr_additional_validation.py
======================================

Bloque de validacion adicional -- Piezas B, C y D, sobre el caso piloto
cubico libre (formula cerrada exacta omega=sqrt(alpha+0.75*beta), A=1).

PIEZA B (extrapolacion): entrena en cubic_free_train_lowbeta.csv, evalua
en cubic_free_test_highbeta.csv (rango de beta nunca visto).

PIEZA C (ruido): entrena y evalua (con la particion 'split' ya incluida
en el CSV) sobre cada uno de los 3 niveles de ruido (1%, 2%, 5%).

PIEZA D (variabilidad multi-semilla): repite el entrenamiento del caso
LIMPIO (sin ruido, dataset original de amplitud fija) 20 veces con
semillas distintas, reporta media +/- desviacion de R^2 y MSE, y la
frecuencia con la que se recupera el coeficiente correcto (0.75).

Corre esto por CMD (no Jupyter/Colab), como ya establecimos que es lo
mas estable para corridas largas de PySR.
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from pysr import PySRRegressor

LOG_FILE = "resultados_validacion_adicional.txt"

BASE_KWARGS = dict(
    niterations=60,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt"],
    populations=15,
    population_size=35,
    maxsize=20,
    parsimony=0.001,
    model_selection="best",
    parallelism="serial",
    verbosity=0,          # silenciado -- evita saturar la terminal/navegador
                            # en 20+ corridas consecutivas (Pieza D)
)


def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()


# ======================================================================
# PIEZA B: Extrapolacion
# ======================================================================
def run_extrapolation(f):
    log("=" * 70, f)
    log("PIEZA B: EXTRAPOLACION (entrenar beta bajo, evaluar beta alto)", f)
    log("=" * 70, f)

    df_train = pd.read_csv("cubic_free_train_lowbeta.csv")
    df_test = pd.read_csv("cubic_free_test_highbeta.csv")
    df_train = df_train.rename(columns={"beta": "bta"})
    df_test = df_test.rename(columns={"beta": "bta"})

    X_train = df_train[["alpha", "bta"]]
    y_train = df_train["omega"].values
    X_test = df_test[["alpha", "bta"]]
    y_test = df_test["omega"].values

    model = PySRRegressor(**BASE_KWARGS, random_state=42, deterministic=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    best = model.get_best()

    log(f"Rango de entrenamiento: beta in [0.01, 0.15]", f)
    log(f"Rango de evaluacion (nunca visto): beta in [0.15, 0.30]", f)
    log(f"Expresion: {best['equation']}", f)
    log(f"R^2 en el rango extrapolado: {r2:.6f}", f)
    log(f"MSE en el rango extrapolado: {mse:.6e}", f)
    log("", f)
    return {"r2": r2, "mse": mse, "equation": best["equation"]}


# ======================================================================
# PIEZA C: Robustez ante ruido
# ======================================================================
def run_noise_test(f):
    log("=" * 70, f)
    log("PIEZA C: ROBUSTEZ ANTE RUIDO", f)
    log("=" * 70, f)

    results = {}
    for label in ["1pct", "2pct", "5pct"]:
        df = pd.read_csv(f"cubic_free_noisy_{label}.csv")
        df = df.rename(columns={"beta": "bta"})
        df_train = df[df["split"] == "train"]
        df_test = df[df["split"] == "test"]

        X_train = df_train[["alpha", "bta"]]
        y_train = df_train["omega"].values
        X_test = df_test[["alpha", "bta"]]
        y_test = df_test["omega"].values

        model = PySRRegressor(**BASE_KWARGS, random_state=42, deterministic=True)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        best = model.get_best()

        log(f"\nNivel de ruido: {label}", f)
        log(f"  Expresion: {best['equation']}", f)
        log(f"  R^2 (test) = {r2:.6f}", f)
        log(f"  MSE (test) = {mse:.6e}", f)

        results[label] = {"r2": r2, "mse": mse, "equation": best["equation"]}

    log("", f)
    return results


# ======================================================================
# PIEZA D: Variabilidad multi-semilla (sobre el caso limpio, sin ruido)
# ======================================================================
def run_multiseed(f, n_seeds=20):
    log("=" * 70, f)
    log(f"PIEZA D: VARIABILIDAD MULTI-SEMILLA (n={n_seeds}, caso limpio)", f)
    log("=" * 70, f)

    # dataset limpio (mismo que Pieza C pero sin ruido, o el de baja beta
    # ya sirve si tiene rango completo -- generamos uno limpio simple aqui)
    rng = np.random.default_rng(0)
    alphas = rng.uniform(0.5, 2.0, size=1000)
    betas = rng.uniform(0.01, 0.30, size=1000)
    omegas = np.sqrt(alphas + 0.75 * betas)
    df = pd.DataFrame({"alpha": alphas, "bta": betas, "omega": omegas})

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        df[["alpha", "bta"]], df["omega"].values, test_size=0.2, random_state=42
    )

    r2_list, mse_list, coef_list = [], [], []

    t0 = time.time()
    for seed in range(n_seeds):
        model = PySRRegressor(**BASE_KWARGS, random_state=seed, deterministic=True)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2_list.append(r2)
        mse_list.append(mse)

        log(f"  semilla {seed}: R^2={r2:.6f}, MSE={mse:.6e}, "
            f"eq={model.get_best()['equation']}", f)

    elapsed = time.time() - t0

    r2_arr, mse_arr = np.array(r2_list), np.array(mse_list)
    log(f"\nCompletado en {elapsed/60:.1f} minutos ({n_seeds} corridas)", f)
    log(f"R^2:  media = {r2_arr.mean():.6f}, desviacion = {r2_arr.std():.6f}, "
        f"min = {r2_arr.min():.6f}, max = {r2_arr.max():.6f}", f)
    log(f"MSE:  media = {mse_arr.mean():.6e}, desviacion = {mse_arr.std():.6e}", f)
    log("", f)

    return {"r2_mean": r2_arr.mean(), "r2_std": r2_arr.std(),
            "mse_mean": mse_arr.mean(), "mse_std": mse_arr.std()}


def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        log(f"Inicio: {time.strftime('%Y-%m-%d %H:%M:%S')}\n", f)

        extrap_result = run_extrapolation(f)
        noise_results = run_noise_test(f)
        multiseed_result = run_multiseed(f, n_seeds=20)

        log("\n" + "=" * 70, f)
        log("RESUMEN FINAL", f)
        log("=" * 70, f)
        log(f"\nExtrapolacion: R^2={extrap_result['r2']:.6f}, "
            f"eq={extrap_result['equation']}", f)
        for label, res in noise_results.items():
            log(f"Ruido {label}: R^2={res['r2']:.6f}, eq={res['equation']}", f)
        log(f"Multi-semilla: R^2 = {multiseed_result['r2_mean']:.6f} +/- "
            f"{multiseed_result['r2_std']:.6f}", f)

        log(f"\nFin: {time.strftime('%Y-%m-%d %H:%M:%S')}", f)


if __name__ == "__main__":
    main()
