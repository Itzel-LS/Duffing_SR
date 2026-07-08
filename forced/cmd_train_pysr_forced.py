"""
cmd_train_pysr_forced.py
==========================

Version standalone para correr desde CMD (no notebook), evitando por
completo cualquier problema de renderizado de navegador/Jupyter.

Uso desde CMD (con el entorno conda activado, p. ej. pysr_env):

    cd C:\\ruta\\donde\\estan\\tus\\csv
    conda activate pysr_env
    python cmd_train_pysr_forced.py

Los 3 CSV (cubic_forced_dataset.csv, quartic_forced_dataset.csv,
quintic_forced_dataset.csv) deben estar en la misma carpeta.

Los resultados se imprimen en la terminal Y se guardan en
resultados_bloqueC.txt, para no perderlos si la ventana se cierra por error.
"""

import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from pysr import PySRRegressor

SEED = 42
TEST_SIZE = 0.20
LOG_FILE = "resultados_bloqueC.txt"

PYSR_KWARGS = dict(
    niterations=200,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt", "cbrt"],
    populations=30,
    population_size=50,
    maxsize=30,
    parsimony=0.0008,
    model_selection="best",
    random_state=SEED,
    deterministic=True,
    parallelism="serial",
    verbosity=1,          # en CMD esto es seguro, no hay navegador que se sature
)

DATASETS = {
    "cubic_forced": "cubic_forced_dataset.csv",
    "quartic_forced": "quartic_forced_dataset.csv",
    "quintic_forced": "quintic_forced_dataset.csv",
}
COLS = ["alpha", "beta", "delta", "F0", "omega_ext"]


def log(msg, f):
    """Imprime en pantalla Y escribe a archivo, con flush inmediato."""
    print(msg)
    f.write(msg + "\n")
    f.flush()


def train_one_case(name, csv_path, cols, f):
    log("=" * 70, f)
    log(f"CASO: {name.upper()}  (inicio: {time.strftime('%H:%M:%S')})", f)
    log("=" * 70, f)

    df = pd.read_csv(csv_path)
    df = df.rename(columns={"beta": "bta"})
    cols_renamed = [c if c != "beta" else "bta" for c in cols]

    X = df[cols_renamed]
    y = df["A"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    t0 = time.time()
    model = PySRRegressor(**PYSR_KWARGS)
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    best = model.get_best()

    log(f"\nTerminado en {elapsed/60:.1f} minutos ({time.strftime('%H:%M:%S')})", f)
    log("Mejor expresion encontrada:", f)
    log(f"  {best['equation']}", f)
    log(f"  (complejidad = {best['complexity']})", f)
    log(f"  R^2 (test) = {r2:.6f}", f)
    log(f"  MSE (test) = {mse:.6e}", f)
    log("\nFrente de Pareto completo:", f)
    log(model.equations_[["complexity", "loss", "equation"]].to_string(index=False), f)
    log("", f)

    return {"name": name, "equation": best["equation"], "r2": r2, "mse": mse}


def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        log(f"Inicio de ejecucion: {time.strftime('%Y-%m-%d %H:%M:%S')}", f)
        log(f"(resultados tambien se guardan en {LOG_FILE})\n", f)

        results = {}
        for name, csv_path in DATASETS.items():
            results[name] = train_one_case(name, csv_path, COLS, f)

        log("\n" + "=" * 70, f)
        log("RESUMEN FINAL", f)
        log("(recordatorio: 'bta' en las ecuaciones = 'beta' del manuscrito)", f)
        log("=" * 70, f)
        for res in results.values():
            log(f"\n{res['name'].upper()}:", f)
            log(f"  Expresion : {res['equation']}", f)
            log(f"  R^2       : {res['r2']:.6f}", f)
            log(f"  MSE       : {res['mse']:.6e}", f)

        log(f"\nFin de ejecucion: {time.strftime('%Y-%m-%d %H:%M:%S')}", f)


if __name__ == "__main__":
    main()
