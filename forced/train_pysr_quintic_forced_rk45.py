"""
train_pysr_quintic_forced_rk45.py
====================================

Entrena PySR sobre el dataset RK45 forzado quintico, completando la fila
pendiente de la Tabla IV (x^5, RK45+FFT, forzado).

Mismo protocolo que el resto del Bloque C: objetivo = amplitud de
respuesta A, F0 incluido explicitamente, operadores {+,-,*,/,sqrt,cbrt}.

Corre esto por CMD (no Jupyter/Colab).
"""

import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from pysr import PySRRegressor

SEED = 42
TEST_SIZE = 0.20
LOG_FILE = "resultados_quintic_forced_rk45.txt"

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
    verbosity=1,          # seguro en terminal CMD
)


def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()


def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        log(f"Inicio: {time.strftime('%Y-%m-%d %H:%M:%S')}\n", f)

        df = pd.read_csv("quintic_forced_rk45_dataset.csv")
        df = df.rename(columns={"beta": "bta"})  # evitar conflicto con sympy.beta()

        X = df[["alpha", "bta", "delta", "F0", "omega_ext"]]
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

        log(f"\nTerminado en {elapsed/60:.1f} minutos", f)
        log("Mejor expresion encontrada:", f)
        log(f"  {best['equation']}", f)
        log(f"  (complejidad = {best['complexity']})", f)
        log(f"  R^2 (test) = {r2:.6f}", f)
        log(f"  MSE (test) = {mse:.6e}", f)
        log("\nFrente de Pareto completo:", f)
        log(model.equations_[["complexity", "loss", "equation"]].to_string(index=False), f)

        log(f"\nFin: {time.strftime('%Y-%m-%d %H:%M:%S')}", f)
        log("\n(recordatorio: 'bta' en la ecuacion = 'beta' del manuscrito)", f)


if __name__ == "__main__":
    main()
