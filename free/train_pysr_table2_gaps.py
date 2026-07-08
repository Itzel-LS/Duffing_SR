"""
train_pysr_table2_gaps.py
============================

Cierra los dos huecos pendientes de la Tabla II (caso libre, A=1 fijo):
  - x^4: entrena contra quartic_free_dataset.csv (generado en el Bloque A
    con quartic_harmonic_balance.py + generate_quartic_dataset.py)
  - x^5 RK45+FFT: entrena contra quintic_free_rk45_dataset.csv (generado
    con rk45_reference.py + generate_quintic_free_rk45_dataset.py, ya con
    la correccion omega=2*pi*f aplicada)

Requiere PySR instalado en tu entorno. Corre esto por CMD (no
Jupyter/Colab), como ya establecimos que es lo mas estable.
"""

import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from pysr import PySRRegressor

SEED = 42
TEST_SIZE = 0.20
LOG_FILE = "resultados_tabla2_gaps.txt"

PYSR_KWARGS = dict(
    niterations=80,              # igual que train_pysr_amplitude.py (ya
                                    # logro R^2=1.0000 exacto con 3 variables;
                                    # aqui solo hay 2, deberia bastar de sobra)
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt"],     # sin cbrt: no hay estructura tipo Cardano
                                    # en estos dos casos (libres, A=1 fijo)
    populations=20,
    population_size=40,
    maxsize=25,
    parsimony=0.001,
    model_selection="best",
    random_state=SEED,
    deterministic=True,
    parallelism="serial",
    verbosity=1,
)

DATASETS = {
    "quartic_free": {
        "csv": "quartic_free_dataset.csv",
        "cols": ["alpha", "beta"],
        "target": "omega_hb",
    },
    "quintic_free_rk45": {
        "csv": "quintic_free_rk45_dataset.csv",
        "cols": ["alpha", "beta"],
        "target": "omega",
    },
}


def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()


def train_one_case(name, csv_path, cols, target, f):
    log("=" * 70, f)
    log(f"CASO: {name.upper()}  (inicio: {time.strftime('%H:%M:%S')})", f)
    log("=" * 70, f)

    df = pd.read_csv(csv_path)
    df = df.rename(columns={"beta": "bta"})  # evitar conflicto con sympy.beta()
    cols_renamed = [c if c != "beta" else "bta" for c in cols]

    X = df[cols_renamed]
    y = df[target].values

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
    log(f"Mejor expresion: {best['equation']}", f)
    log(f"  (complejidad = {best['complexity']})", f)
    log(f"  R^2 (test) = {r2:.6f}", f)
    log(f"  MSE (test) = {mse:.6e}", f)
    log("\nFrente de Pareto completo:", f)
    log(model.equations_[["complexity", "loss", "equation"]].to_string(index=False), f)
    log("", f)

    return {"name": name, "equation": best["equation"], "r2": r2, "mse": mse}


def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        log(f"Inicio: {time.strftime('%Y-%m-%d %H:%M:%S')}\n", f)

        results = {}
        for name, cfg in DATASETS.items():
            results[name] = train_one_case(name, cfg["csv"], cfg["cols"],
                                              cfg["target"], f)

        log("\n" + "=" * 70, f)
        log("RESUMEN FINAL -- copia esta tabla para compartirla", f)
        log("(recordatorio: 'bta' en las ecuaciones = 'beta' del manuscrito)", f)
        log("=" * 70, f)
        for res in results.values():
            log(f"\n{res['name'].upper()}:", f)
            log(f"  Expresion : {res['equation']}", f)
            log(f"  R^2       : {res['r2']:.6f}", f)
            log(f"  MSE       : {res['mse']:.6e}", f)

        log(f"\nFin: {time.strftime('%Y-%m-%d %H:%M:%S')}", f)


if __name__ == "__main__":
    main()
