"""
train_pysr_amplitude.py
=========================

Bloque B - entrenamiento de PySR sobre los tres datasets con amplitud
variable generados por generate_amplitude_datasets.py.

Objetivo (Rev1 #2, Rev4 #2): verificar si SR recupera explicitamente la
dependencia A^2 en el caso cubico y A^4 en el caso quintico, ahora que A
es una variable de entrada mas (junto con alpha y beta) en vez de estar
fijada en 1.

Requiere PySR instalado (con su backend de Julia) en tu entorno -- este
script NO se puede correr en el entorno donde te lo estoy generando, solo
en el tuyo.

Uso
---
    python train_pysr_amplitude.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from pysr import PySRRegressor

# ======================================================================
# CONFIGURACION -- ajusta aqui si tu protocolo original usaba otros
# valores (semillas, iteraciones, poblacion, etc.) para mantener
# consistencia con el resto del estudio.
# ======================================================================
SEED = 42
TEST_SIZE = 0.20

# Mismo operador restringido que el resto del estudio (Sec. II.C del
# manuscrito): {+, -, *, /, sqrt}, sin trigonometricas ni exponenciales.
PYSR_KWARGS = dict(
    niterations=80,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt"],
    populations=20,
    population_size=40,
    maxsize=25,
    parsimony=0.001,           # penalizacion de complejidad (ajustable)
    model_selection="best",     # criterio de Pareto: mejor accuracy/complejidad
    random_state=SEED,
    deterministic=True,
    parallelism="serial",       # requerido junto con deterministic=True para reproducibilidad
    verbosity=1,
)

DATASETS = {
    "cubic": {
        "csv": "cubic_amplitude_dataset.csv",
        "expected_scaling": "A^2 (LP: omega = sqrt(alpha + 0.75*beta*A^2))",
    },
    "quartic": {
        "csv": "quartic_amplitude_dataset.csv",
        "expected_scaling": "sin forma cerrada -- referencia = balance armonico numerico",
    },
    "quintic": {
        "csv": "quintic_amplitude_dataset.csv",
        "expected_scaling": "A^4 (HAM-P: omega = sqrt(alpha + 0.625*beta*A^4))",
    },
}


def train_one_case(name, csv_path, expected_scaling):
    print("=" * 70)
    print(f"CASO: {name.upper()}")
    print(f"Dependencia esperada: {expected_scaling}")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    # NOTA: 'beta' choca con sympy.beta() (funcion Beta), y 'gamma' con
    # sympy.gamma() (funcion Gamma) -- PySR usa sympy internamente para
    # construir las expresiones, asi que renombramos aqui SOLO para el
    # entrenamiento. En la ecuacion resultante, 'bta' = beta del manuscrito.
    df = df.rename(columns={"beta": "bta"})
    X = df[["alpha", "bta", "A"]]  # DataFrame -> PySR toma los nombres de columna
    y = df["omega"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    model = PySRRegressor(**PYSR_KWARGS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    best = model.get_best()  # pandas.Series con la fila seleccionada por Pareto
    best_equation_str = best["equation"]
    best_complexity = best["complexity"]

    print("\nMejor expresion encontrada (segun criterio de Pareto):")
    print(f"  {best_equation_str}")
    print(f"  (complejidad = {best_complexity})")
    print(f"  R^2 (test) = {r2:.6f}")
    print(f"  MSE (test) = {mse:.6e}")
    print("\nFrente de Pareto completo (complejidad vs. precision):")
    print(model.equations_[["complexity", "loss", "equation"]].to_string(index=False))
    print()

    return {
        "name": name,
        "model": model,
        "best_equation": best_equation_str,
        "r2": r2,
        "mse": mse,
        "pareto_front": model.equations_,
    }


if __name__ == "__main__":
    results = {}
    for name, cfg in DATASETS.items():
        results[name] = train_one_case(name, cfg["csv"], cfg["expected_scaling"])

    print("\n" + "=" * 70)
    print("RESUMEN FINAL -- copia esta tabla para compartirla")
    print("(recordatorio: 'bta' en las ecuaciones = 'beta' del manuscrito,")
    print(" renombrado solo para evitar el conflicto con sympy.beta())")
    print("=" * 70)
    for name, res in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Expresion : {res['best_equation']}")
        print(f"  R^2       : {res['r2']:.6f}")
        print(f"  MSE       : {res['mse']:.6e}")
