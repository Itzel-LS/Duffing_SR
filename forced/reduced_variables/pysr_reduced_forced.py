"""
Generación de los conjuntos forzados en variables reducidas y corrida de PySR.

Responde al punto B3 de la segunda ronda.  El cambio de variable
    x = (alpha/beta)^{1/(n-1)} X ,   t = s/sqrt(alpha)
lleva  x'' + delta x' + alpha x + beta x^n = F0 cos(omega_ext t)  a

    X'' + 2 zeta X' + X + X^n = f cos(Omega s)

con  Omega = omega_ext/sqrt(alpha),  zeta = delta/(2 sqrt(alpha)),
     f     = F0 beta^{1/(n-1)} / alpha^{n/(n-1)} ,
y la amplitud física se recupera como A = (alpha/beta)^{1/(n-1)} * Acal.

La búsqueda simbólica se corre sobre Acal(Omega, zeta, f): tres entradas en vez
de cinco.  Muestreamos uniformemente en los tres grupos, en lugar de heredar el
sesgo que induce el muestreo uniforme en los parámetros crudos.

Uso:
    python3 pysr_reduced_forced.py 3      # cúbico
    python3 pysr_reduced_forced.py 4      # cuártico
    python3 pysr_reduced_forced.py 5      # quíntico
"""
import sys
import numpy as np
from scipy.optimize import brentq

N_SAMPLES = 1000
SEED = 42

# Rangos de los grupos reducidos.  Cubren el interior de la caja cruda usada en
# el artículo sin reproducir su sesgo; ajustar si se prefiere otro criterio.
OMEGA_RANGE = (0.60, 1.60)
ZETA_RANGE = (0.005, 0.150)
F_RANGE = (0.02, 1.00)


def hb_roots(Omega, zeta, f, n):
    """Amplitud de respuesta por balance de primer armónico en variables reducidas.

    Para n impar el término no lineal aporta  kappa_n * Acal^{n-1}  a la rigidez
    efectiva:  kappa_3 = 3/4,  kappa_5 = 5/8.  La relación de respuesta es

        [(1 - Omega^2 + kappa_n Acal^{n-1})^2 + (2 zeta Omega)^2] Acal^2 = f^2.

    Para n par (n=4) el primer armónico no recibe contribución del término no
    lineal y hay que usar balance multiarmónico; ver quartic_multiharmonic().
    """
    kappa = {3: 0.75, 5: 0.625}[n]

    def g(A):
        return ((1 - Omega ** 2 + kappa * A ** (n - 1)) ** 2
                + (2 * zeta * Omega) ** 2) * A ** 2 - f ** 2

    grid = np.linspace(1e-6, 20.0, 4000)
    vals = g(grid)
    roots = []
    for i in range(len(grid) - 1):
        if vals[i] == 0.0:
            roots.append(grid[i])
        elif vals[i] * vals[i + 1] < 0:
            roots.append(brentq(g, grid[i], grid[i + 1]))
    return roots


def build_dataset(n, seed=SEED, N=N_SAMPLES):
    rng = np.random.default_rng(seed)
    Om = rng.uniform(*OMEGA_RANGE, N)
    ze = rng.uniform(*ZETA_RANGE, N)
    ff = rng.uniform(*F_RANGE, N)
    X, y, n_multi, n_fail = [], [], 0, 0
    for O, z, f in zip(Om, ze, ff):
        if n == 4:
            raise NotImplementedError(
                "El caso n=4 requiere balance multiarmónico; reutilizar la rutina "
                "del artículo, reescrita en variables reducidas, e imponer la misma "
                "definición de amplitud de suelta adoptada en la Sec. II B.")
        r = hb_roots(O, z, f, n)
        if len(r) == 0:
            n_fail += 1
            continue
        if len(r) > 1:
            n_multi += 1          # descartada, no asignada a rama
            continue
        X.append([O, z, f])
        y.append(r[0])
    print(f"n={n}: {len(y)} muestras retenidas de {N} "
          f"({n_multi} multirraíz, {n_fail} sin raíz)")
    return np.array(X), np.array(y)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    X, y = build_dataset(n)
    np.savez(f"reduced_forced_n{n}.npz", X=X, y=y,
             columns=np.array(["Omega", "zeta", "f"]))
    print(f"guardado reduced_forced_n{n}.npz  X={X.shape}  y={y.shape}")
    print()
    print("Corrida de PySR (requiere Julia instalado):")
    print("""
from pysr import PySRRegressor
d = np.load("reduced_forced_n3.npz")
X, y = d["X"], d["y"]
model = PySRRegressor(
    niterations=200,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt", "cbrt"],
    maxsize=30,
    model_selection="best",
    random_state=42,
    deterministic=True,
    procs=0,
)
model.fit(X, y, variable_names=["Omega", "zeta", "f"])
print(model)
""")
    print("Repetir con 20 semillas y reportar media y desviacion de R2 y MSE, "
          "igual que en el Apendice A.")
