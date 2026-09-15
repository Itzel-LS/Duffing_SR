"""
Implementacion faltante en pysr_reduced_forced.py: balance multiarmonico
cuartico FORZADO, en variables reducidas (Omega, zeta, f).

pysr_reduced_forced.py ya resuelve n=3 y n=5 (formula cerrada de un armonico)
y ya muestrea (Omega, zeta, f) uniformemente y de forma directa, como pide la
NOTA del articulo. Lo unico que faltaba era n=4, porque el termino cuartico
NO tiene proyeccion sobre el armonico fundamental (igual que en el caso libre,
Sec. II B 2), asi que hace falta un ansatz multiarmonico completo -- y, a
diferencia del caso libre, con senos ademas de cosenos, porque la respuesta
forzada tiene un desfase respecto al forzamiento que la simetria del caso
libre no tiene.

Ecuacion reducida (Ec. 8 del articulo):
    X'' + 2 zeta X' + X + X^4 = f cos(Omega s)

Ansatz (theta = Omega s):
    X(theta) = c0 + sum_{k=1}^{M} [ a_k cos(k theta) + b_k sin(k theta) ]

La amplitud de respuesta se define IGUAL que en la Sec. II E para el caso
quintico numerico (proyeccion sobre el armonico de excitacion):
    A = sqrt(a1^2 + b1^2)
para mantener una unica definicion de "amplitud de respuesta" en todo el
articulo. Si ya tienes tu propio codigo del balance multiarmonico cuartico
forzado (el que genero la fila cuartica de la Tabla IV / Fig. 5b originales),
compara los valores de A en un par de puntos de control antes de confiar en
esta version -- no reemplaza a tu codigo, es una reconstruccion pensada para
ser consistente con el resto del articulo.

Deteccion de multirraiz / no-convergencia: se resuelve el sistema desde
varias condiciones iniciales cualitativamente distintas (respuesta lineal en
fase, respuesta en cuadratura cerca de resonancia, rama de amplitud grande) y
se agrupan las soluciones convergidas por su valor de A. 0 soluciones ->
'sin raiz'; >1 solucion distinta -> 'multirraiz' (se descarta, igual que en
build_dataset() para n=3,5); exactamente 1 -> se conserva.

Uso:
    python3 quartic_reduced_forced.py            # N=1000, protocolo
    python3 quartic_reduced_forced.py --n 40      # prueba rapida
"""
import argparse
import time
import numpy as np
from scipy.optimize import fsolve

M = 4          # armonicos (c0 + M cosenos + M senos = 2M+1 incognitas)
NGRID = 2048
OMEGA_RANGE = (0.60, 1.60)
ZETA_RANGE = (0.005, 0.150)
F_RANGE = (0.02, 1.00)


def residuals(u, Omega, zeta, f, M=M, ngrid=NGRID):
    c0 = u[0]
    a = u[1:1 + M]
    b = u[1 + M:1 + 2 * M]
    th = np.arange(ngrid) * 2 * np.pi / ngrid
    k = np.arange(1, M + 1)
    cos_kt = np.cos(np.outer(k, th))
    sin_kt = np.sin(np.outer(k, th))
    X = c0 + a @ cos_kt + b @ sin_kt
    Xpp = -(Omega ** 2) * ((k ** 2 * a) @ cos_kt + (k ** 2 * b) @ sin_kt)
    Xp = Omega * ((-k * a) @ sin_kt + (k * b) @ cos_kt)
    R = Xpp + 2 * zeta * Xp + X + X ** 4 - f * np.cos(th)
    r0 = R.mean()
    rc = 2.0 * (R @ cos_kt.T) / ngrid
    rs = 2.0 * (R @ sin_kt.T) / ngrid
    return np.concatenate(([r0], rc, rs))


def initial_guesses(Omega, zeta, f, M=M):
    """Semillas cualitativamente distintas para detectar ramas multiples."""
    guesses = []
    denom = 1.0 - Omega ** 2
    # 1) respuesta lineal, en fase o antifase segun el signo del denominador
    u = np.zeros(1 + 2 * M)
    if abs(denom) > 1e-3:
        u[1] = f / denom
    else:
        u[1 + M] = f / (2 * zeta * Omega + 1e-6)
    guesses.append(u.copy())
    # 2) respuesta en cuadratura (cerca de resonancia)
    u2 = np.zeros(1 + 2 * M)
    u2[1 + M] = -f / (2 * zeta * Omega + 1e-6)
    guesses.append(u2)
    # 3) rama de amplitud grande, con offset negativo (favorecido por x^4 asimetrico)
    u3 = np.zeros(1 + 2 * M)
    u3[0] = -0.3
    u3[1] = 1.5
    guesses.append(u3)
    # 4) rama de amplitud pequena
    u4 = np.zeros(1 + 2 * M)
    u4[1] = 0.2 * np.sign(f)
    guesses.append(u4)
    return guesses


def solve_quartic_forced(Omega, zeta, f, M=M, tol=1e-8):
    sols = []
    for u0 in initial_guesses(Omega, zeta, f, M):
        sol, info, ier, msg = fsolve(residuals, u0, args=(Omega, zeta, f, M),
                                      full_output=True, maxfev=3000)
        if ier != 1:
            continue
        resid_norm = np.max(np.abs(info["fvec"]))
        if resid_norm > tol:
            continue
        a1, b1 = sol[1], sol[1 + M]
        A = np.hypot(a1, b1)
        if not np.isfinite(A) or A > 50:
            continue
        sols.append(A)
    if not sols:
        return []
    sols = np.array(sorted(sols))
    # agrupa soluciones que coinciden dentro de 1e-3 en A (misma rama)
    clusters = [sols[0]]
    for s in sols[1:]:
        if s - clusters[-1] > 1e-3:
            clusters.append(s)
    return clusters


def build_dataset(n=4000_000, N=1000, seed=42, verbose_every=100):
    # (n se ignora; la firma se deja compatible con build_dataset(n, ...)
    # de pysr_reduced_forced.py, que llama con n=4)
    rng = np.random.default_rng(seed)
    Om = rng.uniform(*OMEGA_RANGE, N)
    ze = rng.uniform(*ZETA_RANGE, N)
    ff = rng.uniform(*F_RANGE, N)
    X, y = [], []
    n_multi = n_fail = 0
    t0 = time.time()
    for i, (O, z, f) in enumerate(zip(Om, ze, ff)):
        branches = solve_quartic_forced(O, z, f)
        if len(branches) == 0:
            n_fail += 1
        elif len(branches) > 1:
            n_multi += 1
        else:
            X.append([O, z, f])
            y.append(branches[0])
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"  [{i+1}/{N}] elapsed={time.time()-t0:.1f}s "
                  f"retenidas={len(y)} multirraiz={n_multi} sin_raiz={n_fail}")
    print(f"n=4: {len(y)} muestras retenidas de {N} "
          f"({n_multi} multirraiz, {n_fail} sin raiz)")
    return np.array(X), np.array(y), n_multi, n_fail


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()

    print("=== Punto de control (Omega=1.10, zeta=0.05, f=0.30) ===")
    br = solve_quartic_forced(1.10, 0.05, 0.30)
    print(f"ramas encontradas: {br}\n")

    print(f"=== Generando dataset cuartico forzado reducido, N={args.n} ===")
    X, y, n_multi, n_fail = build_dataset(N=args.n)
    out = f"{args.outdir}/reduced_forced_n4.npz"
    np.savez(out, X=X, y=y, columns=np.array(["Omega", "zeta", "f"]),
             n_multi=n_multi, n_fail=n_fail, n_requested=args.n)
    print(f"-> guardado {out}  X={X.shape}  y={y.shape}")

    print("""
Siguiente paso (en tu pysr_env, no aqui) -- igual que para n=3 y n=5:

from pysr import PySRRegressor
d = np.load("reduced_forced_n4.npz")
X, y = d["X"], d["y"]
model = PySRRegressor(
    niterations=200,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt", "cbrt"],
    maxsize=30,
    model_selection="best",
    random_state=42,
    deterministic=True,
    parallelism="serial",
    procs=0,
)
model.fit(X, y, variable_names=["Omega", "zeta", "f"])
print(model.get_best())

Repetir con 20 semillas para n=3, 4 y 5, y reportar R2/MSE media +- desviacion,
igual que el Apendice A -- eso es lo que llena la seccion "Forced oscillators
in reduced variables" (III C) completa.
""")
