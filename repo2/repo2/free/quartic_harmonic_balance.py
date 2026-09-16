"""
Regeneracion completa del caso cuartico libre bajo la convencion de amplitud
consistente (Referee A, punto A1, ronda 2).

Reutiliza sin cambios la logica de balance armonico de `quartic_amplitude.py`
(M=5, funcion hb_fixed_release), y la aplica a los DOS datasets que el
manuscrito necesita:

  1. Dataset "amplitud fija" (A = 1, Tabla II / Fig. 2b / Fig. 3b):
         alpha ~ U(0.5, 2.0),  beta ~ U(0.01, 0.3),   N = 1000
     igual que el resto de los datasets libres del articulo.

  2. Dataset "amplitud variable" (Tabla III / Fig. 4b / Fig. 6, fila cuartica):
         alpha ~ U(0.5, 2.0),  beta ~ U(0.01, 0.3),
         A     ~ grid {0.5, 1.0, 1.5, 2.0} (como en el articulo),
         N = 1000 combinaciones antes del filtro de escape.

En ambos casos se aplica el filtro fisico V(A) < V(x_c) con la definicion de
amplitud de punto de suelta (x(0) = A, x'(0) = 0), y se reporta cuantas
muestras sobreviven -- ese es el numero que reemplaza al "625" viejo en la
Tabla III y su nota al pie.

Salida: dos archivos .npz listos para cargar con PySR
    quartic_free_fixedA.npz    columnas: alpha, beta            -> omega
    quartic_free_variableA.npz columnas: alpha, beta, A         -> omega

Uso:
    python3 quartic_free_regen.py            # corre con N=1000 (protocolo)
    python3 quartic_free_regen.py --n 50     # corrida rapida de prueba
"""
import argparse
import time
import numpy as np
from scipy.optimize import fsolve, brentq

M = 5  # armonicos, igual que en el manuscrito y en quartic_amplitude.py


def hb_residual_projections(c, omega, alpha, beta, M, ngrid=2048):
    th = np.arange(ngrid) * 2 * np.pi / ngrid
    k = np.arange(M + 1)
    C = np.cos(np.outer(k, th))
    x = c @ C
    xdd = -(omega ** 2) * ((k ** 2 * c) @ C)
    R = xdd + alpha * x + beta * x ** 4
    proj = np.empty(M + 1)
    proj[0] = R.mean()
    for kk in range(1, M + 1):
        proj[kk] = 2.0 * (R * np.cos(kk * th)).mean()
    return proj


def hb_fixed_release(alpha, beta, A_rel, M=M, c_guess=None):
    """Convencion consistente: x(0) = sum_k c_k = A_rel (punto de suelta)."""
    def eqs(u):
        omega, c = u[0], u[1:]
        r = hb_residual_projections(c, omega, alpha, beta, M)
        return np.concatenate((r, [c.sum() - A_rel]))

    if c_guess is None:
        c0 = np.zeros(M + 1)
        c0[1] = A_rel
        u0 = np.concatenate(([np.sqrt(alpha)], c0))
    else:
        u0 = c_guess
    sol, info, ier, msg = fsolve(eqs, u0, full_output=True, maxfev=2000)
    return sol[0], sol[1:], ier


def escape_amplitude(alpha, beta):
    """Maxima amplitud de suelta con orbita acotada: V(A) < V(x_c)."""
    V = lambda x: 0.5 * alpha * x ** 2 + beta * x ** 5 / 5.0
    xc = -(alpha / beta) ** (1.0 / 3.0)
    Vc = V(xc)
    try:
        return brentq(lambda a: V(a) - Vc, 1e-6, 50.0)
    except ValueError:
        return np.inf


def build_fixed_A_dataset(n, seed=0, A=1.0, verbose_every=200):
    """Dataset principal (A=1), igual estructura que los demas casos libres."""
    rng = np.random.default_rng(seed)
    alphas = rng.uniform(0.5, 2.0, n)
    betas = rng.uniform(0.01, 0.3, n)

    out_a, out_b, out_w = [], [], []
    n_excluded = 0
    t0 = time.time()
    for i, (a, b) in enumerate(zip(alphas, betas)):
        Aesc = escape_amplitude(a, b)
        if A >= Aesc:
            n_excluded += 1
            continue
        # SIEMPRE arranque en frio (sin "warm start" de la muestra anterior).
        # El warm start entre muestras aleatorias consecutivas podia llevar a
        # fsolve a una rama espuria de media frecuencia (bug detectado al
        # revisar los resultados de PySR: ~12% de las muestras del dataset
        # original tenian omega ~ sqrt(alpha)/2, una solucion matematicamente
        # valida del sistema pero fisicamente incorrecta).
        omega, c, ier = hb_fixed_release(a, b, A, c_guess=None)
        if ier != 1 or not np.isfinite(omega):
            n_excluded += 1
            continue
        # filtro de sanidad fisica: la correccion cuartica es una perturbacion
        # del oscilador lineal; una desviacion grande respecto a sqrt(alpha)
        # es indicio de rama espuria, no de una solucion fisica valida.
        if abs(omega - np.sqrt(a)) > 0.3 * np.sqrt(a):
            n_excluded += 1
            continue
        out_a.append(a)
        out_b.append(b)
        out_w.append(omega)
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"  [{i+1}/{n}] elapsed={time.time()-t0:.1f}s "
                  f"excluidas hasta ahora={n_excluded}")
    print(f"Dataset A=1 fijo: {len(out_w)} muestras validas de {n} "
          f"({n_excluded} excluidas por barrera de escape o no-convergencia)")
    return (np.array(out_a), np.array(out_b), np.array(out_w), n_excluded)


def build_variable_A_dataset(n, seed=1, A_grid=(0.5, 1.0, 1.5, 2.0),
                              verbose_every=200):
    """Dataset de amplitud variable: alpha, beta continuos; A sobre la malla."""
    rng = np.random.default_rng(seed)
    alphas = rng.uniform(0.5, 2.0, n)
    betas = rng.uniform(0.01, 0.3, n)
    A_choices = rng.choice(A_grid, size=n)

    out_a, out_b, out_A, out_w = [], [], [], []
    n_excluded = 0
    t0 = time.time()
    for i, (a, b, A) in enumerate(zip(alphas, betas, A_choices)):
        Aesc = escape_amplitude(a, b)
        if A >= Aesc:
            n_excluded += 1
            continue
        omega, c, ier = hb_fixed_release(a, b, A)
        if ier != 1 or not np.isfinite(omega):
            n_excluded += 1
            continue
        out_a.append(a)
        out_b.append(b)
        out_A.append(A)
        out_w.append(omega)
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"  [{i+1}/{n}] elapsed={time.time()-t0:.1f}s "
                  f"excluidas hasta ahora={n_excluded}")
    print(f"Dataset amplitud variable: {len(out_w)} muestras validas de {n} "
          f"({n_excluded} excluidas por barrera de escape o no-convergencia)")
    return (np.array(out_a), np.array(out_b), np.array(out_A), np.array(out_w),
            n_excluded)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000,
                     help="numero de muestras a intentar generar por dataset "
                          "(1000 = protocolo del manuscrito)")
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()

    print("=== Referencia puntual de control (alpha=1, beta=0.15) ===")
    Aesc_ref = escape_amplitude(1.0, 0.15)
    om_ref, _, ier_ref = hb_fixed_release(1.0, 0.15, 1.0)
    print(f"A_esc = {Aesc_ref:.6f} (esperado 1.359082), "
          f"omega(A=1) = {om_ref:.6f} (esperado 0.988831), ier={ier_ref}\n")

    print("=== 1. Dataset A=1 fijo (reemplaza Tabla II, Fig. 2b, Fig. 3b) ===")
    a1, b1, w1, exc1 = build_fixed_A_dataset(args.n)
    np.savez(f"{args.outdir}/quartic_free_fixedA.npz",
             alpha=a1, beta=b1, omega=w1,
             n_requested=args.n, n_excluded=exc1,
             columns=np.array(["alpha", "beta"]))
    print(f"  -> guardado {args.outdir}/quartic_free_fixedA.npz "
          f"({len(w1)} muestras)\n")

    print("=== 2. Dataset amplitud variable (reemplaza Tabla III, Fig. 4b, Fig. 6) ===")
    a2, b2, A2, w2, exc2 = build_variable_A_dataset(args.n)
    np.savez(f"{args.outdir}/quartic_free_variableA.npz",
             alpha=a2, beta=b2, A=A2, omega=w2,
             n_requested=args.n, n_excluded=exc2,
             columns=np.array(["alpha", "beta", "A"]))
    print(f"  -> guardado {args.outdir}/quartic_free_variableA.npz "
          f"({len(w2)} muestras)")
    print(f"  -> este es el numero que reemplaza al '625' en la Tabla III y "
          f"su nota al pie: {len(w2)} de {args.n}\n")

    print("Siguiente paso (en tu entorno pysr_env, no aqui):")
    print("""
import numpy as np
from pysr import PySRRegressor

# --- dataset A=1 fijo -> reemplaza fila x^4 de la Tabla II -----------------
d = np.load("quartic_free_fixedA.npz")
X = np.column_stack([d["alpha"], d["beta"]])
y = d["omega"]
model = PySRRegressor(
    niterations=200,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt"],
    maxsize=30,
    model_selection="best",
    random_state=42,
    deterministic=True,
    parallelism="serial",   # requerido junto con deterministic=True
    procs=0,
)
model.fit(X, y, variable_names=["alpha", "bta"])   # 'bta', no 'beta' (choca con sympy.beta)
print(model.get_best())                            # NO existe model.sexpr()

# --- dataset de amplitud variable -> reemplaza fila x^4 de la Tabla III ----
d2 = np.load("quartic_free_variableA.npz")
X2 = np.column_stack([d2["alpha"], d2["beta"], d2["A"]])
y2 = d2["omega"]
model2 = PySRRegressor(
    niterations=200,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt"],
    maxsize=35,
    model_selection="best",
    random_state=42,
    deterministic=True,
    parallelism="serial",
    procs=0,
)
model2.fit(X2, y2, variable_names=["alpha", "bta", "A"])
print(model2.get_best())
""")
