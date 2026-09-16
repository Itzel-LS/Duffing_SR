"""
Regeneracion completa de la referencia RK45+FFT del quintico libre con
extraccion de pico interpolada (Referee A, punto A2, ronda 2).

`quintic_fft_check.py` ya demostro que el argmax crudo del bin FFT introduce
un sesgo de hasta 1.08%, y que la interpolacion parabolica del pico lo reduce
por debajo de 5e-4, sin necesidad de alargar el registro. Este script aplica
esa correccion al protocolo COMPLETO del manuscrito:

    x(0) = A = 1, x'(0) = 0
    40 periodos integrados, se descarta el primer 40%,
    64 puntos por periodo, ventana de Hann,
    pico = argmax + interpolacion parabolica (en vez de argmax crudo),
    N = 1000 muestras, split 800/200 con semilla fija (igual que el resto
    de los datasets del articulo).

La etiqueta 'high precision' / 'exact in the numerical sense' NO debe
reintroducirse: el dataset resultante sigue siendo una extraccion espectral,
solo que con un sesgo de lectura mucho menor.

Salida:
    quintic_free_fft_interp.npz   columnas: alpha, beta -> omega
    (con la particion train/test 800/200 ya marcada en 'split')

Uso:
    python3 quintic_fft_regen.py           # N=1000, protocolo completo
    python3 quintic_fft_regen.py --n 50    # corrida rapida de prueba
"""
import argparse
import time
import numpy as np
from scipy.integrate import solve_ivp

A0 = 1.0


def rhs_factory(alpha, beta):
    def rhs(t, y):
        return [y[1], -alpha * y[0] - beta * y[0] ** 5]
    return rhs


def omega_fft_interp(alpha, beta, n_periods=40, ppp=64, discard=0.40,
                      rtol=1e-9, atol=1e-9):
    """Protocolo del manuscrito + interpolacion parabolica del pico."""
    Test = 2 * np.pi / np.sqrt(alpha)
    tmax = n_periods * Test
    N = int(n_periods * ppp)
    tt = np.linspace(0, tmax, N, endpoint=False)
    s = solve_ivp(rhs_factory(alpha, beta), [0, tmax], [A0, 0.0],
                  t_eval=tt, rtol=rtol, atol=atol)
    x = s.y[0]
    n0 = int(discard * len(x))
    x = x[n0:]
    n = len(x)
    fs = 1.0 / (tt[1] - tt[0])
    w = np.hanning(n)
    X = np.abs(np.fft.rfft(x * w))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    k = int(np.argmax(X[1:]) + 1)
    if 1 <= k < len(X) - 1:
        a, b, c = np.log(X[k - 1]), np.log(X[k]), np.log(X[k + 1])
        denom = (a - 2 * b + c)
        d = 0.5 * (a - c) / denom if denom != 0 else 0.0
    else:
        d = 0.0
    f_peak = freqs[k] + d * (freqs[1] - freqs[0])
    return 2 * np.pi * f_peak


def build_dataset(n, seed=0, verbose_every=100):
    rng = np.random.default_rng(seed)
    alphas = rng.uniform(0.5, 2.0, n)
    betas = rng.uniform(0.01, 0.3, n)
    omegas = np.empty(n)
    t0 = time.time()
    for i, (a, b) in enumerate(zip(alphas, betas)):
        omegas[i] = omega_fft_interp(a, b)
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"  [{i+1}/{n}] elapsed={time.time()-t0:.1f}s")
    return alphas, betas, omegas


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0,
                     help="usar la MISMA semilla que se uso para el dataset "
                          "publicado originalmente, si se conoce, para que "
                          "solo cambie el metodo de extraccion y no la "
                          "muestra de (alpha,beta)")
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()

    print("=== Referencia puntual de control (alpha=1, beta=0.15) ===")
    w_ref = omega_fft_interp(1.0, 0.15)
    w_theory = np.sqrt(1.0 + 0.625 * 0.15)
    print(f"omega interpolado = {w_ref:.6f}, "
          f"sqrt(alpha+5/8 beta) = {w_theory:.6f}, "
          f"diferencia relativa = {100*abs(w_ref-w_theory)/w_theory:.4f}%\n")

    print(f"=== Generando {args.n} muestras (protocolo completo, "
          f"extraccion interpolada) ===")
    alphas, betas, omegas = build_dataset(args.n, seed=args.seed)

    # split 80/20 igual que el resto de los datasets del articulo
    rng_split = np.random.default_rng(args.seed)
    idx = rng_split.permutation(args.n)
    n_train = int(0.8 * args.n)
    split = np.array(["train"] * args.n, dtype=object)
    split[idx[n_train:]] = "test"

    out_path = f"{args.outdir}/quintic_free_fft_interp.npz"
    np.savez(out_path, alpha=alphas, beta=betas, omega=omegas, split=split,
              columns=np.array(["alpha", "beta"]))
    print(f"-> guardado {out_path}  ({args.n} muestras, "
          f"{n_train} train / {args.n - n_train} test)")

    print("\nSiguiente paso (en tu entorno pysr_env, no aqui):")
    print("""
import numpy as np
from pysr import PySRRegressor

d = np.load("quintic_free_fft_interp.npz", allow_pickle=True)
train = d["split"] == "train"
test = d["split"] == "test"

X_train = np.column_stack([d["alpha"][train], d["beta"][train]])
y_train = d["omega"][train]
X_test = np.column_stack([d["alpha"][test], d["beta"][test]])
y_test = d["omega"][test]

model = PySRRegressor(
    niterations=200,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt"],
    maxsize=30,
    model_selection="best",
    random_state=42,
    deterministic=True,
    parallelism="serial",
    procs=0,
)
model.fit(X_train, y_train, variable_names=["alpha", "bta"])
print(model.get_best())

pred = model.predict(X_test)
ss_res = np.sum((y_test - pred) ** 2)
ss_tot = np.sum((y_test - y_test.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
mse = np.mean((y_test - pred) ** 2)
print(f"R2 = {r2:.4f}  MSE = {mse:.3e}")
# Este R2/MSE, junto con la expresion descubierta, es lo que reemplaza
# omega = sqrt(alpha + 0.6178 beta), R2=0.9946 en Tabla II y en el texto.
# NO reportar el 0.6169 del curve_fit como si fuera este resultado.
""")
