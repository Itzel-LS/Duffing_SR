"""
rk45_forced_reference.py
===========================

Cierra la fila pendiente de la Tabla IV: integra numericamente el
oscilador forzado quintico (ecuacion completa, sin aproximacion de
balance armonico) y extrae la AMPLITUD de respuesta en estado
estacionario -- no la frecuencia, que en estado estacionario es
trivialmente omega_ext.

Ecuacion: x'' + delta*x' + alpha*x + beta*x^5 = F0*cos(omega_ext*t)

Metodologia (consistente con el resto del estudio):
  - Condiciones iniciales: x(0)=0, v(0)=0 (se parte del reposo).
  - Integracion: t_end fijo, suficientemente largo para que el
    transitorio decaiga incluso en el caso de amortiguamiento debil
    (delta pequeno).
  - Se descarta una fraccion del inicio (transitorio) y se ajusta la
    señal restante por minimos cuadrados a A*cos(omega_ext*t)+B*sin(omega_ext*t),
    extrayendo la amplitud de respuesta A = sqrt(A^2+B^2) -- la misma
    convencion que ya usamos en el balance armonico forzado
    (forced_resonance.py), para que ambos metodos sean directamente
    comparables.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

T_END = 300.0            # tiempo total de integracion (unidades adimensionales)
TRANSIENT_FRACTION = 2/3  # fraccion inicial descartada como transitorio
RTOL = ATOL = 1e-8


def duffing_forced_rhs(t, y, alpha, beta, delta, F0, omega_ext):
    x, v = y
    dxdt = v
    dvdt = F0 * np.cos(omega_ext * t) - delta * v - alpha * x - beta * x**5
    return [dxdt, dvdt]


def rk45_forced_amplitude(alpha: float, beta: float, delta: float, F0: float,
                            omega_ext: float) -> dict:
    """
    Integra el oscilador forzado quintico con RK45 y extrae la amplitud
    de respuesta en estado estacionario, ajustando la señal (tras
    descartar el transitorio) a A*cos(omega_ext*t) + B*sin(omega_ext*t)
    por minimos cuadrados.

    Returns
    -------
    dict con: A (amplitud de respuesta), fit_residual (bondad del ajuste,
    normalizado), t, x (para diagnostico/graficar si se requiere).
    """
    period = 2 * np.pi / omega_ext
    max_step = period / 20  # resolver bien la forma de onda

    sol = solve_ivp(
        fun=duffing_forced_rhs,
        t_span=(0.0, T_END),
        y0=[0.0, 0.0],
        method="RK45",
        args=(alpha, beta, delta, F0, omega_ext),
        rtol=RTOL,
        atol=ATOL,
        dense_output=True,
        max_step=max_step,
    )
    if not sol.success:
        raise RuntimeError(f"Integracion RK45 fallida: {sol.message}")

    n_samples = int(T_END / max_step)
    t_eval = np.linspace(0.0, T_END, n_samples)
    x_full = sol.sol(t_eval)[0]

    n_discard = int(TRANSIENT_FRACTION * n_samples)
    t = t_eval[n_discard:]
    x = x_full[n_discard:]

    if not np.all(np.isfinite(x)):
        raise RuntimeError("Trayectoria diverge (no acotada) -- fuera de la region valida.")
    if np.max(np.abs(x)) > 1e3:
        raise RuntimeError("Amplitud creciente sin cota -- posible resonancia sin saturar "
                             "o region no fisica.")

    design = np.column_stack([np.cos(omega_ext * t), np.sin(omega_ext * t)])
    coeffs, *_ = np.linalg.lstsq(design, x, rcond=None)
    a_coef, b_coef = coeffs
    A = np.hypot(a_coef, b_coef)

    fit = design @ coeffs
    residual = x - fit
    fit_residual = np.sqrt(np.mean(residual**2)) / (np.abs(A) + 1e-12)

    return {
        "A": A,
        "fit_residual": fit_residual,
        "t": t,
        "x": x,
    }


if __name__ == "__main__":
    # Validacion cruzada contra el balance de un armonico (forced_resonance.py),
    # mismo punto que usamos antes: alpha=1, beta=0.15, delta=0.05, F0=0.3, omega_ext=1.1
    alpha, beta, delta, F0, omega_ext = 1.0, 0.15, 0.05, 0.3, 1.1

    print("=== Validacion: RK45 forzado quintico vs. balance de un armonico ===\n")
    res = rk45_forced_amplitude(alpha, beta, delta, F0, omega_ext)
    print(f"RK45 (integracion completa): A = {res['A']:.6f}, "
          f"residuo de ajuste relativo = {res['fit_residual']:.4e}")

    # comparacion con el balance de un armonico ya construido (forced_resonance.py)
    try:
        exec(open("forced_resonance.py").read().split('if __name__')[0])
        u_roots = solve_forced_closed_form(alpha, beta, delta, F0, omega_ext, 5)
        if len(u_roots):
            A_hb = np.sqrt(pick_branch(u_roots))
            print(f"Balance de un armonico (Eq. 14):  A = {A_hb:.6f}")
            print(f"Diferencia relativa: {abs(res['A']-A_hb)/A_hb*100:.2f}%")
    except FileNotFoundError:
        print("(forced_resonance.py no encontrado en este directorio -- "
              "omite la comparacion cruzada)")
