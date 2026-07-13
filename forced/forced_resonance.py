"""
forced_resonance.py
=====================

Bloque C - deriva la referencia forzada para los tres casos (cubico,
cuartico, quintico), reemplazando las mezclas lineales arbitrarias
retiradas (Rev1 #5, Rev2 #6, Rev3 #6, Rev4 #5).

Ecuacion base (balance armonico de un solo termino, caso cubico), ya en
el manuscrito revisado como Ec. (13):

    [(alpha - omega_ext^2 + (3/4)*beta*A^2)^2 + (delta*omega_ext)^2] * A^2 = F0^2

Esta es la ecuacion de "backbone + resonancia" estandar (Nayfeh & Mook;
Mickens) para un oscilador de Duffing cubico forzado armonicamente en
estado estacionario. Generaliza a n=5 de forma cerrada (ambas no
linealidades son impares, mismo tipo de balance de un armonico). Para
n=4 (asimetrica) NO existe un analogo cerrado simple -- se requiere el
mismo balance armonico multi-termino que usamos en el caso libre
(quartic_harmonic_balance.py), extendido para incluir el forzamiento.

Este script implementa AMBOS caminos:
  1. Solucion cerrada (polinomio en A^2) para n=3 y n=5 -- exacta.
  2. Balance armonico multi-armonico NUMERICO, valido para n=3,4,5 --
     permite validar 1) contra 2) en los casos donde ambos existen, y es
     el UNICO camino disponible para n=4.

Convencion: en estado estacionario, la respuesta esta bloqueada a la
frecuencia de forzamiento omega_ext (ver Sec. II.A del manuscrito
revisado). La cantidad a predecir por SR en el caso forzado es por tanto
la AMPLITUD DE RESPUESTA A, no una "frecuencia de respuesta" separada
(que es simplemente omega_ext).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve

# ======================================================================
# 1. SOLUCION CERRADA (polinomio en u = A^2), para n=3 y n=5
# ======================================================================

def _backbone_coeff(n):
    """Coeficiente c tal que omega^2 = alpha + c*beta*A^(n-1) en el caso
    libre (backbone). c=3/4 para n=3, c=5/8 para n=5 (ver Ecs. 4 y 6 del
    manuscrito). Solo definido para n impar (odd nonlinearity)."""
    if n == 3:
        return 0.75
    elif n == 5:
        return 0.625
    else:
        raise ValueError("Solucion cerrada solo definida para n=3 o n=5.")


def solve_forced_closed_form(alpha, beta, delta, F0, omega_ext, n):
    """
    Resuelve exactamente el polinomio en u=A^2 que resulta del balance
    armonico de un termino para osciladores forzados con no linealidad
    impar (n=3 o n=5):

        [(alpha - omega_ext^2 + c*beta*u^((n-1)/2))^2 + (delta*omega_ext)^2] * u = F0^2

    Devuelve TODAS las raices reales positivas de u (puede haber mas de
    una en la region de salto/histeresis cerca de la resonancia -- el
    fenomeno de "jump" caracteristico del oscilador de Duffing forzado).
    """
    c = _backbone_coeff(n)
    C = alpha - omega_ext**2
    D = c * beta

    if n == 3:
        # (C + D*u)^2 * u + (delta*omega_ext)^2 * u - F0^2 = 0
        # D^2 u^3 + 2*C*D u^2 + (C^2 + (delta*omega_ext)^2) u - F0^2 = 0
        coeffs = [D**2, 2 * C * D, C**2 + (delta * omega_ext) ** 2, -F0**2]
    elif n == 5:
        # (C + D*u^2)^2 * u + (delta*omega_ext)^2 * u - F0^2 = 0
        # D^2 u^5 + 2*C*D u^3 + (C^2 + (delta*omega_ext)^2) u - F0^2 = 0
        coeffs = [D**2, 0, 2 * C * D, 0, C**2 + (delta * omega_ext) ** 2, -F0**2]
    else:
        raise ValueError("n debe ser 3 o 5 para la solucion cerrada.")

    roots = np.roots(coeffs)
    real_positive = np.array([r.real for r in roots
                                if abs(r.imag) < 1e-8 and r.real > 0])
    return np.sort(real_positive)


def pick_branch(u_roots, prev_u=None):
    """
    Selecciona la rama fisicamente relevante cuando hay multiples raices
    (histeresis cerca de resonancia). Sin informacion de estabilidad
    explicita, usamos continuidad: la raiz mas cercana a la solucion del
    paso anterior (barrido en omega_ext). Si no hay solucion previa, se
    toma la raiz de mayor amplitud (rama de resonancia primaria).
    """
    if len(u_roots) == 0:
        return None
    if prev_u is None:
        return u_roots[-1]
    return u_roots[np.argmin(np.abs(u_roots - prev_u))]


# ======================================================================
# 2. BALANCE ARMONICO MULTI-ARMONICO NUMERICO (n=3,4,5 -- unico camino
#    para n=4)
# ======================================================================

def _forced_residual_fourier(params, alpha, beta, delta, F0, omega_ext, n,
                                n_harmonics, n_grid=2048):
    """
    Ansatz: x(theta) = c0 + sum_{k=1}^{M} [a_k*cos(k*theta) + b_k*sin(k*theta)],
    theta = omega_ext * t (la respuesta esta bloqueada a la frecuencia de
    forzamiento en estado estacionario).

    Residuo: omega_ext^2 * x''(theta) + delta*omega_ext*x'(theta) +
             alpha*x(theta) + beta*x(theta)^n - F0*cos(theta) = 0

    Se anula por armonico (cos y sin, k=0..M) via FFT real.
    """
    c0 = params[0]
    a = params[1 : n_harmonics + 1]
    b = params[n_harmonics + 1 :]

    theta = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    x = np.full_like(theta, c0)
    xp = np.zeros_like(theta)   # dx/dtheta
    xpp = np.zeros_like(theta)  # d^2x/dtheta^2
    for k in range(1, n_harmonics + 1):
        ak, bk = a[k - 1], b[k - 1]
        x += ak * np.cos(k * theta) + bk * np.sin(k * theta)
        xp += -k * ak * np.sin(k * theta) + k * bk * np.cos(k * theta)
        xpp += -(k**2) * ak * np.cos(k * theta) - (k**2) * bk * np.sin(k * theta)

    residual = (omega_ext**2) * xpp + delta * omega_ext * xp + alpha * x \
        + beta * x**n - F0 * np.cos(theta)

    R = np.fft.rfft(residual) / n_grid
    coeff0 = R[0].real
    cos_coeffs = 2 * R[1 : n_harmonics + 1].real
    sin_coeffs = -2 * R[1 : n_harmonics + 1].imag  # signo por convencion de FFT

    return np.concatenate(([coeff0], cos_coeffs, sin_coeffs))


def solve_forced_hb(alpha, beta, delta, F0, omega_ext, n, n_harmonics=5,
                      initial_guess=None):
    """
    Resuelve el balance armonico forzado multi-armonico para cualquier n
    (3, 4, o 5). Devuelve la amplitud de respuesta A = sqrt(a1^2+b1^2)
    (magnitud del armonico fundamental), junto con los coeficientes
    completos y el residuo maximo (diagnostico de convergencia).
    """
    if initial_guess is not None:
        params0 = initial_guess
    else:
        # estimacion inicial simple: respuesta lineal amortiguada
        denom = np.sqrt((alpha - omega_ext**2) ** 2 + (delta * omega_ext) ** 2)
        A0 = F0 / max(denom, 1e-6)
        params0 = np.zeros(2 * n_harmonics + 1)
        params0[1] = A0  # a_1

    sol, info, ier, msg = fsolve(
        _forced_residual_fourier, params0,
        args=(alpha, beta, delta, F0, omega_ext, n, n_harmonics),
        full_output=True,
    )
    if ier != 1:
        raise RuntimeError(f"Balance armonico forzado no convergio: {msg}")

    c0 = sol[0]
    a = sol[1 : n_harmonics + 1]
    b = sol[n_harmonics + 1 :]
    A_response = np.hypot(a[0], b[0])  # magnitud del armonico fundamental

    residual = _forced_residual_fourier(sol, alpha, beta, delta, F0, omega_ext,
                                          n, n_harmonics, n_grid=4096)
    max_residual = np.max(np.abs(residual))

    return {
        "A": A_response,
        "c0": c0,
        "a": a,
        "b": b,
        "params": sol,
        "max_residual": max_residual,
    }


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Validacion cruzada: cubico y quintico, forma cerrada vs. HB numerico
    # ------------------------------------------------------------
    alpha, beta, delta, F0, omega_ext = 1.0, 0.15, 0.05, 0.3, 1.1

    print("=== Validacion cruzada: forma cerrada vs. balance armonico numerico ===\n")
    for n in (3, 5):
        u_roots = solve_forced_closed_form(alpha, beta, delta, F0, omega_ext, n)
        A_closed = np.sqrt(pick_branch(u_roots)) if len(u_roots) else None

        hb = solve_forced_hb(alpha, beta, delta, F0, omega_ext, n, n_harmonics=5)
        A_hb = hb["A"]

        print(f"n={n}:")
        print(f"  raices u=A^2 (forma cerrada): {u_roots}")
        print(f"  A (forma cerrada, rama elegida) = {A_closed:.6f}")
        print(f"  A (balance armonico numerico)   = {A_hb:.6f}")
        print(f"  diferencia relativa             = {abs(A_closed-A_hb)/A_closed:.2e}")
        print(f"  residuo maximo (HB)              = {hb['max_residual']:.2e}")
        print()

    # ------------------------------------------------------------
    # Caso cuartico: solo camino numerico disponible
    # ------------------------------------------------------------
    hb4 = solve_forced_hb(alpha, beta, delta, F0, omega_ext, n=4, n_harmonics=5)
    print(f"n=4 (cuartico, solo HB numerico):")
    print(f"  A = {hb4['A']:.6f}, c0 = {hb4['c0']:.6f}, "
          f"residuo maximo = {hb4['max_residual']:.2e}")
