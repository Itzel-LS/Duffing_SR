"""
quartic_harmonic_balance.py
============================

Bloque A - punto #5 de la tabla de modificaciones (Rev1 #3, Rev4 #3a).

El oscilador cuartico libre  x'' + alpha*x + beta*x^4 = 0  tiene una fuerza
restauradora NO impar (x^4 no cambia de signo con x), por lo que el
potencial es asimetrico y el punto de equilibrio se desplaza:

    alpha*x0 + beta*x0^4 = 0   =>   x0 = 0  o  x0 = -(alpha/beta)^(1/3)

Como senalo el Revisor 1, el ansatz de un solo armonico x(t) = A*cos(wt+phi)
NO es suficiente: x^4 = A^4*cos^4(wt+phi) no tiene componente en la
frecuencia fundamental sin agregar un termino constante (offset) y
armonicos superiores. Proyectar solo sobre el fundamental con ese ansatz
da omega^2 = alpha a orden dominante, no la Ec. (5) original.

Este script implementa un balance armonico NUMERICO con:

    x(theta) = c0 + A*cos(theta) + c2*cos(2*theta) + ... + cM*cos(M*theta)

donde theta = omega*t, A (coeficiente del armonico fundamental) se fija
como la "amplitud" de la oscilacion, y (c0, c2, ..., cM, omega) se
resuelven numericamente exigiendo que el residuo de la ecuacion de
movimiento se anule en cada armonico hasta el orden M considerado.

El numero de armonicos M es un parametro: al aumentarlo, la solucion debe
converger (los coeficientes de orden alto y la correccion a omega deben
estabilizarse). Ese estudio de convergencia ES la evidencia que sostiene
la eleccion final de M para el manuscrito.

Uso tipico
----------
    from quartic_harmonic_balance import solve_quartic_hb

    sol = solve_quartic_hb(alpha=1.0, beta=0.15, A=1.0, n_harmonics=3)
    print(sol["omega"], sol["c0"], sol["coeffs"])
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve


def _residual_fourier_coeffs(params, alpha, beta, A, n_harmonics, n_grid=2048):
    """
    Dado un vector de parametros [omega, c0, c2, c3, ..., c_M], construye
    x(theta) = c0 + A*cos(theta) + sum_{k=2}^{M} c_k*cos(k*theta),
    evalua el residuo R(theta) = omega^2 * x''(theta) + alpha*x(theta) + beta*x(theta)^4
    sobre una grilla fina de un periodo, y devuelve sus coeficientes de
    Fourier (coseno) para los armonicos 0, 1, ..., M -- estos deben
    anularse todos para que el ansatz resuelva la ecuacion en ese
    subespacio de armonicos.
    """
    omega = params[0]
    c0 = params[1]
    higher = params[2:]  # c2, c3, ..., c_M

    theta = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)

    x = np.full_like(theta, c0) + A * np.cos(theta)
    xpp = -A * np.cos(theta)  # segunda derivada respecto de theta del termino fundamental
    for k, ck in enumerate(higher, start=2):
        x += ck * np.cos(k * theta)
        xpp += -(k**2) * ck * np.cos(k * theta)

    residual = (omega**2) * xpp + alpha * x + beta * x**4

    # Coeficientes de Fourier (coseno) del residuo, via FFT real
    R = np.fft.rfft(residual) / n_grid
    # coeficiente de cos(k*theta): 2*Re(R[k]) para k>=1, Re(R[0]) para k=0
    coeff0 = R[0].real
    coeffs_k = 2 * R[1 : n_harmonics + 1].real  # armonicos 1..M

    # El armonico fundamental (k=1) se fuerza a que su coeficiente en el
    # residuo sea cero (es la condicion que fija omega dado A).
    # Junto con coeff0 y los armonicos 2..M, formamos el sistema completo.
    return np.concatenate(([coeff0], coeffs_k))


def solve_quartic_hb(alpha: float, beta: float, A: float, n_harmonics: int = 3,
                       initial_guess: dict | None = None) -> dict:
    """
    Resuelve el balance armonico multi-armonico para el oscilador cuartico
    libre, con `n_harmonics` armonicos superiores ademas del fundamental
    (es decir, el ansatz incluye c0, c1=A (fijo), c2, ..., c_{n_harmonics}).

    Parameters
    ----------
    alpha, beta : parametros del oscilador (alpha*x + beta*x^4).
    A : amplitud fijada del armonico fundamental.
    n_harmonics : numero de armonicos mas alla del fundamental a incluir
        (n_harmonics=1 -> ansatz minimo c0 + A*cos(theta), el que el
        Revisor 1 mostro que es insuficiente; n_harmonics>=2 agrega
        correcciones autoconsistentes).

    Returns
    -------
    dict con omega, c0, coeffs (armonicos superiores), y el residuo final
    (para verificar convergencia: debe ser cercano a cero).
    """
    # valores iniciales: omega ~ sqrt(alpha), c0 desde el equilibrio
    # desplazado a orden dominante, armonicos superiores en cero
    x0_guess = -np.sign(beta) * (abs(alpha / beta)) ** (1 / 3) * 1e-2  # arranque conservador
    omega_guess = np.sqrt(alpha)

    if initial_guess:
        omega_guess = initial_guess.get("omega", omega_guess)
        x0_guess = initial_guess.get("c0", x0_guess)

    params0 = [omega_guess, x0_guess] + [0.0] * (n_harmonics - 1)

    sol, info, ier, msg = fsolve(
        _residual_fourier_coeffs,
        params0,
        args=(alpha, beta, A, n_harmonics),
        full_output=True,
    )

    if ier != 1:
        raise RuntimeError(f"El balance armonico no convergio: {msg}")

    omega = sol[0]
    c0 = sol[1]
    higher_coeffs = sol[2:]

    # verificacion: evaluar el residuo maximo en la grilla completa
    theta = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    x = np.full_like(theta, c0) + A * np.cos(theta)
    xpp = -A * np.cos(theta)
    for k, ck in enumerate(higher_coeffs, start=2):
        x += ck * np.cos(k * theta)
        xpp += -(k**2) * ck * np.cos(k * theta)
    residual = (omega**2) * xpp + alpha * x + beta * x**4
    max_residual = np.max(np.abs(residual))

    return {
        "omega": omega,
        "c0": c0,
        "coeffs": dict(zip(range(2, n_harmonics + 1), higher_coeffs)),
        "max_residual": max_residual,
        "n_harmonics": n_harmonics,
    }


def convergence_study(alpha: float, beta: float, A: float, max_harmonics: int = 5):
    """
    Corre solve_quartic_hb para n_harmonics = 1, 2, ..., max_harmonics y
    reporta como convergen omega y el residuo maximo. Esta es la evidencia
    de convergencia que debe acompanar la referencia cuartica en el
    manuscrito (p. ej. como tabla en el apendice).
    """
    results = []
    prev_solution = None
    for m in range(1, max_harmonics + 1):
        guess = None
        if prev_solution is not None:
            guess = {"omega": prev_solution["omega"], "c0": prev_solution["c0"]}
        sol = solve_quartic_hb(alpha, beta, A, n_harmonics=m, initial_guess=guess)
        results.append(sol)
        prev_solution = sol
    return results


if __name__ == "__main__":
    alpha, beta, A = 1.0, 0.15, 1.0

    print("=== Estudio de convergencia: oscilador cuartico libre ===")
    print(f"alpha={alpha}, beta={beta}, A={A}\n")
    print(f"{'n_harmonics':>12} | {'omega':>12} | {'c0':>12} | {'max_residual':>14}")
    print("-" * 58)
    results = convergence_study(alpha, beta, A, max_harmonics=5)
    for r in results:
        print(f"{r['n_harmonics']:>12} | {r['omega']:>12.8f} | {r['c0']:>12.8f} | {r['max_residual']:>14.2e}")

    print()
    print("Comparacion con la referencia (retirada) Eq. (5) del manuscrito original:")
    omega_old = np.sqrt(alpha + (3 / 16) * alpha**2 * beta**2)
    print(f"  Eq. (5) original (no derivable del ansatz simple): omega = {omega_old:.8f}")
    print(f"  omega^2 = alpha puro (orden dominante, ansatz de 1 armonico): omega = {np.sqrt(alpha):.8f}")
    print(f"  Balance armonico convergido (este script):                  omega = {results[-1]['omega']:.8f}")
