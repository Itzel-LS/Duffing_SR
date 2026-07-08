"""
rk45_reference.py
==================

Bloque A - puntos #8 y #9 de la tabla de modificaciones.

Genera la referencia numerica RK45+FFT para el oscilador de Duffing libre,
resolviendo dos observaciones de los revisores:

  (Rev2 #3, Rev3 #3) La FFT devuelve frecuencia ciclica f [Hz]; la referencia
  angular correcta es omega = 2*pi*f. La version anterior del manuscrito
  usaba f directamente como si fuera omega, lo que producia el coeficiente
  espureo ~1/(2*pi) = 0.159... en las Tablas II y III.

  (Rev1 #7, Rev4 #7) La Ec. (7) del manuscrito original no describe un
  esquema RK45 adaptativo real. Aqui se usa scipy.integrate.solve_ivp con
  method='RK45' (Dormand-Prince embebido, orden 4(5), paso adaptativo) y se
  documentan explicitamente todos los parametros de reproducibilidad que
  pedian los revisores: condiciones iniciales, intervalo de integracion,
  tolerancias, tasa de muestreo, ventaneo, resolucion FFT y regla de
  seleccion de pico.

Uso tipico
----------
    from rk45_reference import rk45_fft_reference

    result = rk45_fft_reference(alpha=1.0, beta=0.15, n=5, delta=0.0,
                                  A0=1.0)
    print(result["omega_rad_s"])   # frecuencia angular corregida (rad/s)
    print(result["diagnostics"])   # armonicos/subarmonicos detectados
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

# ======================================================================
# PARAMETROS DE REPRODUCIBILIDAD (documentados explicitamente, Rev1 #7 / Rev4 #7)
# ======================================================================
# Estos son los valores por defecto usados en el estudio. Cambialos aqui si
# tu protocolo real fue distinto -- lo importante es que ahora SI estan
# declarados, en vez de dejarlos implicitos.

DEFAULT_SETTINGS = dict(
    # --- condiciones iniciales ---
    # Oscilador liberado desde el reposo con desplazamiento inicial = A0.
    # v(0) = 0 es la convencion estandar para amplitud-frecuencia libre.
    x0_is_amplitude=True,   # x(0) = A0, v(0) = 0

    # --- integracion ---
    n_periods_estimate=40,      # numero de periodos "estimados" a integrar
    rtol=1e-9,                  # tolerancia relativa (RK45 adaptativo)
    atol=1e-9,                  # tolerancia absoluta

    # --- muestreo para la FFT (se requiere grilla uniforme) ---
    # El integrador es de paso adaptativo; se usa dense_output=True y se
    # reevalua la solucion en una grilla temporal uniforme para poder
    # aplicar la FFT correctamente.
    samples_per_estimated_period=64,

    # --- remocion de transitorio ---
    transient_fraction=0.40,    # se descarta el primer 40% de la senal

    # --- ventaneo espectral ---
    window="hann",               # ventana de Hann para reducir fuga espectral

    # --- seleccion de pico ---
    # Se busca el pico dominante en el espectro de magnitud (excluyendo el
    # bin de frecuencia cero) y se reportan tambien los dos siguientes picos
    # mas altos como diagnostico de armonicos/subarmonicos.
    n_diagnostic_peaks=3,
)


def duffing_rhs(t, y, alpha, beta, delta, n, F0=0.0, omega_ext=0.0):
    """
    Lado derecho de x'' + delta*x' + alpha*x + beta*x^n = F0*cos(omega_ext*t).

    y = [x, v], con v = x'.
    """
    x, v = y
    forcing = F0 * np.cos(omega_ext * t)
    dxdt = v
    dvdt = forcing - delta * v - alpha * x - beta * x**n
    return [dxdt, dvdt]


def rk45_fft_reference(
    alpha: float,
    beta: float,
    n: int,
    delta: float = 0.0,
    A0: float = 1.0,
    F0: float = 0.0,
    omega_ext: float = 0.0,
    settings: dict | None = None,
) -> dict:
    """
    Integra el oscilador de Duffing con RK45 (Dormand-Prince adaptativo) y
    extrae la frecuencia dominante por FFT, ya en rad/s (omega = 2*pi*f).

    Parameters
    ----------
    alpha, beta, n, delta : parametros fisicos del oscilador (ver Ec. 1/2/3
        del manuscrito).
    A0 : amplitud/desplazamiento inicial (x(0) = A0, v(0) = 0) para el caso
        libre. Ignorado en la practica si F0 != 0 y se prefiere partir del
        reposo (ver nota abajo).
    F0, omega_ext : parametros de forzamiento externo. F0 = 0 reproduce el
        caso libre.
    settings : sobreescribe DEFAULT_SETTINGS si se requiere.

    Returns
    -------
    dict con:
        omega_rad_s      -> frecuencia angular dominante, YA corregida (2*pi*f)
        freq_hz           -> frecuencia ciclica dominante (antes de corregir)
        t, x              -> senal en tiempo (post-transitorio, para graficar)
        spectrum_freq_hz  -> eje de frecuencias del espectro (Hz)
        spectrum_mag      -> magnitud del espectro (para graficar/depurar)
        diagnostics       -> lista de (freq_hz, magnitud_relativa) de los
                              picos dominantes, util para detectar
                              armonicos/subarmonicos
        settings_used     -> copia de los parametros de reproducibilidad
                              efectivamente usados (para el paper / anexo)
    """
    cfg = dict(DEFAULT_SETTINGS)
    if settings:
        cfg.update(settings)

    # --- condiciones iniciales ---
    omega0_guess = np.sqrt(alpha)  # frecuencia natural lineal, solo para
                                     # estimar la ventana temporal de integracion
    if F0 == 0.0:
        y0 = [A0, 0.0]
    else:
        # Para el caso forzado se integra desde el reposo hasta alcanzar el
        # estado estacionario; el transitorio se descarta mas abajo.
        y0 = [0.0, 0.0]
        omega0_guess = max(omega0_guess, omega_ext)

    # --- intervalo de integracion ---
    T_est = 2 * np.pi / omega0_guess
    t_end = cfg["n_periods_estimate"] * T_est
    n_samples = cfg["samples_per_estimated_period"] * cfg["n_periods_estimate"]
    t_eval = np.linspace(0.0, t_end, n_samples)

    # --- integracion RK45 (Dormand-Prince embebido, paso adaptativo) ---
    sol = solve_ivp(
        fun=duffing_rhs,
        t_span=(0.0, t_end),
        y0=y0,
        method="RK45",
        args=(alpha, beta, delta, n, F0, omega_ext),
        rtol=cfg["rtol"],
        atol=cfg["atol"],
        dense_output=True,
        max_step=T_est / 20,  # asegura resolucion minima aun con paso adaptativo
    )
    if not sol.success:
        raise RuntimeError(f"La integracion RK45 no convergio: {sol.message}")

    # Reevaluar en grilla uniforme (requerida para la FFT)
    x_full = sol.sol(t_eval)[0]

    # --- remover transitorio ---
    n_discard = int(cfg["transient_fraction"] * n_samples)
    t = t_eval[n_discard:]
    x = x_full[n_discard:]

    # --- ventaneo ---
    if cfg["window"] == "hann":
        win = np.hanning(len(x))
    else:
        win = np.ones(len(x))
    x_windowed = (x - np.mean(x)) * win

    # --- FFT ---
    dt = t_eval[1] - t_eval[0]
    fs = 1.0 / dt
    spectrum = np.fft.rfft(x_windowed)
    freqs_hz = np.fft.rfftfreq(len(x_windowed), d=dt)
    magnitude = np.abs(spectrum)

    # --- seleccion de pico (excluye bin DC) ---
    magnitude_no_dc = magnitude.copy()
    magnitude_no_dc[0] = 0.0
    peak_idx = np.argmax(magnitude_no_dc)
    freq_hz_dominant = freqs_hz[peak_idx]

    # frecuencia angular, CORREGIDA (Rev2 #3 / Rev3 #3)
    omega_rad_s = 2 * np.pi * freq_hz_dominant

    # --- diagnostico de armonicos / subarmonicos ---
    order = np.argsort(magnitude_no_dc)[::-1][: cfg["n_diagnostic_peaks"]]
    diagnostics = [
        (freqs_hz[i], magnitude_no_dc[i] / magnitude_no_dc[peak_idx])
        for i in order
    ]

    # resolucion espectral (Delta f = 1 / duracion de la ventana)
    delta_f = fs / len(x_windowed)

    return {
        "omega_rad_s": omega_rad_s,
        "freq_hz": freq_hz_dominant,
        "t": t,
        "x": x,
        "spectrum_freq_hz": freqs_hz,
        "spectrum_mag": magnitude,
        "diagnostics": diagnostics,
        "fft_resolution_hz": delta_f,
        "sampling_rate_hz": fs,
        "settings_used": cfg,
    }


if __name__ == "__main__":
    # Ejemplo rapido de verificacion: caso quintico libre, alpha=1, beta=0.15
    res = rk45_fft_reference(alpha=1.0, beta=0.15, n=5, delta=0.0, A0=1.0)
    print("=== Ejemplo: quintico libre, alpha=1.0, beta=0.15, A0=1.0 ===")
    print(f"omega (rad/s), corregido : {res['omega_rad_s']:.6f}")
    print(f"f (Hz), crudo de la FFT  : {res['freq_hz']:.6f}")
    print(f"resolucion FFT (Hz)      : {res['fft_resolution_hz']:.6f}")
    print("Picos dominantes (freq_hz, magnitud relativa):")
    for f, m in res["diagnostics"]:
        print(f"   {f:.4f} Hz   rel_mag={m:.4f}")
    print()
    print("NOTA: el coeficiente ~0.159*sqrt(alpha) que aparecia en la version")
    print("anterior del manuscrito es consistente con NO haber aplicado el")
    print("factor 2*pi (1/(2*pi) = 0.15915...). Con esta correccion, el nuevo")
    print("coeficiente deberia acercarse a sqrt(alpha) ~ 1.0 para este ejemplo,")
    print("no a 0.159*sqrt(alpha).")
