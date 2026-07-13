# Duffing SR Analysis — Code and Data Generation Pipeline

Companion code repository for:

**"Symbolic Rediscovery of Amplitude-Frequency Relations in Nonlinear Duffing
Oscillators via Machine Learning"**
I. Luviano-Soto, J. P. Pérez-Aguilar, A. Raya

This repository accompanies the manuscript per the Data Availability Statement
and responds to the reproducibility observations raised by all four referees
(Referee 1 pt. 8, Referee 2 pt. 2.8/2.9, Referee 3 pt. 4, Referee 4 pt. 4.8).

## Requirements

```bash
pip install numpy scipy pandas scikit-learn matplotlib sympy pysr
python -c "import pysr; pysr.install()"   # installs the Julia backend, once
```

All datasets and PySR runs use `random_state = 42` (or documented offsets
thereof) throughout. PySR runs use `deterministic=True, parallelism="serial"`
for exact reproducibility of the reported expressions.

**Running order matters.** Each `.py` file is written to be pasted as a
single notebook cell (or run as a standalone script from the command line).
Files within each folder must be run **in the order listed below**, since
later scripts depend on functions defined in earlier ones — no cross-file
`import` statements are used, so that each stage can be pasted directly into
a Jupyter cell without maintaining a package structure.

**PySR training scripts are best run from a terminal (CMD/shell), not from a
Jupyter or Colab cell.** Long PySR runs with verbose output can overwhelm
notebook rendering engines; a plain terminal does not have this problem. See
`forced/cmd_train_pysr_forced.py` for the pattern (results are logged to a
`.txt` file with immediate flush, so no output is lost).

---

## `free/` — Free-oscillator pipeline (Table II, Table III, Fig. 4)

Run in this order:

1. **`rk45_reference.py`** — RK45 (Dormand–Prince, via `scipy.integrate.solve_ivp`)
   + FFT frequency extraction, with the ω = 2πf unit correction applied
   (Referee 2 pt. 2.3, Referee 3 pt. 3.3) and full reproducibility settings
   documented: initial conditions, integration interval, tolerances,
   sampling, windowing, peak-selection rule (Referee 1 pt. 1.7, Referee 4
   pt. 4.7).
2. **`quartic_harmonic_balance.py`** — multi-harmonic balance solver for the
   quartic free oscillator, replacing the withdrawn Eq. (5) (Referee 1 pt.
   1.3, Referee 4 pt. 4.3). Includes the bounded-orbit physical validity
   criterion (escape barrier of the asymmetric potential
   V(x) = ½αx² + ⅕βx⁵).
3. **`generate_quartic_dataset.py`** — generates `quartic_free_dataset.csv`
   (A = 1 fixed, 1000 samples) using step 2. → Table II, row x⁴.
4. **`generate_quintic_free_rk45_dataset.py`** — generates
   `quintic_free_rk45_dataset.csv` (A = 1 fixed, 1000 samples) using step 1.
   → Table II, row x⁵ RK45+FFT.
5. **`generate_amplitude_datasets.py`** — generates the three
   amplitude-variable datasets (`cubic_amplitude_dataset.csv`,
   `quartic_amplitude_dataset.csv`, `quintic_amplitude_dataset.csv`) for the
   amplitude-dependence study (Referee 1 pt. 1.2, Referee 4 pt. 4.2).
6. **`train_pysr_amplitude.py`** — trains SR on the amplitude-variable
   datasets. → Table III, Fig. 4.
7. **`train_pysr_table2_gaps.py`** — trains SR on the A = 1 fixed quartic and
   quintic-RK45 datasets. → completes Table II (rows x⁴ and x⁵ RK45+FFT).

## `forced/` — Forced-oscillator pipeline (Table IV, Fig. 5)

Run in this order:

1. **`forced_resonance.py`** — closed-form resonance relation (single-harmonic
   balance, Eq. 13/14 and quintic analogue) with branch selection for the
   hysteretic region (jump phenomenon), plus a general multi-harmonic
   balance solver used for the quartic forced case (no closed form exists
   there).
2. **`generate_forced_datasets.py`** — generates the three forced datasets;
   target variable is the response amplitude A, with F0 included explicitly
   as an input (Referee 1 pt. 1.5, Referee 4 pt. 4.5). Samples with multiple
   real roots (hysteresis) or non-convergent quartic solutions are excluded
   and counted explicitly.
3. **`cmd_train_pysr_forced.py`** — trains SR on the three closed-form/
   multi-harmonic-balance forced datasets (`sqrt` and `cbrt` operators).
   → Table IV (rows: x³ single-harm. balance, x⁴ multi-harm. balance, x⁵
   single-harm. balance), Fig. 5.
4. **`rk45_forced_reference.py`** — direct numerical integration (RK45) of
   the forced quintic oscillator, without the single-harmonic approximation;
   extracts the steady-state response amplitude A via least-squares
   projection onto the fundamental harmonic. Cross-validated against the
   single-harmonic balance of step 1 (~1.3% agreement).
5. **`generate_quintic_forced_rk45_dataset.py`** — generates the RK45-based
   quintic forced dataset (1000 samples) using step 4.
6. **`train_pysr_quintic_forced_rk45.py`** — trains SR on this dataset.
   → Table IV, row x⁵ RK45+FFT ($R^2=0.768$, modestly higher than the
   single-harmonic-balance quintic row, since the RK45 reference required
   no sample exclusion for hysteresis).

## `validation/` — Appendix A (additional validation)

Run in this order:

1. **`generate_validation_datasets.py`** — generates the extrapolation
   split (train on β∈[0.01,0.15], test on the disjoint β∈[0.15,0.30]) and
   the three noise-perturbed datasets (1%, 2%, 5% relative noise) for the
   cubic free-oscillator pilot case.
2. **`train_pysr_additional_validation.py`** — runs the extrapolation test,
   the noise-robustness test, and the 20-seed run-to-run variability study.
   → Appendix A (all four subsections: residual analysis relies on
   `figures/residual_analysis.py` instead, see below).

Responds jointly to Referee 1 pt. 1.9, Referee 2 pt. 2.3/2.8/2.9, Referee 3
"Additional points" 4, and Referee 4 pt. 4.9.

## `figures/` — Figure generation

- **`figure2_free_scatter.py`** → **Fig. 2** (predicted vs. reference ω,
  200-point scatter, for the four free-oscillator reference methods: LP,
  multi-harmonic balance, RK45+FFT, HAM-P). Requires
  `free/quartic_harmonic_balance.py` loaded first, and the datasets from
  `free/generate_quartic_dataset.py` and
  `free/generate_quintic_free_rk45_dataset.py`.
- **`figure3_free_curves.py`** → **Fig. 3** (ω vs. α at fixed β=0.15, with
  absolute error panels, for the three free cases with closed-form or
  semi-analytical references: LP, multi-harmonic balance, HAM-P; RK45+FFT
  excluded since it has no continuous ω(α) form). Requires
  `free/quartic_harmonic_balance.py` loaded first.
- **`figure6_amplitude.py`** → **Fig. 4** (ω vs. A at fixed α, β, for the
  three free cases). Requires `free/quartic_harmonic_balance.py` loaded
  first.
- **`figure5_forced_amplitude.py`** → **Fig. 5** (response amplitude A vs.
  ω_ext, the classical resonance curve, for the three forced cases).
  Requires `forced/forced_resonance.py` loaded first.
- **`residual_analysis.py`** → **Fig. 6** (free-oscillator residuals) and
  **Fig. 7** (forced-oscillator residuals), colored by A and by β/detuning
  respectively (Appendix A.1). Uses the already-discovered SR expressions
  directly (hardcoded), reading the datasets generated above — does not
  require PySR to run.
- **`noise_figure.py`** → supporting figure for Appendix A.3 (noise
  robustness), showing R² and the recovered physical coefficient as a
  function of noise level. Uses the numeric results from
  `train_pysr_additional_validation.py` (hardcoded from the actual run
  output) — does not require PySR to run.

---

## Known physical/numerical findings documented via this code

- The free quartic potential V(x) = ½αx² + ⅕βx⁵ is unbounded from below
  (escape barrier at x_c = −(α/β)^(1/3)); only parameter combinations with
  bounded periodic orbits are retained (625/1000 in the amplitude study).
- The forced-case resonance relation is cubic (or quintic) in A² and can
  have multiple real roots near resonance (the classical Duffing
  jump/hysteresis phenomenon); samples with multiple roots were excluded
  from training (120/1000 cubic, 133/1000 quintic).
- SR could not reach machine-precision R² for the forced cases
  (0.53–0.83) even with `cbrt` added to the operator set, consistent with
  the target being an implicit polynomial root that in the multi-valued
  regime requires trigonometric functions in closed form (*casus
  irreducibilis*).
- A pilot noise-robustness test shows that a 5% relative perturbation to
  the reference frequencies causes SR to recover a qualitatively incorrect
  expression (collapsing the physical coefficient 3/4 to an implicit 1)
  despite a still-reasonable R² = 0.895 — illustrating that R² alone does
  not certify correct physics recovery.
- For the quintic forced case, direct numerical integration (RK45) achieves
  a modestly higher R² (0.768) than the single-harmonic-balance closed form
  (0.718), plausibly because the RK45 reference required no sample
  exclusion for the hysteretic multi-root region, unlike the closed-form
  reference (133/1000 samples excluded).

## PySR hyperparameters used (for reproducibility)

| Script | niterations | populations | population_size | operators |
|---|---|---|---|---|
| `train_pysr_amplitude.py` | 80 | 20 | 40 | +,-,*,/,sqrt |
| `train_pysr_table2_gaps.py` | 80 | 20 | 40 | +,-,*,/,sqrt |
| `cmd_train_pysr_forced.py` | 200 | 30 | 50 | +,-,*,/,sqrt,cbrt |
| `train_pysr_quintic_forced_rk45.py` | 200 | 30 | 50 | +,-,*,/,sqrt,cbrt |
| `train_pysr_additional_validation.py` | 60 | 15 | 35 | +,-,*,/,sqrt |

All runs use `random_state=42` (extrapolation/noise tests) or `random_state=0..19`
(20-seed variability study), `deterministic=True`, `parallelism="serial"`,
`model_selection="best"`.

## Note on variable naming

The symbol `beta` collides with `sympy.beta()` (the Beta function), which
PySR uses internally to construct symbolic expressions from the search.
All scripts rename the `beta` column to `bta` immediately before passing
data to `PySRRegressor` — this is purely a naming workaround and does not
affect the physical meaning of the results; in reported equations, `bta`
should be read as β.
