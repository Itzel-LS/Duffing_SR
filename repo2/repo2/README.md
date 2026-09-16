# Duffing SR Analysis — Code and Data Generation Pipeline

Companion code repository for:

**"Symbolic Rediscovery of Amplitude-Frequency Relations in Nonlinear Duffing
Oscillators via Machine Learning"** I. Luviano-Soto, J. P. Pérez-Aguilar, A. Raya

This repository accompanies the manuscript per the Data Availability Statement
and responds to the reproducibility observations raised by all four referees
across both review rounds.

## Round-2 updates (second referee report)

- **Quartic free oscillator** (`free/quartic_free_dataset.csv`,
  `free/quartic_amplitude_dataset.csv`): regenerated under a single,
  consistent amplitude definition throughout — the release amplitude,
  $x(0)=A$, $\dot{x}(0)=0$, imposed as $\sum_k c_k = A$ in the harmonic-balance
  ansatz — replacing the earlier $c_1=A$ prescription, which is not
  equivalent for this asymmetric potential. Physical-validity filtering
  (escape barrier) is applied under this definition; 965/1000 samples are
  retained for the fixed-$A$ dataset and 662/1000 for the amplitude-variable
  dataset.
- **Quintic RK45+FFT** (`free/quintic_free_rk45_dataset.csv`): frequency
  extraction now uses parabolic interpolation of the spectral peak, rather
  than the raw FFT-bin argmax, which carries a systematic bias of up to
  1.08% at the sampled record length.
- **New**: `free/quintic_free_firstharm_dataset.csv` — the closed-form
  first-harmonic-balance reference for the quintic free oscillator
  (Table II, row $x^5$ first-harm. balance), generated directly from the
  analytical expression $\omega=\sqrt{\alpha+\tfrac{5}{8}\beta A^4}$.
- **New**: `forced/quintic_forced_rk45_dataset.csv` — the forced quintic
  case in which the response amplitude is obtained by projecting the
  steady-state trajectory onto the driving harmonic
  ($A=\sqrt{a_c^2+a_s^2}$), rather than from the closed-form single-harmonic
  balance relation; both are now reported as separate rows in Table V.
- **New**: `forced/reduced_variables/` — the three forced configurations
  retrained directly in the dimensionless groups $(\Omega,\zeta,f)$ of the
  reduction $x=(\alpha/\beta)^{1/(n-1)}X$, $t=s/\sqrt{\alpha}$, sampled
  uniformly in the reduced variables themselves (Sec. III C, Table IV).

## Requirements

```
pip install numpy scipy pandas scikit-learn matplotlib sympy pysr
python -c "import pysr; pysr.install()"   # installs the Julia backend, once
```

All datasets and PySR runs use `random_state = 42` (or documented offsets
thereof) throughout. PySR runs use `deterministic=True, parallelism="serial"` for exact reproducibility of the reported expressions.

**Running order matters.** Each `.py` file is written to be pasted as a
single notebook cell (or run as a standalone script from the command line).
Files within each folder must be run **in the order listed below**, since
later scripts depend on functions defined in earlier ones — no cross-file `import` statements are used, so that each stage can be pasted directly into
a Jupyter cell without maintaining a package structure.

**PySR training scripts are best run from a terminal (CMD/shell), not from a
Jupyter or Colab cell.** Long PySR runs with verbose output can overwhelm
notebook rendering engines; a plain terminal does not have this problem.

---

## `free/` — Free-oscillator pipeline (Table II, Table III, Fig. 3, Fig. 4)

Run in this order:

1. **`rk45_reference.py`** — RK45 (Dormand–Prince, via `scipy.integrate.solve_ivp`)
  FFT frequency extraction with parabolic peak interpolation, and full
  reproducibility settings documented: initial conditions, integration
  interval, tolerances, sampling, windowing, peak-selection rule.
2. **`quartic_harmonic_balance.py`** — multi-harmonic balance solver for the
  quartic free oscillator, using the release-amplitude constraint
  $\sum_k c_k = A$. Includes the bounded-orbit physical validity criterion
  (escape barrier of the asymmetric potential $V(x)=\tfrac12\alpha x^2+\tfrac15\beta x^5$).
3. Generates `quartic_free_dataset.csv` (A = 1 fixed, 965/1000 samples) using step 2. → Table II, row x⁴.
4. Generates `quintic_free_rk45_dataset.csv` (A = 1 fixed, 1000 samples,
  interpolated extraction) using step 1. → Table II, row x⁵ RK45+FFT.
5. Generates `quintic_free_firstharm_dataset.csv` directly from the closed-form
  expression. → Table II, row x⁵ first-harm. balance.
6. Generates the three amplitude-variable datasets (`cubic_amplitude_dataset.csv`,
  `quartic_amplitude_dataset.csv`, `quintic_amplitude_dataset.csv`) for the
  amplitude-dependence study — quartic under the release-amplitude
  definition (662/1000 retained).
7. Trains SR on the amplitude-variable datasets. → Table III, Fig. 4.
8. Trains SR on the A = 1 fixed quartic and quintic-RK45 datasets. → completes Table II.

## `forced/` — Forced-oscillator pipeline (Table V, Fig. 5)

Run in this order:

1. **`generate_forced_datasets.py`** — generates the forced datasets in raw
  variables; target variable is the response amplitude A, with F0 included
  explicitly as an input. Samples with multiple real roots (hysteresis) or
  non-convergent quartic solutions are excluded and counted explicitly:
  880/1000 (cubic), 787/1000 (quartic), 867/1000 (quintic, closed form).
  `quintic_forced_rk45_dataset.csv` (projection method) converges for all
  1000 samples.
2. Trains SR on the four forced datasets (`sqrt` and `cbrt` operators). → Table V, Fig. 5.

### `forced/reduced_variables/` — Sec. III C, Table IV

1. **`pysr_reduced_forced.py`** — cubic and quintic forced cases, retrained
  directly in $(\Omega,\zeta,f)$, sampled uniformly over
  $\Omega\in[0.60,1.60]$, $\zeta\in[0.005,0.150]$, $f\in[0.02,1.00]$.
2. **`quartic_reduced_forced.py`** — the quartic forced case in reduced
  variables, using a full multi-harmonic balance with both cosine and sine
  components (the driven steady state has a phase lag absent in the free
  case).
3. Retained samples: 908/1000 (n=3), 671/1000 (n=4), 846/1000 (n=5), after
  discarding multi-root and non-converged cases.

## `validation/` — Appendix A (additional validation)

Unchanged from the first round; run in this order:

1. **`generate_validation_datasets.py`** — generates the extrapolation
  split (train on β∈[0.01,0.15], test on the disjoint β∈[0.15,0.30]) and
  the three noise-perturbed datasets (1%, 2%, 5% relative noise) for the
  cubic free-oscillator pilot case.
2. **`train_pysr_additional_validation.py`** — runs the extrapolation test,
  the noise-robustness test, and the 20-seed run-to-run variability study.

## `figures/` — Figure generation

- **`figure2_free_scatter.py`** → **Fig. 2** (predicted vs. reference,
  four free-oscillator panels).
- **`figure3_free_curves.py`** → **Fig. 3** ($\omega(\alpha)$ at fixed
  $\beta$, cubic/quartic/quintic first-harm.).
- **`figure6_amplitude.py`** → **Fig. 4** ($\omega$ vs. $A$ at fixed
  $\alpha,\beta$, for the three free cases).
- **`figure5_forced_amplitude.py`** → **Fig. 5** (response amplitude A vs.
  $\omega_\mathrm{ext}$, for the four forced cases).
- **`residual_analysis.py`** → **Fig. 6** (free-oscillator residuals),
  colored by A. Uses the already-discovered SR expressions directly,
  reading the datasets generated above — does not require PySR to run.

---

## Known physical/numerical findings documented via this code

- The free quartic potential $V(x)=\tfrac12\alpha x^2+\tfrac15\beta x^5$ is
  unbounded from below (escape barrier at $x_c=-(\alpha/\beta)^{1/3}$); only
  parameter combinations with bounded periodic orbits are retained
  (662/1000 in the amplitude study, under the release-amplitude definition).
- The forced-case resonance relation is cubic (or quintic) in $A^2$ and can
  have multiple real roots near resonance (the classical Duffing
  jump/hysteresis phenomenon); samples with multiple roots were excluded
  from training (120/1000 cubic, 133/1000 quintic closed-form).
- SR could not reach machine-precision $R^2$ for the forced cases in raw
  variables (0.53–0.83); retraining directly in the reduced dimensionless
  groups $(\Omega,\zeta,f)$ raises this substantially (0.887–0.969),
  indicating that a meaningful part of the earlier degradation was
  attributable to the redundant five-parameter representation rather than
  solely to the algebraic operator set. A residual gap persists.
- A pilot noise-robustness test shows that a 5% relative perturbation to
  the reference frequencies causes SR to recover a qualitatively incorrect
  expression (collapsing the physical coefficient 3/4 to an implicit 1)
  despite a still-reasonable $R^2=0.895$ — illustrating that $R^2$ alone
  does not certify correct physics recovery.

## Note on variable naming

The symbol `beta` collides with `sympy.beta()` (the Beta function), and
`zeta` with `sympy.zeta()` (the Riemann zeta function) — both are used
internally by PySR to construct symbolic expressions. All scripts rename
these columns to `bta` and `zet` immediately before passing data to
`PySRRegressor`; this is purely a naming workaround and does not affect the
physical meaning of the results. In reported equations, `bta` should be
read as $\beta$ and `zet` as $\zeta$.
