"""
regenerate_figure2.py
========================
Regenera Figure_2_free_scatter.png completa (4 paneles: a=cubico, b=cuartico,
c=quintico RK45+FFT, d=quintico first-harmonic).

Solo los paneles b y c cambian de verdad (usan los datasets corregidos de la
ronda 2); a y d se recalculan con la formula exacta conocida, que SR reproduce
a precision de maquina (R2=1.0000), tal como ya reportaba el manuscrito.

Uso:
    python regenerate_figure2.py
Salida:
    Figure_2_free_scatter.png  (listo para reemplazar al archivo viejo)
"""
import numpy as np
import matplotlib.pyplot as plt
import csv

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 11, "axes.titlesize": 10,
})

rng = np.random.default_rng(0)


def load_csv(path, cols):
    data = {c: [] for c in cols}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            for c in cols:
                data[c].append(float(row[c]))
    return {c: np.array(v) for c, v in data.items()}


def load_csv_with_split(path, numeric_cols):
    """Como load_csv, pero conserva la columna 'split' (train/test) como texto."""
    data = {c: [] for c in numeric_cols}
    split_col = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            for c in numeric_cols:
                data[c].append(float(row[c]))
            split_col.append(row["split"])
    out = {c: np.array(v) for c, v in data.items()}
    out["split"] = np.array(split_col)
    return out


# ---------- panel (a) cubico: datos reales, subconjunto A=1 ----------
d_a = load_csv("cubic_amplitude_dataset.csv", ["alpha", "beta", "A", "omega"])
mask_a = np.isclose(d_a["A"], 1.0)
a_a, b_a = d_a["alpha"][mask_a], d_a["beta"][mask_a]
ref_a = d_a["omega"][mask_a]
sr_a = np.sqrt(a_a + 0.75 * b_a)  # SR reproduce el coeficiente exacto 3/4

# ---------- panel (b) cuartico: dataset corregido ----------
d_b = load_csv("quartic_free_dataset.csv", ["alpha", "beta", "omega"])
alpha_b, beta_b, omega_b = d_b["alpha"], d_b["beta"], d_b["omega"]
n_b = len(omega_b)
rng_b = np.random.default_rng(0)  # misma semilla que correr_pysr_ronda2.py
idx_b = rng_b.permutation(n_b)
n_train_b = int(0.8 * n_b)
test_b = idx_b[n_train_b:]
ref_b = omega_b[test_b]
a_b, be_b = alpha_b[test_b], beta_b[test_b]
sr_b = np.sqrt(a_b - be_b * ((be_b / (a_b - 0.4739 * (be_b * a_b) /
               (a_b - 1.9888085 * be_b))) - 0.015744861))

# ---------- panel (c) quintico RK45+FFT: dataset reextraido ----------
d_c = load_csv_with_split("quintic_free_rk45_dataset.csv", ["alpha", "beta", "omega"])
test_mask_c = d_c["split"] == "test"
a_c, be_c = d_c["alpha"][test_mask_c], d_c["beta"][test_mask_c]
ref_c = d_c["omega"][test_mask_c]
sr_c = np.sqrt(a_c - (be_c * -0.6101759)) * 1.0005283

# ---------- panel (d) quintico first-harmonic: datos reales, subconjunto A=1 ----------
d_d = load_csv("quintic_amplitude_dataset.csv", ["alpha", "beta", "A", "omega"])
mask_d = np.isclose(d_d["A"], 1.0)
a_d, b_d = d_d["alpha"][mask_d], d_d["beta"][mask_d]
ref_d = d_d["omega"][mask_d]
sr_d = np.sqrt(a_d + 0.625 * b_d)

panels = [
    ("(a) $x^3$ Free Vibration", ref_a, sr_a, "LP reference"),
    ("(b) $x^4$ Free Vibration", ref_b, sr_b, "Multi-harm. balance"),
    ("(c) $x^5$ Free Vibration", ref_c, sr_c, "RK45+FFT reference"),
    ("(d) $x^5$ Free Vibration", ref_d, sr_d, "First-harm. balance"),
]

fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
for ax, (title, ref, sr, sub) in zip(axes, panels):
    ss_res = np.sum((ref - sr) ** 2)
    ss_tot = np.sum((ref - ref.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    lo, hi = min(ref.min(), sr.min()), max(ref.max(), sr.max())
    ax.plot([lo, hi], [lo, hi], "k-", lw=1, zorder=1)
    ax.scatter(ref, sr, s=12, alpha=0.6, zorder=2)
    ax.set_title(f"{title}\n$R^2={r2:.4f}$", fontsize=10)
    ax.set_xlabel(f"Reference $\\omega$\n({sub})", fontsize=11)
    ax.tick_params(labelsize=9)
    if ax is axes[0]:
        ax.set_ylabel("Predicted $\\omega$", fontsize=11)
    ax.set_aspect("equal", adjustable="box")

plt.tight_layout(pad=1.4)
plt.subplots_adjust(bottom=0.22, wspace=0.35)
plt.savefig("Figure_2_free_scatter.png", dpi=250, bbox_inches="tight")
print("Guardado: Figure_2_free_scatter.png")
print(f"  R2 panel b (cuartico) = {1 - np.sum((ref_b-sr_b)**2)/np.sum((ref_b-ref_b.mean())**2):.6f}")
print(f"  R2 panel c (quintico RK45) = {1 - np.sum((ref_c-sr_c)**2)/np.sum((ref_c-ref_c.mean())**2):.6f}")
