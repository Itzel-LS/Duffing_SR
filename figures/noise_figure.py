import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12,
})

noise_levels = [1, 2, 5]
r2_values = [0.9960, 0.9855, 0.8951]
coef_values = [0.7474, 0.7474, 1.0]  # 1.0 representa el colapso a alpha+beta
true_coef = 0.75

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

ax1 = axes[0]
ax1.plot(noise_levels, r2_values, 'o-', color='C0', markersize=8, linewidth=2)
ax1.set_xlabel('Noise level (%)')
ax1.set_ylabel(r'$R^2$')
ax1.set_title('(a) Goodness of fit')
ax1.grid(True, linestyle=':', alpha=0.4)
ax1.set_ylim(0.85, 1.01)

ax2 = axes[1]
ax2.plot(noise_levels, coef_values, 'o-', color='C3', markersize=8, linewidth=2,
          label='Recovered coefficient')
ax2.axhline(true_coef, color='gray', linestyle='--', linewidth=1.5,
             label=r'True value ($3/4$)')
ax2.set_xlabel('Noise level (%)')
ax2.set_ylabel(r'Coefficient of $\beta$')
ax2.set_title('(b) Recovered physical coefficient')
ax2.grid(True, linestyle=':', alpha=0.4)
ax2.legend(frameon=False, fontsize=9)
ax2.annotate('Collapses to\nqualitatively\nwrong form', xy=(5, 1.0),
              xytext=(2.7, 0.92), fontsize=8.5,
              arrowprops=dict(arrowstyle='->', color='black', lw=1))

plt.tight_layout()
plt.savefig('noise_robustness.png', dpi=250, bbox_inches='tight')
print("Guardada: noise_robustness.png")
