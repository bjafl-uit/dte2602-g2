import numpy as np
import matplotlib.pyplot as plt

from prefs import OUTPUT_DIR

x = np.linspace(-5, 5, 200)
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
#fig.suptitle('Aktiveringsfunksjoner', fontsize=16)

# Step Function
axs[0, 0].plot(x, np.where(x >= 0, 1, 0), color='blue')
axs[0, 0].set_title('Trinn-funksjon', fontsize=12)
axs[0, 0].set_ylabel('Output')

# Sigmoid Function
axs[0, 1].plot(x, 1 / (1 + np.exp(-x)), color='red')
axs[0, 1].set_title('Sigmoid funksjon', fontsize=12)

# Tanh Function
axs[1, 0].plot(x, np.tanh(x), color='green')
axs[1, 0].set_title('Hyperbolsk tangent (tanh)', fontsize=12)
axs[1, 0].set_xlabel('Sum av vektede input')
axs[1, 0].set_ylabel('Output')

# ReLU Function
axs[1, 1].plot(x, np.maximum(0, x), color='purple')
axs[1, 1].set_title('Rektifisert Lineær Enhet (ReLU)', fontsize=12)
axs[1, 1].set_xlabel('Sum av vektede input')

y_lim = (-1.1, 1.5)
y_ticks = np.arange(-1, 1.6, 0.5)
for ax in axs.flat:
    ax.set_ylim(y_lim)
    ax.set_yticks(y_ticks)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, linestyle=':', alpha=0.7)

plt.savefig(OUTPUT_DIR / 'activation_functions.svg')
plt.tight_layout()
plt.show()