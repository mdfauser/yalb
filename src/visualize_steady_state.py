import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("output/output_99999.csv")
Nx = df['x'].max() + 1
Ny = df['y'].max() + 1

ux = df['ux'].values.reshape(Nx, Ny)
uy = df['uy'].values.reshape(Nx, Ny)
speed = np.sqrt(ux**2 + uy**2)

# Normalized coordinates (0 to 1) — the x/L, y/L style
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

fig, ax = plt.subplots(figsize=(7, 6))

# Streamlines colored by velocity magnitude
strm = ax.streamplot(
    X.T, Y.T, ux.T, uy.T,
    color=speed.T,
    cmap='viridis',
    density=2.0,         # bump for denser streamlines like the reference
    linewidth=0.8,
    arrowsize=1.0,
)

cbar = plt.colorbar(strm.lines, ax=ax)
cbar.set_label(r'Velocity magnitude $|u|$ (lattice units)')

# Lid arrow on top
ax.annotate(
    '', xy=(0.7, 1.02), xytext=(0.3, 1.02),
    xycoords='axes fraction',
    arrowprops=dict(arrowstyle='->', color='red', lw=2)
)
ax.text(0.5, 1.05, r'$u_{\mathrm{lid}}$', color='red',
        ha='center', transform=ax.transAxes, fontsize=12)

ax.set_xlabel(r'$x/L$')
ax.set_ylabel(r'$y/L$')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig("cavity_streamlines.png", dpi=200, bbox_inches='tight')
plt.show()