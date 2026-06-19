import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("../output/output_500.csv")

Nx = df['x'].max() + 1
Ny = df['y'].max() + 1

# reshape flat CSV back into 2D grids
rho = df['rho'].values.reshape(Nx, Ny)
ux  = df['ux'].values.reshape(Nx, Ny)
uy  = df['uy'].values.reshape(Nx, Ny)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# density
axes[0].imshow(rho.T, origin='lower')
axes[0].set_title('density')
plt.colorbar(axes[0].images[0], ax=axes[0])

# velocity magnitude
speed = np.sqrt(ux**2 + uy**2)
axes[1].imshow(speed.T, origin='lower')
axes[1].set_title('velocity magnitude')
plt.colorbar(axes[1].images[0], ax=axes[1])

# velocity field as arrows
x = np.arange(Nx)
y = np.arange(Ny)
X, Y = np.meshgrid(x, y)
axes[2].quiver(X[::4, ::4], Y[::4, ::4],
               ux.T[::4, ::4], uy.T[::4, ::4])
axes[2].set_title('velocity field')

plt.tight_layout()
plt.savefig("output_500.png")
plt.show()