import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import glob

files = sorted(glob.glob("output/output_*.csv"))

fig, ax = plt.subplots()

def update(filename):
    ax.clear()
    df = pd.read_csv(filename)
    Nx = df['x'].max() + 1
    Ny = df['y'].max() + 1
    rho = df['rho'].values.reshape(Nx, Ny)
    im = ax.imshow(rho.T, origin='lower', vmin=0.995, vmax=1.005, cmap='RdBu_r')
    ax.set_title(filename)

ani = animation.FuncAnimation(fig, update, frames=files, interval=100)
plt.show()