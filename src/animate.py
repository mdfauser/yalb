import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import glob

files = sorted(glob.glob("output/output_*.csv"),
               key=lambda x: int(x.split('_')[-1].split('.')[0]))

fig, ax = plt.subplots()

def update(filename):
    ax.clear()
    df = pd.read_csv(filename)
    Ny = df['y'].max() + 1
    # take one column (x=0) since all columns are identical
    col = df[df['x'] == 0].sort_values('y')
    ax.plot(col['y'], col['ux'], 'b-', linewidth=2)
    ax.set_ylim(-0.06, 0.06)  # fixed range so you see the decay
    ax.set_xlabel('y')
    ax.set_ylabel('ux')
    step = filename.split('_')[-1].split('.')[0]
    ax.set_title(f"step {step}  amp={col['ux'].abs().max():.6f}")

ani = animation.FuncAnimation(fig, update, frames=files, interval=100)
plt.show()