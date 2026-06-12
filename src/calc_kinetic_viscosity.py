import pandas as pd
import numpy as np
import glob

files = sorted(glob.glob("output/output_*.csv"),
               key=lambda x: int(x.split('_')[-1].split('.')[0]))

steps = []
amps  = []

for filename in files:
    df  = pd.read_csv(filename)
    col = df[df['x'] == 0].sort_values('y')
    amp = col['ux'].abs().max()
    step = int(filename.split('_')[-1].split('.')[0])
    if amp > 1e-10:  # skip steps where wave is already flat
        steps.append(step)
        amps.append(amp)

steps = np.array(steps)
amps  = np.array(amps)

# fit a line to log(amplitude) vs time
Ny     = 20  # your grid size
k      = 2.0 * np.pi / Ny
coeffs = np.polyfit(steps, np.log(amps), 1)  # linear fit in log space
slope  = coeffs[0]
tau = 0.8


nu_measured = -slope / (k * k)
nu_theory   = (tau - 0.5) / 3.0  # set tau to whatever you used

print(f"nu measured : {nu_measured:.6f}")
print(f"nu theory   : {nu_theory:.6f}")
print(f"relative error: {abs(nu_measured - nu_theory) / nu_theory * 100:.2f}%")