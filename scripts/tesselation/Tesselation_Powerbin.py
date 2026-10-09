"""
Created on Mon Oct 05 2026
@author: M. Pólvora Fonseca

PowerBin tessellation of the continuum image.

Input file contains:
    x, y, signal, noise
"""

# --------------------------------- Libraries -------------------------------- #

import sys
import numpy as np
import matplotlib.pyplot as plt
from powerbin import PowerBin


# ----------------------------- Parameters ---------------------------------- #

input_file = ('/home/polaris/nitrogen_haro11/data/cont_7000_7200_perchannel_input_tess.csv')
output_file = ('/home/polaris/nitrogen_haro11/data/powerbin_tessellation.txt')

target_sn = 50


# ----------------------------- Read input ---------------------------------- #

data = np.loadtxt(
    input_file,
    delimiter=',',
    skiprows=1
)

x = data[:, 0]
y = data[:, 1]
signal = data[:, 2]
variance = data[:, 3]

# ----------------------------- S/N selection ------------------------------- #

noise = np.sqrt(variance)

sn = signal / noise

# Keep only spaxels with finite values and S/N > 1
good = (
    np.isfinite(x)
    & np.isfinite(y)
    & np.isfinite(signal)
    & np.isfinite(variance)
)

x = x[good]
y = y[good]
signal = signal[good]
variance = variance[good]
noise = noise[good]

xy = np.column_stack([x, y])

print(f"Spaxels with S/N > 1: {np.sum(good)} / {len(good)}")
print(f"Minimum S/N: {np.min(signal / noise):.2f}")
print(f"Maximum S/N: {np.max(signal / noise):.2f}")


# --------------------------- Plot S/N distribution -------------------------- #

plt.figure(figsize=(8, 6))
plt.hist(
    sn,
    bins=50,
    color='blue',
    alpha=0.7,
    edgecolor='black',
    log=True
)
plt.title('S/N Distribution')
plt.xlabel('S/N')
plt.ylabel('Number of Spaxels') 

plt.show()

#sys.exit("Check the S/N distribution plot and adjust the target S/N if necessary.")


# ----------------------------- PowerBin ------------------------------------- #

print('----------------------------------------')
print('PowerBin tessellation')
print('----------------------------------------')
print(f'Number of pixels: {len(x)}')
print(f'Target S/N:       {target_sn}')
print('----------------------------------------')


def capacity_spec(index):
    """
    Calculate (S/N)^2 for a proposed bin.

    For independent noise:

        S/N = sum(signal) / sqrt(sum(noise^2))
    """

    sn = (np.sum(signal[index])/ np.sqrt(np.sum(noise[index]**2)))

    return sn**2


pow = PowerBin(xy, capacity_spec, target_capacity=target_sn**2)


# ----------------------------- Plot ----------------------------------------- #

pow.plot(capacity_scale='sqrt', ylabel='S/N')
plt.savefig('/home/polaris/nitrogen_haro11/data/PowerBinTesselation.pdf')
plt.show()

# ----------------------------- Save binning --------------------------------- #

np.savetxt(
    output_file,
    np.column_stack([
        x,
        y,
        pow.bin_num
    ]),
    fmt='%10.6f %10.6f %8i',
    header='x y binNum',
    comments=''
)

print(f'Number of bins: {len(np.unique(pow.bin_num))}')
print(f'Saved to: {output_file}')