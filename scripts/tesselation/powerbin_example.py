"""
#####################################################################

Copyright (C) 2025 Michele Cappellari  
E-mail: michele.cappellari_at_physics.ox.ac.uk  

Updated versions of this software are available at:  
https://pypi.org/project/powerbin/  

If you use this software in published research, please acknowledge it as:  
“PowerBin method by Cappellari (2025, MNRAS, 544, 1432)”  
https://ui.adsabs.harvard.edu/abs/2025MNRAS.544.1432C

This software is provided “as is”, without any warranty of any kind,  
express or implied.  

Permission is granted for:  
 - Non-commercial use.  
 - Modification for personal or internal use, provided that this  
   copyright notice and disclaimer remain intact and unaltered  
   at the beginning of the file.  

All other rights are reserved. Redistribution of the code, in whole or in part,  
is strictly prohibited without prior written permission from the author.  

#####################################################################

V1.0.0: PowerBin created — MC, Oxford, 10 September 2025

"""

from importlib import resources

import numpy as np
import matplotlib.pyplot as plt

from powerbin import PowerBin

#-----------------------------------------------------------------------------

# Usage example for the PowerBin class.
#
# Input data: Columns 1–4 of sample_data_ngc2273.txt contain x, y coordinates
# of each spaxel, followed by their Signal and Noise.
#
# Capacity specification:
# 1) Additive (array): Provide (S/N)^2 per pixel, which adds up in the
#    Poisson limit. This is the simplest and, for very large datasets
#    (millions of pixels), also the fastest approach.
# 2) Non-additive (function): Provide a callable that computes the bin
#    capacity from its pixel indices. This can model effects like correlated
#    noise. For small or moderate datasets the speed difference is negligible.
#
# S/N Formula:
# The S/N for a bin is calculated as `np.sum(signal) / np.sqrt(np.sum(noise**2))`.
# This is the standard formula for uncorrelated noise and should be the default
# choice for most applications. One would only modify this for special cases,
# such as:
#  - To account for known covariance in the noise between pixels. The example
#    shows a commented-out empirical correction, but this is data-dependent
#    and not a general prescription (see Fig.11 from
#    http://adsabs.harvard.edu/abs/2015A%26A...576A.135G).
#  - To perform complex calculations on each bin (e.g., fitting a model to
#    extract kinematics), which is a possible but advanced use case.

data_path = '/home/polaris/nitrogen_mrk996/scripts/sample_data_ngc2273.txt'
x, y, signal, noise = np.loadtxt(data_path).T
xy = np.column_stack([x, y])

target_sn = 50

# --- Define Capacity Specification ---
additive = False  # Set True for additive array, False for non-additive function

if additive:
    # ADDITIVE CASE: (S/N)^2 is additive when noise is Poissonian.
    capacity_spec = (signal / noise)**2

else:
    # NON-ADDITIVE CASE: Define a function for custom capacity logic.
    def capacity_spec(index):
        # Standard S/N for the bin
        sn = np.sum(signal[index]) / np.sqrt(np.sum(noise[index]**2))
        # Example of modelling correlated noise (commented out):
        # sn /= 1 + 1.07 * np.log10(len(index))
        return sn**2

pow = PowerBin(xy, capacity_spec, target_capacity=target_sn**2)

# The binning was performed on (S/N)^2, but for plotting we want S/N.
pow.plot(capacity_scale='sqrt', ylabel='S/N')

plt.show(block=True)
