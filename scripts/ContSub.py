"""
Created on Tue Oct 13 2026

@author: M. Pólvora Fonseca

Continuum subtract the initial map.

Outputs:
    - Continuum subtracted spectral cube as .fits file
"""

# --------------------------------- Libraries -------------------------------- #

import os
import glob

from astropy.io import fits
import numpy as np

# ----------------------------------- Files ---------------------------------- #

cube_file_path = '/data2/dafonseca/nitrogen_haro11/data/Haro11_ec_all.rc.fits'
tess_file_path = '/data2/dafonseca/data/powerbin_tessellation.txt'
FADO_tables_file_path = '/data2/dafonseca/FADOv1b_patch1.0/best/'


# ------------------------------ Pixels per bin ------------------------------ #

tess_data = np.loadtxt(tess_file_path, skiprows=1)

x = tess_data[:, 0].astype(int)
y = tess_data[:, 1].astype(int)
binNum = tess_data[:, 2].astype(int)

# Create a dictionary: bin number -> indices of its pixels.
bin_pixels = {}

for bi in np.unique(binNum):
    mask = binNum == bi
    bin_pixels[bi] = np.column_stack((x[mask], y[mask]))


# --------------------------- Continuum subtraction -------------------------- #

# Create an empty cube equal to the original cube
# This is were I'll put the continuum subtraction results

contsub_cube = np.zeros(, dtype=np.float32)



# ---------------------- Save continuum subtracted file ---------------------- #



