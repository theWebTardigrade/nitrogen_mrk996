"""
Created on Thu Oct 08 2026

@author: M. Pólvora Fonseca

Create the spectrum input files for FADO

Output file contains:

    wavelength (Å), flux (erg/s*cm**2), error (erg/s*cm**2), 0
"""

# --------------------------------- Libraries -------------------------------- #

import sys
import os
import numpy as np
import pandas as pd
from astropy.io import fits


# ----------------------------------- Data ----------------------------------- #

cube_file_path = '/home/polaris/nitrogen_haro11/data/Haro11_ec_all.rc.fits'
tess_file_path = '/home/polaris/nitrogen_haro11/data/powerbin_tessellation.txt'
output_path = '/home/polaris/nitrogen_haro11/data/FADOInput'


# -------------------------- Load the cube to memory ------------------------- #

file = fits.open(cube_file_path)

flux_cube = file['DATA'].data
variance_cube = file['STAT'].data

header = file['DATA'].header


# ---------------------- Import the tesselation regions ---------------------- #

tess_data = np.loadtxt(tess_file_path, skiprows=1)

x = tess_data[:, 0].astype(int)
y = tess_data[:, 1].astype(int)
binNum = tess_data[:, 2].astype(int)


# Create dictionary:
# bin number -> array of [x, y] pixels

bin_pixels = {}

for bi in np.unique(binNum):
    mask = binNum == bi
    bin_pixels[bi] = np.column_stack((x[mask], y[mask]))


# ----------------------------- Wavelength ---------------------------------- #

crval3 = header['CRVAL3']
crpix3 = header['CRPIX3']
cdelt3 = header['CD3_3']

nwave = flux_cube.shape[0]

wavelength = crval3 + (np.arange(nwave) + 1 - crpix3) * cdelt3

# ----------------------------- Average and save ----------------------------- #

os.makedirs(output_path, exist_ok=True)


for bin_number, pixels in bin_pixels.items():

    print(f"Processing bin {bin_number} ({len(pixels)} pixels)")

    spectra = []
    variances = []

    # Get spectrum for every pixel in this bin
    for xi, yi in pixels:

        spectrum = flux_cube[:, yi, xi]
        variance = variance_cube[:, yi, xi]

        spectra.append(spectrum)
        variances.append(variance)

    spectra = np.array(spectra)
    variances = np.array(variances)


    # ------------------------- Average spectrum ---------------------------- #

    # Mean flux over all pixels in the bin
    average_flux = np.nanmean(spectra, axis=0)


    # ----------------------- Error on the mean ------------------------------ #

    # For an average of independent measurements:
    #
    # Var(mean) = sum(Var_i) / N^2
    #

    n_pixels = np.sum(np.isfinite(spectra), axis=0)

    summed_variance = np.nansum(variances, axis=0)

    average_variance = summed_variance / n_pixels**2

    average_error = np.sqrt(average_variance)


    # ------------------------------ Output ---------------------------------- #

    output_data = np.column_stack((
        wavelength,
        average_flux,
        average_error,
        np.zeros_like(wavelength)
    ))

    bin_number = str(bin_number)
    save_bin_number = bin_number.zfill(4)

    output_file = os.path.join(
        output_path,
        f'Spec_{save_bin_number}.txt'
    )


    np.savetxt(
        output_file,
        output_data,
        fmt='%.8e %.8e %.8e %d'
    )


    print(f"  Saved: {output_file}")


file.close()

print("Finished.")