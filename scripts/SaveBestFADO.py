"""
Created on Mon Oct 19 2026

@author: M. Pólvora Fonseca

Save the best fit FADO table

Outputs:
    - .fits file with the best continuum fit for all the bins
"""

# --------------------------------- Libraries -------------------------------- #

import os

import numpy as np

from astropy.io import fits
from astropy.table import Table

# ----------------------------------- Paths ---------------------------------- #

fado_output_path = '/data2/dafonseca/FADOv1b_patch1.0/output/'

best_FADO_output_path = '/data2/dafonseca/FADOv1b_patch1.0/best/'

fado_output_list = os.listdir(os.path.join(fado_output_path,'*1D*'))

# ----------------------------- Take the best fit ---------------------------- #

def readFADO(file_path):
    file = fits.open(file_path)

    # Get the header and data
    header = file[0].header
    data = file[0].data

    # Close file so it does not eat my RAM
    file.close()

    # Get min, max and step wavelength
    min_wave = header['OLSYNINI']
    max_wave = header['OLSYNFIN']
    step_wave = header['OLSYNDEL']

    # Create wavelength list
    wave_list = np.arange(min_wave, max_wave, step_wave)

    # Get best fit data
    best_fit = data[3]

    # Get normalization
    norm_wave = header['LAMBDA_0']
    norm_flux = header['GALSNORM']

    return wave_list, best_fit, norm_wave, norm_flux


# ------------------------- Save best fit in new file ------------------------ #

best_fit_file = 'FADO_best_fit.fits'

bin_indices = []
norm_waves = []
norm_fluxes = []
best_fit_spectra = []

# Iterate over every item in the list
for index, file_name in enumerate(fado_output_list):

    # Get the data
    wave_list, best_fit, norm_wave, norm_flux = readFADO(file_name)

    # Save the wavelength array from the first file
    if index == 0:
        reference_wave = wave_list

    # Check that all spectra use the same wavelength grid
    elif not np.allclose(wave_list, reference_wave):
        raise ValueError(
            f'Wavelength grid in {file_name} differs from the reference grid.'
        )

    # Store the results
    bin_indices.append(index)
    norm_waves.append(norm_wave)
    norm_fluxes.append(norm_flux)
    best_fit_spectra.append(best_fit)


# Convert lists to NumPy arrays
bin_indices = np.asarray(bin_indices, dtype=int)
norm_waves = np.asarray(norm_waves, dtype=float)
norm_fluxes = np.asarray(norm_fluxes, dtype=float)
best_fit_spectra = np.asarray(best_fit_spectra, dtype=float)


# Create the binary table
table = Table()

table['BIN'] = bin_indices
table['NORMWAV'] = norm_waves
table['NORMFLUX'] = norm_fluxes
table['FLUX'] = best_fit_spectra.tolist()


# Create the FITS HDUs
primary_hdu = fits.PrimaryHDU()

table_hdu = fits.BinTableHDU(table, name='BEST_FIT')

# Save the wavelength array in a separate extension
wave_hdu = fits.ImageHDU(
    data=reference_wave,
    name='WAVELENGTH'
)

# Write the FITS file
hdul = fits.HDUList([
    primary_hdu,
    table_hdu,
    wave_hdu
])

hdul.writeto(best_fit_file, overwrite=True)

print(f'Saved best-fit spectra to {best_fit_file}')