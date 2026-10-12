"""
Created on Thu Oct 08 2026

@author: M. Pólvora Fonseca

Create the spectrum input files for FADO

Output file contains:
    wavelength (Å), flux (erg/s*cm**2), error (erg/s*cm**2), 0
"""

# --------------------------------- Libraries -------------------------------- #

import os
import numpy as np
from astropy.io import fits

# ----------------------------------- Data ----------------------------------- #

cube_file_path = '/home/polaris/nitrogen_haro11/data/Haro11_ec_all.rc.fits'
tess_file_path = '/home/polaris/nitrogen_haro11/data/powerbin_tessellation.txt'
output_path = '/home/polaris/nitrogen_haro11/data/FADOInput'

# -------------------------- Load the cube to memory ------------------------- #

with fits.open(cube_file_path, memmap=True) as file:
    flux_cube = file['DATA'].data
    variance_cube = file['STAT'].data
    header = file['DATA'].header.copy()

    # ---------------------- Import the tessellation regions ----------------- #

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

    # ----------------------------- Wavelength -------------------------------- #

    crval3 = header['CRVAL3']
    crpix3 = header['CRPIX3']
    cdelt3 = header['CD3_3']

    nwave = flux_cube.shape[0]

    wavelength = crval3 + (np.arange(nwave) + 1 - crpix3) * cdelt3

    # ----------------------------- Average and save -------------------------- #

    os.makedirs(output_path, exist_ok=True)

    # Determine the number of digits needed for bin filenames
    bin_width = len(str(np.max(np.abs(binNum))))

    # Track minimum and maximum flux across all bins
    global_min_flux = np.inf
    global_max_flux = -np.inf

    for bin_number, pixels in bin_pixels.items():

        print(f"Processing bin {bin_number} ({len(pixels)} pixels)")

        # Accumulate flux and variance without storing all pixel spectra
        flux_sum = np.zeros(nwave, dtype=np.float64)
        flux_count = np.zeros(nwave, dtype=np.int64)
        variance_sum = np.zeros(nwave, dtype=np.float64)

        # Get spectrum for every pixel in this bin
        for xi, yi in pixels:

            spectrum = flux_cube[:, yi, xi]
            variance = variance_cube[:, yi, xi]

            # Average only finite flux measurements
            valid_flux = np.isfinite(spectrum)
            flux_sum[valid_flux] += spectrum[valid_flux]
            flux_count[valid_flux] += 1

            # Sum finite variances
            valid_variance = np.isfinite(variance)
            variance_sum[valid_variance] += variance[valid_variance]

        # ------------------------- Average spectrum -------------------------- #

        average_flux = np.full(nwave, np.nan, dtype=np.float64)

        valid_channels = flux_count > 0
        average_flux[valid_channels] = (
            flux_sum[valid_channels] / flux_count[valid_channels]
        )

        # ----------------------- Error on the mean --------------------------- #

        average_variance = np.full(nwave, np.nan, dtype=np.float64)

        valid_error = valid_channels & (flux_count > 0)

        average_variance[valid_error] = (
            variance_sum[valid_error] / flux_count[valid_error]**2
        )

        average_error = np.sqrt(average_variance)

        # ----------------------- Global minimum and maximum ------------------ #

        finite_flux = average_flux[np.isfinite(average_flux)]

        if finite_flux.size > 0:
            bin_min_flux = np.min(finite_flux)
            bin_max_flux = np.max(finite_flux)

            global_min_flux = min(global_min_flux, bin_min_flux)
            global_max_flux = max(global_max_flux, bin_max_flux)

        # ------------------------------ Output ------------------------------- #

        output_data = np.column_stack((
            wavelength,
            average_flux,
            average_error,
            np.zeros_like(wavelength)
        ))

        save_bin_number = str(bin_number).zfill(bin_width)

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

    # ----------------------- Final flux statistics --------------------------- #

    print("\n----------- Flux statistics across all bins -----------")

    if np.isfinite(global_min_flux) and np.isfinite(global_max_flux):
        print(f"Minimum flux across all bins: {global_min_flux:.8e}")
        print(f"Maximum flux across all bins: {global_max_flux:.8e}")
    else:
        print("No finite flux values were found.")

print("\nFinished.")