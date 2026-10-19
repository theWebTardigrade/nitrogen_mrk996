"""
Created on Thu Oct 08 2026

@author: M. Pólvora Fonseca

Create the spectrum input files for FADO.

Outputs:
    - file per spectra with wavelength (Å), flux (erg/s*cm**2), error (erg/s*cm**2), 0
    - Minimum and maximum flux across all bins.
    - list_spectra.txt containing the spectrum filename, 92.1, and 2.5.
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

    # ----------------------------- Wavelength ------------------------------- #

    crval3 = header['CRVAL3']
    crpix3 = header['CRPIX3']
    cdelt3 = header['CD3_3']

    nwave = flux_cube.shape[0]

    wavelength = crval3 + (np.arange(nwave) + 1 - crpix3) * cdelt3

    # ----------------------------- Output setup ----------------------------- #

    os.makedirs(output_path, exist_ok=True)

    # Automatically determine the number of digits for the bin filenames.
    bin_width = max(len(str(abs(bi))) for bi in np.unique(binNum))

    # Create a dictionary: bin number -> indices of its pixels.
    bin_pixels = {}

    for bi in np.unique(binNum):
        mask = binNum == bi
        bin_pixels[bi] = np.column_stack((x[mask], y[mask]))

    # Store filenames for list_spectra.txt.
    spectrum_filenames = []

    # Global minimum and maximum flux across all bins.
    global_min_flux = np.inf
    global_max_flux = -np.inf

    # -------------------------- Process each bin ---------------------------- #

    for bin_number, pixels in bin_pixels.items():

        print(f"Processing bin {bin_number} ({len(pixels)} pixels)")

        # Process one bin at a time to limit memory usage.
        spectra = []
        variances = []

        for xi, yi in pixels:

            spectrum = flux_cube[:, yi, xi]
            variance = variance_cube[:, yi, xi]

            spectra.append(spectrum)
            variances.append(variance)

        spectra = np.asarray(spectra)
        variances = np.asarray(variances)

        # ------------------------- Average spectrum ------------------------ #

        # Count finite flux measurements at each wavelength.
        n_pixels = np.sum(np.isfinite(spectra), axis=0)

        # Calculate the mean without warnings for wavelengths with no data.
        summed_flux = np.nansum(spectra, axis=0)

        average_flux = np.divide(
            summed_flux,
            n_pixels,
            out=np.full(nwave, np.nan),
            where=n_pixels > 0
        )

        # ----------------------- Error on the mean -------------------------- #

        # For independent measurements:
        #
        # Var(mean) = sum(Var_i) / N^2

        summed_variance = np.nansum(variances, axis=0)

        average_variance = np.divide(
            summed_variance,
            n_pixels**2,
            out=np.full(nwave, np.nan),
            where=n_pixels > 0
        )

        average_error = np.sqrt(average_variance)

        # ---------------------- Global flux extrema ------------------------- #

        # Ignore NaNs when determining the global minimum and maximum.
        finite_flux = average_flux[np.isfinite(average_flux)]

        if finite_flux.size > 0:

            bin_min_flux = np.min(finite_flux)
            bin_max_flux = np.max(finite_flux)

            global_min_flux = min(global_min_flux, bin_min_flux)
            global_max_flux = max(global_max_flux, bin_max_flux)

        # ------------------------------ Output ------------------------------ #

        # Replace NaNs with zero in the output.
        average_flux = np.where(np.isnan(average_flux), 0.0, average_flux)
        average_error = np.where(np.isnan(average_error), 0.0, average_error)

        output_data = np.column_stack((
            wavelength,
            average_flux,
            average_error,
            np.zeros(nwave, dtype=int)
        ))

        # Dynamically pad bin numbers with zeros.
        save_bin_number = str(bin_number).zfill(bin_width)

        output_filename = f'Spec_{save_bin_number}.txt'

        output_file = os.path.join(
            output_path,
            output_filename
        )

        # Wavelengths have two decimal places.
        # Fluxes and errors have scientific notation.
        np.savetxt(
            output_file,
            output_data,
            fmt=['%.2f', '%.8e', '%.8e', '%d']
        )

        print(f"  Saved: {output_file}")

        spectrum_filenames.append(output_filename)

    # ----------------------- Print global flux extrema ---------------------- #

    print("\n-------------- Flux statistics --------------")

    if np.isfinite(global_min_flux) and np.isfinite(global_max_flux):

        print(f"Minimum flux across all bins: {global_min_flux:.8e}")
        print(f"Maximum flux across all bins: {global_max_flux:.8e}")

    else:
        print("No finite flux values found across the bins.")

# --------------------------- Create spectrum list --------------------------- #

list_file = os.path.join(output_path, 'list_spectra.txt')

with open(list_file, 'w') as f:

    for filename in spectrum_filenames:
        f.write(f"{filename}\t92.1\t2.5\n")

print(f"\nSpectrum list saved: {list_file}")
print(f"Number of spectra created: {len(spectrum_filenames)}")
print("Finished.")