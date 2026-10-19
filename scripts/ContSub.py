"""
Created on Tue Oct 13 2026

@author: M. Pólvora Fonseca

Continuum subtract the initial map. Interpolates the wavelenghts of FADO if necessary.

Outputs:
    - Continuum subtracted spectral cube as .fits file
"""

# --------------------------------- Libraries -------------------------------- #

import os
import glob

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u


# ----------------------------------- Files ---------------------------------- #

cube_file_path = '/data2/dafonseca/nitrogen_haro11/data/Haro11_ec_all.rc.fits'
tess_file_path = '/data2/dafonseca/data/powerbin_tessellation.txt'
FADO_tables_file_path = '/data2/dafonseca/FADOv1b_patch1.0/best/FADO_best_fit.fits'

contsub_file_path = '/data2/dafonseca/nitrogen_haro11/data/Haro11_contsub.fits'


# ----------------------------- Read tesselation ----------------------------- #

tess_data = np.loadtxt(tess_file_path, skiprows=1)

x = tess_data[:, 0].astype(int)
y = tess_data[:, 1].astype(int)
binNum = tess_data[:, 2].astype(int)


# --------------------------------- Read FADO -------------------------------- #


with fits.open(FADO_tables_file_path) as fado_file:

    # Read the best-fit spectra table
    fado_table = fado_file['BEST_FIT'].data

    # Read the wavelength grid
    fado_waves = np.asarray(fado_file['WAVELENGTH'].data, dtype=float)

    # Extract bin numbers and best-fit spectra
    fado_bins = np.asarray(fado_table['BIN'], dtype=int)

    fado_spectra = np.asarray(fado_table['FLUX'], dtype=float)

    fado_normflux = np.asarray(fado_table['NORMFLUX'], dtype=float)


# Create a dictionary linking each bin to its FADO spectrum
fado_lookup = {
    bin_number: i
    for i, bin_number in enumerate(fado_bins)
}



# --------------------------- Continuum subtraction -------------------------- #

# Iterate over the pixels in the tesselation (the others will be 0/Nan)

# Open the cube to memory
with fits.open(cube_file_path, memmap=True) as cube_file:

    # Original spectral and varience cube
    flux_cube = cube_file['DATA'].data
    variance_cube = cube_file['STAT'].data

    # Cube dimensions
    n_wave, n_y, n_x = flux_cube.shape


    spectral_wcs = WCS(cube_file['DATA'].header).spectral

    muse_waves = spectral_wcs.pixel_to_world(np.arange(n_wave)).to(u.AA).value


    # Create an cube with nans with the shape of the original MUSE cube
    contsub_cube = np.full(flux_cube.shape, np.nan, dtype=np.float32)



    # Iterate over the available tesselated spaxels
    for x_pixel, y_pixel, bin_number in zip(x, y, binNum):

        # Check that the spaxel is inside the cube
        if not (
            0 <= x_pixel < n_x
            and 0 <= y_pixel < n_y
        ):
            raise IndexError(
                f'Spaxel ({x_pixel}, {y_pixel}) is outside the cube.'
            )

        # Find the corresponding FADO spectrum
        if bin_number not in fado_lookup:
            raise KeyError(
                f'No FADO best-fit spectrum found for bin {bin_number}.'
            )
        

        fado_index = fado_lookup[bin_number]

        # Extract the FADO spectrum
        fado_spectrum = fado_spectra[fado_index].copy()

        # Un-normalize the fado spectrum
        fado_spectrum *= fado_normflux[fado_index]

        # Open the specific pixel in original cube and take out the spectrum
        muse_spectrum = np.array(flux_cube[:, y_pixel, x_pixel], dtype=float)

        # Check if the FADO wavelenghts are the same as the MUSE wavelenghts.
        # If not, we need to interpolate the FADO values to get the values at the MUSE wavelenghts
        # Interpolate the FADO continuum onto the MUSE wavelength grid
        valid_fado = (
            np.isfinite(fado_waves)
            & np.isfinite(fado_spectrum)
        )

        if np.count_nonzero(valid_fado) < 2:
            continue

        wave = fado_waves[valid_fado]
        continuum = fado_spectrum[valid_fado]

        # np.interp requires an increasing wavelength grid
        sort_order = np.argsort(wave)

        wave = wave[sort_order]
        continuum = continuum[sort_order]

        # Interpolate only within the FADO wavelength coverage.
        # Outside that coverage, leave the result as NaN.
        interpolated_continuum = np.interp(
            muse_waves,
            wave,
            continuum,
            left=np.nan,
            right=np.nan
        )

        # ------------------------- Scale the FADO continuum ------------------------- #

        # Define the normalization wavelength range
        norm_mask = (
            (muse_waves >= 7000)
            & (muse_waves <= 7200)
            & np.isfinite(muse_spectrum)
            & np.isfinite(interpolated_continuum)
        )

        # Calculate the scale factor using the median flux ratio
        if np.count_nonzero(norm_mask) > 0:

            valid_muse = muse_spectrum[norm_mask]
            valid_fado = interpolated_continuum[norm_mask]

            # Avoid division by zero
            nonzero = valid_fado != 0

            if np.any(nonzero):

                scale_factor = np.median(valid_muse[nonzero] / valid_fado[nonzero])

                # Scale the interpolated FADO continuum
                interpolated_continuum *= scale_factor

            else:
                raise ValueError(
                    f'FADO continuum is zero throughout the normalization '
                    f'range for bin {bin_number}.'
                )

        else:
            raise ValueError(
                f'No valid wavelength points between 7000 and 7200 Å '
                f'for bin {bin_number}.'
            )

        # ------------------------- Subtract the continuum --------------------------- #

        contsub_spectrum = (
            muse_spectrum - interpolated_continuum
        )

        # Store the continuum-subtracted spectrum
        contsub_cube[:, y_pixel, x_pixel] = contsub_spectrum

    # ----------------------------- Save the cube ----------------------------- #

    # Copy the original FITS structure and replace DATA.
    # The STAT extension, headers, and other extensions are preserved.
    output_hdul = fits.HDUList([
        hdu.copy() for hdu in cube_file
    ])

    output_hdul['DATA'].data = contsub_cube

    output_hdul.writeto(
        contsub_file_path,
        overwrite=True
    )

    output_hdul.close()

print(f'Continuum-subtracted cube saved to:\n{contsub_file_path}')


