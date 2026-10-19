"""
Created on Mon Oct 19 19:46:00 2026

@author: M. Pólvora Fonseca

Makes a map of the continuum.
The results is a fits file with both the data and the noise.

"""


# --------------------------------- Libraries -------------------------------- #
import numpy as np
import os

from astropy import units as u
from spectral_cube import SpectralCube
from astropy.io import fits 


# -------------------------------- Parameters -------------------------------- #
# Define info about target
cube_file_path = '/home/polaris/nitrogen_haro11/data/Haro11_ec_all.rc.fits'
redshift = 0.020598

output_path = '/home/polaris/nitrogen_haro11/data/narrowFieldMaps/'
os.makedirs(output_path, exist_ok=True)


min_wav = 7000
max_wav = 7200


# Open the cube
file = fits.open(cube_file_path)
cube_data = SpectralCube.read(file[1])
cube_variance = SpectralCube.read(file[2])

# --------------------------- Get Data and Variance -------------------------- #

data_subcube = cube_data.spectral_slab(min_wav * u.AA, max_wav * u.AA)
variance_subcube = cube_variance.spectral_slab(min_wav * u.AA, max_wav * u.AA)

flux_data = data_subcube.unmasked_data[:].value
variance_data = variance_subcube.unmasked_data[:].value
noise_data = np.sqrt(variance_data)

# ------------------------ Filter and calculate median ----------------------- #

# Number of finite spectral pixels in each spaxel
n_flux = np.sum(np.isfinite(flux_data), axis=0)
n_variance = np.sum(np.isfinite(variance_data), axis=0)
n_noise = np.sum(np.isfinite(noise_data), axis=0)

# Create maps with the same size as the original with nans
continuum_map = np.full(flux_data.shape[1:], np.nan)
variance_map = np.full(variance_data.shape[1:], np.nan)
noise_map = np.full(noise_data.shape[1:], np.nan)


# Only calculate median where at least one finite value exists
good_flux = n_flux > 0
good_variance = n_variance > 0
good_noise = n_noise > 0


# Take the median
continuum_map[good_flux] = np.nanmedian(
    flux_data[:, good_flux],
    axis=0
)

variance_map[good_variance] = np.nanmedian(
    variance_data[:, good_variance],
    axis=0
)

noise_map[good_noise] = np.nanmedian(
    noise_data[:, good_noise],
    axis=0
)

# ------------------------- Save results in fits file ------------------------ #

output_file = os.path.join( output_path, f"Cont_{min_wav}_{max_wav}.fits" ) 

wcs_header = data_subcube.wcs.celestial.to_header()

# Save everything in a fits file
primary_hdu = fits.PrimaryHDU()

data_hdu = fits.ImageHDU(
    data=continuum_map,
    header=wcs_header,
    name="DATA"
)

noise_hdu = fits.ImageHDU(
    data=variance_map,
    header=wcs_header,
    name="VARIANCE"
)

hdul = fits.HDUList([
    primary_hdu,
    data_hdu,
    noise_hdu
])

hdul.writeto(output_file, overwrite=True)