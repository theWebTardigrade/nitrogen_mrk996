'''
Calculate the flux within a given circular aperture 
'''


# Import libraries
import numpy as np

from astropy.io import fits 
from astropy import units as u

from spectral_cube import SpectralCube



# Import the cube
file = fits.open('Mrk996.fits')
cube = SpectralCube.read(file[1])


# Mask the cube
center_pos = (177, 197)  # pixels
radius_arcsec = 0.15  # arcseconds

spatial_pixel_scale = 0.025 # arcseconds/pixel
radius_pixels = radius_arcsec / spatial_pixel_scale



def calculate_flux(
    cube,
    upper_limit_line_wavelength,
    lower_limit_line_wavelength,
    continuum_wavelength
):
    # Define the circular aperture
    yy, xx = np.indices(cube.shape[1:])

    aperture_mask = (
        (xx - center_pos[0])**2 +
        (yy - center_pos[1])**2
    ) <= radius_pixels**2

   
    # Measure the flux of the continuum
    channel = np.argmin(np.abs(cube.spectral_axis - continuum_wavelength))
    continuum_image =  cube[channel].value
    continuum_flux = np.nansum(continuum_image[aperture_mask])


    # Extend the flux to the size of the line
    line_width = upper_limit_line_wavelength - lower_limit_line_wavelength 
    continuum_line = continuum_flux * line_width


    # Measure the flux of the line
    line_cube = cube.spectral_slab(lower_limit_line_wavelength * u.angstrom , upper_limit_line_wavelength * u.angstrom)
    line_mom0 = line_cube.moment0()

    line_flux = np.nansum(line_mom0[aperture_mask])


    # Calculate actual flux of the line
    line_flux -= continuum_line

    return line_flux

