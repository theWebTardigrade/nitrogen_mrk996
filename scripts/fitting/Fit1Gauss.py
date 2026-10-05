'''
Created on Mon Sep 14 14:16:00 2026

@author: M. Pólvora Fonseca

Uses as reference the narrow map for amplitude of the line.
Fits a single Gaussian to a given line in the spectrum of a data cube.
Saves a map with the flux of the line.

'''

# --------------------------------- Libraries -------------------------------- #

# Import libraries
import numpy as np

from astropy.io import fits 
from astropy import units as u
from spectral_cube import SpectralCube


from lmfit.models import LinearModel, GaussianModel

import MUSEinstruwidth as MUSEinstruwidth



# ----------------------------------- Files ---------------------------------- #


# Target information
cube_file_path = '/home/polaris/nitrogen_haro11/data/Haro11_ec_all.rc.fits'

# Narrow Filter Map information
narrow_map_file_path = '/home/polaris/nitrogen_haro11/data/narrowFilterMaps/NFM_Ha.fits'

# Output directory
output_dir = '/home/polaris/nitrogen_haro11/results/1dFits/Ha/'


# ---------------------------- Read parameter file --------------------------- #

# Target information
redshift = 0.0206467

# Emission line information
line_name = 'Ha'
central_wav_rest = 6562.77
blue_lim_rest = 6550
red_lim_rest = 6570


# Fitting constraints
min_offset = -3.0
max_offset = 3.0

sigma_min = 0.8
sigma_max = 1.4

flux_min = 1e1
flux_max = 1e5



# --------------------- Rest to Observed frame conversion -------------------- #

central_wav_obs = central_wav_rest * (1 + redshift)
blue_lim_obs = blue_lim_rest * (1 + redshift)
red_lim_obs = red_lim_rest * (1 + redshift)

# -------------------------------- Access data ------------------------------- #

print("-------------- Reading cubes --------------")

# Read the data cube
file = fits.open(cube_file_path)
cube = SpectralCube.read(file[1])
var_cube = SpectralCube.read(file[2])

line_cube = cube.spectral_slab(blue_lim_obs, red_lim_obs)
var_line_cube = var_cube.spectral_slab(blue_lim_obs, red_lim_obs)
file.close()

# Dimensions
n_wave, ny, nx = line_cube.shape

wav_list = line_cube.spectral_axis.to(u.AA).value
x_list = range(nx)
y_list = range(ny)


# Read the narrow filter map
narrow_map = fits.open(narrow_map_file_path)[1].data


# -------------------------- Muse Instrumental Width ------------------------- #

instrumental_sigma = MUSEinstruwidth.MUSE_sigmas(MUSEinstruwidth.MUSE_AIP_gFWHM_lin_poly(), central_wav_obs)



# -------------------------------- Model Setup ------------------------------- #

mynan_policy = 'propagate'

gauss = GaussianModel(prefix="gauss", nan_policy=mynan_policy) # Gaussian line
linear = LinearModel(prefix="linear", nan_policy=mynan_policy) # Continuum

model = gauss + linear


# ------------------------------- Storage setup ------------------------------ #

labels = ['center_' + line_name, 
          'sigma_' + line_name, 
          'amp_' + line_name, 
          'vel_' + line_name,
          'disp_' + line_name, 
          'y0_' + line_name,
          'slope_' + line_name, 
          'redchi_' + line_name]
err_labels = ['err_' + item for item in labels]

# Empty map to store the results
results = {item: np.full((nx, ny), np.nan, dtype=float) for item in labels + err_labels}

# --------------------------------- Fit Line --------------------------------- #

print("-------------- Spectra fitting -----------")

for y in y_list:
    for x in x_list:


        # Check the narrow filter map
        narrow_flux = narrow_map[y, x]

        # Skip nan pixels
        if not np.isfinite(narrow_flux):
            continue

        # Skip pixels with low flux in the narrow filter map
        if narrow_flux <= flux_min:
            continue

            
        # Extract the spectrum
        spectrum = line_cube[:, y, x].value
        variance = var_line_cube[:, y, x].value


        # Setup the initial parameters for the fit
        pars = gauss.make_params(amplitude=float(narrow_map[y, x]),
                                    center=float(central_wav_obs),
                                    sigma=float(sigma_min))
        pars += linear.make_params(intercept=0, slope=0)


        # Setup the parameter constraints
        pars['gausssigma'].set(min=sigma_min, max=sigma_max)
        pars['gaussamplitude'].set(min=flux_min, max=flux_max)
        pars['gausscenter'].set(min=central_wav_obs + min_offset , 
                                max=central_wav_obs + max_offset)


        # Perform the fit
        try:
            out = model.fit(spectrum, pars, x=wav_list, weights=1/np.sqrt(variance))
        except Exception:
            continue

        # Store the results in the results dictionary
        results['center_' + line_name][x, y] = out.values['gausscenter']
        results['sigma_' + line_name][x, y] = out.values['gaussigma']
        results['amp_' + line_name][x, y] = out.values['gaussamplitude']
        results['y0_' + line_name][x, y] = out.values['linearintercept']
        results['slope_' + line_name][x, y] = out.values['linearslope']
        results['redchi_' + line_name][x, y] = out.redchi

        # Store the uncertainties in the results dictionary
        if out.params['gausscenter'].stderr is not None:
            results['err_center_' + line_name][x, y] = out.params['gausscenter'].stderr
        if out.params['gaussigma'].stderr is not None:
            results['err_sigma_' + line_name][x, y] = out.params['gaussigma'].stderr
        if out.params['gaussamplitude'].stderr is not None:
            results['err_amp_' + line_name][x, y] = out.params['gaussamplitude'].stderr
        if out.params['linearintercept'].stderr is not None:
            results['err_y0_' + line_name][x, y] = out.params['linearintercept'].stderr
        if out.params['linearslope'].stderr is not None:
            results['err_slope_' + line_name][x, y] = out.params['linearslope'].stderr




# ----------------------------- Plot Diagnostics ----------------------------- #



# ---------------------------- Calculate velocity ---------------------------- #


velocity_map = ( ( results["center_" + line_name] - central_wav_rest ) / central_wav_rest * (u.c.to("km/s").value) )
results["vel_" + line_name] = velocity_map

center_error = results[ "err_center_" + line_name ] 
results["err_vel_" + line_name] = ( center_error / central_wav_rest * u.c.to("km/s").value )


# ------------------------- Remove instrumental width ------------------------ #

observed_sigma = results[ "sigma_" + line_name ]
if instrumental_sigma is not None:
    intrinsic_sigma_squared = ( observed_sigma ** 2 - instrumental_sigma ** 2 )
    intrinsic_sigma_squared[ intrinsic_sigma_squared < 0 ] = np.nan 
    intrinsic_sigma = np.sqrt( intrinsic_sigma_squared)

# -------------- Convert intrinsic sigma to velocity dispersion -------------- #
    
    results["disp_" + line_name] = ( intrinsic_sigma / central_wav_rest * u.c.to("km/s").value )
    results["err_disp_" + line_name] = ( results["err_sigma_" + line_name] / central_wav_rest * u.c.to("km/s").value )

else:
    results["disp_" + line_name] = np.full((nx, ny), np.nan, dtype=float)
    results["err_disp_" + line_name] = np.full((nx, ny), np.nan, dtype=float)


# ------------------------- Save values as fits files ------------------------ #
print("Saving FITS maps")

for item in labels + err_labels:
    hdu = fits.PrimaryHDU(results[item])
    hdu.writeto(f'{output_dir}{item}.fits', overwrite=True)