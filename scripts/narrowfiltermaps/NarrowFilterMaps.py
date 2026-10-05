"""
Created on Thu Sep 10 17:51:00 2026

@author: M. Pólvora Fonseca
Make maps of relatively isolated lines simulating narrrow filters.
The results is a fits file with both the data and the noise.

"""


# --------------------------------- Libraries -------------------------------- #
import numpy as np
import pandas as pd
import os

from astropy import units as u
from spectral_cube import SpectralCube
from astropy.io import fits 


# -------------------------------- Parameters -------------------------------- #
# Define info about target
cube_file_path = '/home/polaris/nitrogen_haro11/data/Haro11_ec_all.rc.fits'
redshift = 0.020598

line_list_file_path = '/home/polaris/nitrogen_haro11/data/NarrowFiltersInput.csv'
output_path = '/home/polaris/nitrogen_haro11/data/narrowFieldMaps'
os.makedirs(output_path, exist_ok=True)


# Open the cube
file = fits.open(cube_file_path)
cube_data = SpectralCube.read(file[1])
cube_variance = SpectralCube.read(file[2])

# ------------------------ Narrow Filter Map Function ------------------------ #

# Function to create the narrow filter maps
def create_narrow_filter_maps(
    redshift,
    data,
    variance,
    line_sup,
    line_inf,
    bcont_sup,
    bcont_inf,
    rcont_sup,
    rcont_inf ):


    # Create the line map
    line_subcube = data.spectral_slab(line_inf*(1+redshift)* u.AA, line_sup*(1+redshift)* u.AA)
    line_map = line_subcube.moment(order=0)

    line_var_subcube = variance.spectral_slab(line_inf*(1+redshift)* u.AA, line_sup*(1+redshift)* u.AA)
    line_var_map = line_var_subcube.moment(order=0)


    # Create the blue continuum map
    blue_cont_subcube = data.spectral_slab(bcont_inf*(1+redshift)* u.AA, bcont_sup*(1+redshift)* u.AA)
    blue_cont_map = blue_cont_subcube.moment(order=0)

    blue_var_subcube = variance.spectral_slab(bcont_inf*(1+redshift)* u.AA, bcont_sup*(1+redshift)* u.AA)
    blue_var_map = blue_var_subcube.moment(order=0)


    # Create the red continuum map
    red_cont_subcube = data.spectral_slab(rcont_inf*(1+redshift)* u.AA, rcont_sup*(1+redshift)* u.AA)
    red_cont_map = red_cont_subcube.moment(order=0)

    red_var_subcube = variance.spectral_slab(rcont_inf*(1+redshift)* u.AA, rcont_sup*(1+redshift)* u.AA)
    red_var_map = red_var_subcube.moment(order=0)


    # Size of the subcubes
    line_size = (line_sup - line_inf)
    blue_cont_size = (bcont_sup - bcont_inf)
    red_cont_size = (rcont_sup - rcont_inf)

    blue_factor = line_size / blue_cont_size
    red_factor = line_size / red_cont_size

    # Create the average continuum map
    mean_cont_map = ((blue_cont_map * blue_factor) + (red_cont_map * red_factor)) / 2.0

    # Create the line continuum subtracted map
    line_contsub_map = line_map - mean_cont_map

    # Propagate variance
    variance_map = ( line_var_map + (blue_factor**2 * blue_var_map + red_factor**2 * red_var_map) / 4.0)

    return line_contsub_map, variance_map


# ------------------------------- Apply in list ------------------------------ #

# Read the file with the line list
lines_list = pd.read_csv(line_list_file_path, sep=',', comment='#', header=0)

# Loop over the lines in the file 
for line in list(lines_list.index):
    name = lines_list.iloc[line]['Name']
    print(name)

    # Get wavelength limits from CSV 
    line_inf = lines_list.iloc[line]['l_linf'] 
    line_sup = lines_list.iloc[line]['l_lsup'] 
    bcont_inf = lines_list.iloc[line]['b_linf'] 
    bcont_sup = lines_list.iloc[line]['b_lsup'] 
    rcont_inf = lines_list.iloc[line]['r_linf'] 
    rcont_sup = lines_list.iloc[line]['r_lsup']

    line_map, variance_map = create_narrow_filter_maps(
        redshift,
        cube_data,
        cube_variance,
        line_sup,
        line_inf,
        bcont_sup,
        bcont_inf,
        rcont_sup,
        rcont_inf )


    output_file = os.path.join( output_path, f"NFM_{name}.fits" ) 

    # Save everything in a fits file
    primary_hdu = fits.PrimaryHDU()

    data_hdu = fits.ImageHDU(
        data=line_map.value,
        header=line_map.wcs.to_header(),
        name="DATA"
    )

    noise_hdu = fits.ImageHDU(
        data=variance_map.value,
        header=variance_map.wcs.to_header(),
        name="VAR"
    )

    hdul = fits.HDUList([
        primary_hdu,
        data_hdu,
        noise_hdu
    ])

    hdul.writeto(output_file, overwrite=True)