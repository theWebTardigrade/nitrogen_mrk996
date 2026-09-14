#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 17:51:00 2026

@author: M. Pólvora Fonseca
Make maps of relatively isolated lines simulating narrrow filters.

"""

# Import Libraries
import numpy as np
import pandas as pd
import os

from astropy import units as u
from spectral_cube import SpectralCube
from astropy.io import fits 



# Define info about target
cube_file_path = '/home/polaris/nitrogen_mrk996/data/Mrk996.fits'
redshift = 0.0054

line_list_file_path = '/home/polaris/nitrogen_mrk996/data/NarrowFiltersInput.csv'
output_path = '/home/polaris/nitrogen_mrk996/data/narrowFieldMaps'
os.makedirs(output_path, exist_ok=True)


# Open the cube
file = fits.open(cube_file_path)
cube = SpectralCube.read(file[1])


# Function to create the narrow filter maps
def create_narrow_filter_maps(
    redshift,
    cube,
    line_sup,
    line_inf,
    bcont_sup,
    bcont_inf,
    rcont_sup,
    rcont_inf ):


    # Create the line map
    line_subcube = cube.spectral_slab(line_inf*(1+redshift)* u.AA, line_sup*(1+redshift)* u.AA)
    line_map = line_subcube.moment(order=0)

    # Create the blue continuum map
    blue_cont_subcube = cube.spectral_slab(bcont_inf*(1+redshift)* u.AA, bcont_sup*(1+redshift)* u.AA)
    blue_cont_map = blue_cont_subcube.moment(order=0)

    # Create the red continuum map
    red_cont_subcube = cube.spectral_slab(rcont_inf*(1+redshift)* u.AA, rcont_sup*(1+redshift)* u.AA)
    red_cont_map = red_cont_subcube.moment(order=0)

    # Size of the subcubes
    line_size = (line_sup - line_inf)
    blue_cont_size = (bcont_sup - bcont_inf)
    red_cont_size = (rcont_sup - rcont_inf)

    # Create the average continuum map
    mean_cont_map = (blue_cont_map * line_size/blue_cont_size 
                        + red_cont_map * line_size/red_cont_size) / 2.0

    # Create the line continuum subtracted map
    line_contsub_map = line_map - mean_cont_map

    return line_contsub_map




# Read the file with the line list
lines_list = pd.read_csv(line_list_file_path, sep=',', comment='#', header=0)



# Loop over the lines in the file 
for line in list(lines_list.index):
    name = lines_list.iloc[line]['Name']

    # Get wavelength limits from CSV 
    line_inf = lines_list.iloc[line]['l_linf'] 
    line_sup = lines_list.iloc[line]['l_lsup'] 
    bcont_inf = lines_list.iloc[line]['b_linf'] 
    bcont_sup = lines_list.iloc[line]['b_lsup'] 
    rcont_inf = lines_list.iloc[line]['r_linf'] 
    rcont_sup = lines_list.iloc[line]['r_lsup']

    line_map = create_narrow_filter_maps(
        redshift,
        cube,
        line_sup,
        line_inf,
        bcont_sup,
        bcont_inf,
        rcont_sup,
        rcont_inf )

    # Skip line if it was saturated
    if line_map is None: 
        continue

    output_file = os.path.join( output_path, f"NFM_{name}.fits" ) 
    line_map.write(output_file, overwrite=True)
