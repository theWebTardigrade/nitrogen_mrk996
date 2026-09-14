#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Sep 14 14:16:00 2026

@author: M. Pólvora Fonseca
Fit Gaussians to the lines

'''


# Import libraries
import numpy as np
import pandas as pd

from astropy.io import fits 
from astropy import units as u
from spectral_cube import SpectralCube

from astropy.coordinates import SkyCoord 


# Define info about target
cube_file_path = '/home/polaris/nitrogen_mrk996/data/Mrk996.fits'
redshift = 0.0054

line_list_file_path = '/home/polaris/nitrogen_mrk996/data/NarrowFiltersInput.csv'
output_path = '/home/polaris/nitrogen_mrk996/data'

# Read the file with the line list
lines_list = pd.read_csv(line_list_file_path, sep=',', comment='#', header=0)

# Open the cube
file = fits.open(cube_file_path)
cube = SpectralCube.read(file[1])


# Fit Gaussian per line
def fitGaussian(cube):
    return



# Loop over the lines in the file 
for line in list(lines_list.index):
    name = lines_list.iloc[line]['Name']

    # Get wavelength limits from CSV
    line_cen = lines_list.iloc[line]['lc']
    line_inf = lines_list.iloc[line]['l_linf'] 
    line_sup = lines_list.iloc[line]['l_lsup'] 
    bcont_inf = lines_list.iloc[line]['b_linf'] 
    bcont_sup = lines_list.iloc[line]['b_lsup'] 
    rcont_inf = lines_list.iloc[line]['r_linf'] 
    rcont_sup = lines_list.iloc[line]['r_lsup']