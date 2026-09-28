'''
Created on Mon Sep 14 14:16:00 2026

@author: M. Pólvora Fonseca
Fit Gaussians to the lines

'''

# --------------------------------- Libraries -------------------------------- #

# Import libraries
import numpy as np
import pandas as pd

from astropy.io import fits 
from astropy import units as u
from spectral_cube import SpectralCube

from astropy.coordinates import SkyCoord 

from lmfit.models import LinearModel, GaussianModel

from datetime import datetime


# -------------------------------- Parameters -------------------------------- #


# Define info about target
cube_file_path = '/home/polaris/nitrogen_haro11/data/Haro11_ec_all.rc.fits'
redshift = 0.0206467







# ---------------------------------- Models ---------------------------------- #

mynan_policy = 'propagate'

gauss = GaussianModel(prefix="gauss", nan_policy=mynan_policy)
linear = LinearModel(prefix="linear", nan_policy=mynan_policy) # Continuum


print("-------------- Spectra fitting -----------")
print("        " +     str(datetime.now()))
print("------------------------------------------")








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