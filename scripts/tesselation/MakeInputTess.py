"""
Created on Wed Sep 30 13:16 2026

@author: M. Pólvora Fonseca
Creates a text file with x, y positions, pixel values and noise
"""

# --------------------------------- Libraries -------------------------------- #

import numpy as np
from astropy.io import fits

# -------------------------------- Parameters -------------------------------- #

file_path = '/home/polaris/nitrogen_haro11/data/narrowFieldMaps/NFM_cont_6250_6400.fits'
output_file = '/home/polaris/nitrogen_haro11/data/cont_input_tess.csv'

file = fits.open(file_path)

data = file['DATA'].data
variance = file['VARIANCE'].data

noise = np.sqrt(variance)


x_values = list(range(data.shape[1])) 
y_values = list(range(data.shape[0]))

# ------------------------------- File Creation ------------------------------ #

# Filters out nanpixels and 

with open(output_file, 'w') as f:
    f.write('x,y,signal,noise\n')

    for x_pixel in range(data.shape[1]):
        for y_pixel in range(data.shape[0]):

            if np.isnan(data[y_pixel, x_pixel]):
                continue

            if data[y_pixel, x_pixel]<0:
                continue

            f.write(f'{x_pixel:.6f},{y_pixel:.6f},{data[y_pixel, x_pixel]:.2f},{noise[y_pixel, x_pixel]:.2f}\n')

file.close()