#!/usr/bin/env python3

# -*- coding: utf-8 -*-

'''
Created on Thu 17 Sep 21:16:00 2026

@author: M. Pólvora Fonseca

Tesselate the our image 
'''

# Import Libraries
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import vorbin
from vorbin.voronoi_2d_binning import voronoi_2d_binning


#-----------------------------------------------------------------------------


# Import the Halpha map and its variance
data_file = fits.open('/home/polaris/nitrogen_mrk996/data/haro11/narrowFieldMaps/NFM_Ha.fits')

variance_file = fits.open('/home/polaris/nitrogen_mrk996/data/haro11/narrowFieldMapsNoise/NFM_noise_Ha.fits')

image_data = data_file[0].data
variance_data = variance_file[0].data

data_file.close()
variance_file.close()


# Coordinates of every pixel
y_indices, x_indices = np.indices(image_data.shape)

# Flatten everything
x = x_indices.flatten()
y = y_indices.flatten()
signal = image_data.flatten()
variance = variance_data.flatten()

# Remove pixels containing NaN/inf values
valid = (
    np.isfinite(x)
    & np.isfinite(y)
    & np.isfinite(signal)
    & np.isfinite(variance)
    & (variance >= 0)
)

x = x[valid]
y = y[valid]
signal = signal[valid]
variance = variance[valid]

# Coordinates passed to PowerBin
xy = np.column_stack((x, y))


# Target S/N
target_sn = 100


binNum, x_gen, y_gen, x_bar, y_bar, sn, nPixels, scale = voronoi_2d_binning(x, y, signal, variance, target_sn, pixelsize=0.025,  plot=True, quiet=0)

np.savetxt('voronoi_2d_binning.txt', np.column_stack([x, y, binNum]), fmt='%10.6f %10.6f %8i')