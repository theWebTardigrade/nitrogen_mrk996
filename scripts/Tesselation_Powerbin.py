#!/usr/bin/env python3

# -*- coding: utf-8 -*-

'''
Created on Thu 17 Sep 21:16:00 2026

@author: M. Pólvora Fonseca

Tesselate the our image 
'''

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from powerbin import PowerBin


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
target_sn = 1e3


# Capacity function
def capacity_spec(index):
    """
    Return (S/N)^2 for a proposed bin.

    Assumes variance_data contains the variance:
        sigma^2
    """

    bin_signal = np.sum(signal[index])
    bin_variance = np.sum(variance[index])

    if bin_variance <= 0:
        return 0

    sn = bin_signal / np.sqrt(bin_variance)

    return sn**2


# Perform the binning
pow = PowerBin(
    xy,
    capacity_spec,
    target_capacity=target_sn**2
)


# Plot as S/N rather than (S/N)^2
pow.plot(capacity_scale='sqrt', ylabel='S/N')

plt.show(block=True)
