#!/usr/bin/env python3

# -*- coding: utf-8 -*-

'''
Created on Mon Sep 14 14:16:00 2026

@author: M. Pólvora Fonseca

Spectrum image with the emission lines marked
'''

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from astropy import units as u
from spectral_cube import SpectralCube
from astropy.coordinates import SkyCoord
from astropy.io import fits

# Define info about target
cube_file_path = '/home/polaris/nitrogen_mrk996/data/Mrk996.fits'
redshift = 0.0054

# Define point of spectrum
centerCoordsObj = SkyCoord(ra=21.89848459, dec=-6.32641114, unit='deg', frame='icrs')
radius_arcsec = 0.15 * u.arcsec

# Define the file with the lines
linesFilePath = '/home/polaris/nitrogen_mrk996/data/NarrowFiltersInput.csv'
linesList = pd.read_csv(linesFilePath, sep=',', comment='#', header=0)
linesNames = linesList['Name']
linesWav = linesList['lc']

# Open the cube
file = fits.open(cube_file_path)
data_file = file[1]
cube = SpectralCube.read(data_file)

# Define position of spectrum
center_pos = [177, 197]
point_spectrum = cube[:, center_pos[1], center_pos[0]]

# Spectral axis
spectral_axis = cube.spectral_axis.to_value(u.AA)

# Convert spectrum to values
spectrum = point_spectrum.value
spectrum[~np.isfinite(spectrum)] = np.nan

# Find points for the beginning and end of the plots
idx = np.round(np.linspace(0, len(spectral_axis) - 1, 6)).astype(int)

minEdge = spectral_axis[0]
maxEdge = spectral_axis[-1]

edges = [(minEdge, 5800),
         (6050, 7700),
         (7700, maxEdge)]

# Observed wavelengths of the emission lines
linesWav_obs = linesWav.to_numpy() * (1 + redshift)



# Create figure with 5 subplots
fig, axes = plt.subplots(len(edges), 1, figsize=(30, 10), sharey=True)


# Plot each wavelength section
for i, ax in enumerate(axes):

    # Wavelength limits for this subplot
    wav_min = edges[i][0]
    wav_max = edges[i][1]

    # Select spectrum in this wavelength range
    mask = (spectral_axis >= wav_min) & (spectral_axis <= wav_max)
    ax.plot(spectral_axis[mask], spectrum[mask], color='black')

    # Mark emission lines in this wavelength range
    for name, wav in zip(linesNames, linesWav_obs):

        if wav_min <= wav <= wav_max:
            ax.axvline(wav, linestyle='--', alpha=0.7)
            ax.text(
                wav + 2,
                0.95,
                name,
                rotation=90,
                transform=ax.get_xaxis_transform(),
                verticalalignment='top'
            )

    ax.set_yscale('log')
    ax.set_xlim(wav_min, wav_max)
    ax.set_ylim(7e1, 1e3)
    ax.grid(alpha=0.2)

    if i < 4:
        ax.tick_params(labelbottom=True)

axes[-1].set_xlabel('Observed wavelength (Å)')
axes[2].set_ylabel('Flux')

plt.tight_layout()



# Save figure
output_path = '/home/polaris/nitrogen_mrk996/data/spectrum_lines.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')

plt.show()

file.close()

