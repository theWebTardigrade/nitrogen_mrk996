#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Sep 14 14:16:00 2026

@author: M. Pólvora Fonseca
Calculate the flux within a given circular aperture (can be used for multiple circular apertures)

'''

# Import libraries
import numpy as np
import pandas as pd

from astropy.io import fits 
from astropy import units as u
from astropy.wcs import WCS
from spectral_cube import SpectralCube

from astropy.coordinates import SkyCoord 
from photutils.aperture import SkyCircularAperture, aperture_photometry


# Define the file with the lines
linesFilePath = '/home/polaris/nitrogen_mrk996/data/NarrowFiltersInput.csv'
linesList = pd.read_csv(linesFilePath, sep=',', comment='#', header=0)

# Define folder with contsub maps
folderPath = '/home/polaris/nitrogen_mrk996/data/narrowFieldMaps'
mapsPrefix = 'NFM_'

# Define folder for output 
outputFolderPath = '/home/polaris/nitrogen_mrk996/data/'
outputName = 'NFM_Fluxes.csv'
outputPath = outputFolderPath + outputName


# Define the zone of flux measured, in this case a circular aperture
centerCoords = [[21.89848459, -6.32641114]]
raCoords = [ x[0] for x in centerCoords]
decCoords = [ x[1] for x in centerCoords]
centerCoordsObj = SkyCoord(ra=raCoords, dec=decCoords, unit='deg', frame='icrs')
radius_arcsec = 0.15  * u.arcsec

aperture = SkyCircularAperture(centerCoordsObj, r=radius_arcsec)


# Define data for save file
linesNames = linesList['Name']
fluxes = []
fluxes_err = []


# Loop over the lines in the file 
for line in list(linesList.index):
    lineName = linesList.iloc[line]['Name']
    print(f"Processing {lineName}")

    # Access the narrow-filter map for that line 
    mapPath = folderPath + '/' + mapsPrefix + lineName + '.fits' 

    try: 
        with fits.open(mapPath) as hdul: 
            narrowFilterData = hdul[0].data 
            wcs = hdul[0].header 
    except FileNotFoundError: 
        print(f" Map not found: {mapPath}") 
        fluxes.append(np.nan) 
        fluxes_err.append(np.nan) 
        continue 

    # Create WCS from the FITS header 
    wcs = WCS(wcs) 
    # Perform aperture photometry 
    phot = aperture_photometry( narrowFilterData, aperture, wcs=wcs ) 

    # Extract aperture sum 
    flux = phot['aperture_sum'][0] 
    fluxes.append(flux)


# Create a dataFrame with the information
data = {'name': linesNames,
        'flux': fluxes
        }

df = pd.DataFrame(data)

# Save everything to .csv file
df.to_csv(outputPath, index=False)

print(f'Saved {outputName} to {outputPath}')
