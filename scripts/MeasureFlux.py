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
from spectral_cube import SpectralCube

from astropy.coordinates import SkyCoord 
from photutils.aperture import SkyCircularAperture, AperturePhotometry


# Define the file with the lines
linesFilePath = '/home/polaris/nitrogen_mrk996/data/NarrowFiltersInput.csv'
linesList = pd.read_csv(linesFilePath, sep=',', comment='#', header=0)

# Define folder with contsub maps
folderPath = '/home/polaris/nitrogen_mrk996/data/narrowFieldMaps'
mapsPrefix = 'NFM_'

# Define folder for output 
ouputPath = '/home/polaris/nitrogen_mrk996/data'
outputName = 'NFM_Fluxes.csv'



# Define the zone of flux measured, in this case a circular aperture
centerCoords = [[21.89848459, -6.32641114]]
raCoords = [ x[0] for x in centerCoords]
decCoords = [ x[1] for x in centerCoords]
centerCoordsObj = SkyCoord(ra=raCoords, dec=decCoords, unit='deg', frame='icrs')
radius_arcsec = 0.15  * u.arcsec

aperture = SkyCircularAperture(centerCoordsObj, r=radius_arcsec)



# Loop over the lines in the file 
for line in list(linesList.index):
    lineName = linesList.iloc[line]['Name']

    # Access the narrow filter map for that line 
    mapPath = folderPath + '/' + mapsPrefix + lineName + '.fits'
    narrowFilterMap = fits.open(mapPath)
    narrowFilterData = narrowFilterMap[0].data
    narrowFilterMap.close()

    # Measure the flux in the previously defined regios 
    phot = AperturePhotometry(narrowFilterData, aperture)

