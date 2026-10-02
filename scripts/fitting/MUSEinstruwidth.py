#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:42:09 2021

@author: anamonrealibero
"""
import numpy as np
# Routines by Peter to correct for instrumental resolution
# Gaussian FWHM in polynomial form, for linear sampling
# with 1.25 Angstrom/pixel
def MUSE_AIP_gFWHM_lin_poly():
    return np.array([4.44, -0.000354422, -9.74002e-09, 2.83627e-12])

# array of lambdas, array of output FWHMs
def MUSE_get_FWHMs(coeffs, lambdas):
    poly = np.zeros_like(lambdas)
    for i in range(0, len(coeffs)):
        poly += coeffs[i] * lambdas**i
    return poly

# array of lambdas, array of output sigmas
def MUSE_sigmas(coeffs, lambdas):
    return MUSE_get_FWHMs(coeffs, lambdas) / 2.35482
    