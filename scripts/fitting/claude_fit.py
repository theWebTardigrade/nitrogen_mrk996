#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fits a single Gaussian + linear continuum to a given line in every spaxel of a
MUSE data cube, using a narrow-filter map as reference for the initial amplitude
and as a spaxel selector.

Usage:  python fit_line_gauss.py parameters.csv

The parameter file is a CSV with columns  Param,Value,Descript  ('#' = comment).
Required: myfilecube, mypath, myblue, myred, mypathout, mypathNF, myz, minsig,
          maxsig, minlam, maxlam, myslope, raiz, mylcen, mylinename, myimaNF,
          myscale, fmin, fmax
Optional: fixvel (0/1), imavref, mypathvref   (velocity map for initial centre)
"""

# --------------------------------- Libraries -------------------------------- #
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from astropy.io import fits
from astropy import units as u
from scipy.constants import c  # Light speed in m/s
from spectral_cube import SpectralCube
from lmfit.models import LinearModel, GaussianModel

import MUSEinstruwidth as MUSEinstruwidth

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ---------------------------- Read parameter file ---------------------------- #
param_file = sys.argv[1]
df = pd.read_csv(param_file, comment="#", names=["Param", "Value", "Descript"])


def get_str(name):
    return df.loc[df['Param'] == name, 'Value'].values[0].strip()


def get_float(name):
    return float(df.loc[df['Param'] == name, 'Value'].astype(float).values[0])


myfilecube = get_str("myfilecube")
mypath = get_str("mypath")
mypathout = get_str("mypathout")
mypathNF = get_str("mypathNF")
myimaNF = get_str("myimaNF")
raiz = get_str("raiz")
mylinename = get_str("mylinename")

myblue, myred = get_float("myblue"), get_float("myred")     # observed frame (A)
myz = get_float("myz")
minsig, maxsig = get_float("minsig"), get_float("maxsig")
minlam, maxlam = get_float("minlam"), get_float("maxlam")
myslope = get_float("myslope")
mylcen = get_float("mylcen")                                # rest frame (A)
myscale = get_float("myscale")
fmin, fmax = get_float("fmin"), get_float("fmax")
print(myz)

# Optional: take the initial line centre from a reference velocity map
try:
    fixvel = int(df.loc[df['Param'] == "fixvel", "Value"].astype(int).values[0])
    imavref = get_str("imavref")
    mypathvref = get_str("mypathvref")
    myv_ref_ini = fits.getdata(os.path.join(mypath, mypathvref, imavref))
except Exception:
    fixvel = 0

os.makedirs(mypathout, exist_ok=True)
mypathplot = os.path.join(mypathout, "plots")
os.makedirs(mypathplot, exist_ok=True)


# -------------------------------- Read the data ------------------------------ #
print("-- Reading Line with NF for initial condi --")
print("        " + str(datetime.now()))
print("------------------------------------------")

# Narrow-filter map: initial amplitude + spaxel selection
myf_ini = np.abs(fits.getdata(os.path.join(mypathNF, myimaNF)) * myscale)

# Cube (MUSE layout: ext 1 = DATA, ext 2 = STAT/variance)
with fits.open(os.path.join(mypath, myfilecube)) as hdul:
    wave_full = SpectralCube.read(hdul[1]).spectral_axis.to(u.AA).value
    sel = (wave_full >= myblue) & (wave_full <= myred)
    mylamsel = wave_full[sel]
    mydatasel = np.asarray(hdul[1].data[sel], dtype=float)
    myerrsel = np.sqrt(np.asarray(hdul[2].data[sel], dtype=float))  # std. dev.
    myhdr = hdul[1].header

mysize = mydatasel.shape[1:]          # (ny, nx)

# Rough continuum level per spaxel for the initial intercept
myintercept = np.nanmedian(mydatasel, axis=0)

l0 = np.array([mylcen])
mysig = 1.2
mysiginstru = MUSEinstruwidth.MUSE_sigmas(
    MUSEinstruwidth.MUSE_AIP_gFWHM_lin_poly(), l0[0] * (1 + myz))


# ---------------------------------- Model ----------------------------------- #
mynan_policy = 'propagate'
mod1 = GaussianModel(prefix="mod1", nan_policy=mynan_policy)   # Line
mod2 = LinearModel(prefix="mod2", nan_policy=mynan_policy)     # Continuum
mod = mod1 + mod2

mylabels = ['l' + mylinename, 's' + mylinename, 'f' + mylinename,
            'v' + mylinename, 'disp' + mylinename,
            'y0_' + raiz, 'slope_' + raiz, 'redchi_' + raiz]
myerrlabels = ['e' + item for item in mylabels]
arr = {item: np.full(mysize, np.nan, dtype=float)
       for item in mylabels + myerrlabels}


# --------------------------------- Fitting ---------------------------------- #
print("-------------- Spectra fitting -----------")
print("        " + str(datetime.now()))
print("------------------------------------------")

myaxismain = [0.12, 0.25, 0.82, 0.72]
myaxisresi = [0.12, 0.10, 0.82, 0.15]

for i1 in range(mysize[0]):
    for i2 in range(mysize[1]):
        f_ini = myf_ini[i1, i2]
        if not (np.isfinite(f_ini) and f_ini > fmin):
            continue

        l0ini = l0 * (1 + myz)
        lo, hi = minlam, maxlam
        if fixvel == 1:
            l0ini = l0 * (1 + myv_ref_ini[i1, i2] / c)
            lo, hi = -0.05, 0.05

        pars = mod1.make_params(amplitude=float(f_ini),
                                center=float(l0ini[0]), sigma=float(mysig))
        pars += mod2.make_params(intercept=float(myintercept[i1, i2]),
                                 slope=myslope)
        pars['mod1sigma'].set(min=minsig, max=maxsig)
        pars['mod1amplitude'].set(min=fmin, max=fmax)
        pars['mod1center'].set(min=l0ini[0] + lo, max=l0ini[0] + hi)

        x = mylamsel
        y = mydatasel[:, i1, i2]
        yerr = myerrsel[:, i1, i2]
        try:
            out = mod.fit(y, pars, x=x, weights=1 / yerr)
        except Exception:
            continue

        arr['l' + mylinename][i1, i2] = out.values['mod1center']
        arr['s' + mylinename][i1, i2] = out.values['mod1sigma']
        arr['f' + mylinename][i1, i2] = out.values['mod1amplitude']
        arr['y0_' + raiz][i1, i2] = out.values['mod2intercept']
        arr['slope_' + raiz][i1, i2] = out.values['mod2slope']
        arr['redchi_' + raiz][i1, i2] = out.redchi

        # Uncertainties (stderr can be None if the covariance failed)
        for key, par in [('el' + mylinename, 'mod1center'),
                         ('es' + mylinename, 'mod1sigma'),
                         ('ef' + mylinename, 'mod1amplitude'),
                         ('ey0_' + raiz, 'mod2intercept'),
                         ('eslope_' + raiz, 'mod2slope')]:
            err = out.params[par].stderr
            if err is not None:
                arr[key][i1, i2] = err

        try:
            myerror = out.eval_uncertainty()
        except Exception:
            myerror = np.zeros(len(x), dtype=float)

        # Diagnostic plot: every 25th spaxel only
        if (i1 % 25 == 0) and (i2 % 25 == 0):
            f1 = plt.figure()
            plt.axes(myaxismain)
            plt.plot(x, y, 'k-', label='Data')
            plt.plot(x, out.init_fit, 'k--', label='Condi. Ini.')
            plt.plot(x, out.best_fit, 'g-', label='Best fit')
            plt.fill_between(x, out.best_fit - myerror, out.best_fit + myerror,
                             alpha=0.2, color='g')
            comps = out.eval_components()
            plt.plot(x, comps['mod1'], 'b--', label='Line')
            plt.plot(x, comps['mod2'], 'r--', label='Continuum')
            plt.legend(loc=1)
            xmin, xmax = plt.xlim(x[0], x[-1])
            try:
                ymin, ymax = plt.ylim(-0.1 * np.nanmax(y), np.nanmax(y) * 1.1)
            except Exception:
                ymin, ymax = plt.ylim(-0.1, 1.1)
            plt.xlabel(r'Wavelength ($\AA$)')
            plt.ylabel(r'Flux (10$^{-20}$ erg/s/cm${^2}$/$\AA$)')
            myi1 = '{:03d}'.format(i1)
            myi2 = '{:03d}'.format(i2)
            plt.text(xmin + 0.05 * (xmax - xmin), ymin + 0.95 * (ymax - ymin),
                     "Spaxel (" + myi1 + "," + myi2 + ")")

            plt.axes(myaxisresi)
            plt.plot(x, out.residual, 'k-')
            plt.xlim(x[0], x[-1])
            plt.xlabel(r'Wavelength ($\AA$)')

            f1.savefig(os.path.join(mypathplot,
                                    "Spec_" + myi1 + "_" + myi2 + ".pdf"))
            plt.close(f1)
            print("End of Fit " + myi1 + " " + myi2)


# ------------------- Velocity and velocity dispersion maps ------------------- #
kms = c / 1.E3
for ii, item in enumerate([mylinename]):
    arr['v' + item] = (arr['l' + item] - l0[ii]) / l0[ii] * kms
    arr['ev' + item] = arr['el' + item] / l0[ii] * kms

    # Remove instrumental width in quadrature (negative -> 0)
    tempo = arr['s' + item] ** 2 - mysiginstru ** 2
    tempo[tempo < 0] = 0
    arr['s' + item] = np.sqrt(tempo)
    arr['disp' + item] = arr['s' + item] / l0[ii] * kms
    arr['edisp' + item] = arr['es' + item] / l0[ii] * kms


# --------------------------------- Save maps -------------------------------- #
# Spatial WCS only (drop the spectral axis keywords)
from astropy.wcs import WCS
whdr = WCS(myhdr).celestial.to_header()

for item in mylabels + myerrlabels:
    hdu = fits.PrimaryHDU(arr[item], header=whdr)
    hdu.writeto(os.path.join(mypathout, item + ".fits"), overwrite=True)

print("-------------- End script ----------------")
print("        " + str(datetime.now()))
print("------------------------------------------")