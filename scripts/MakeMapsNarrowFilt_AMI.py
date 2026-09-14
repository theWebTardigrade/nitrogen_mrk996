#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:25:43 2018

@author: Ana Monreal Ibero
Make maps of relatively isolated lines simulating narrrow filters

"""

import os
import sys
import numpy as np
import pandas as pd
from mpdaf.obj import Cube, Image

def domap(myl_linf,myl_lsup,myb_linf,myb_lsup,myr_linf,myr_lsup, myz):
    # Add flux in the line, sum the number of pixels
    myimal = mycube.select_lambda(myl_linf*(1+myz),myl_lsup*(1+myz)).sum(axis=0)
    pixl = np.logical_and(mylam > myl_linf, mylam < myl_lsup).sum()
    # Add flux in the blue continuum, sum the number of pixels
    myimab = mycube.select_lambda(myb_linf*(1+myz),myb_lsup*(1+myz)).sum(axis=0)
    pixb = np.logical_and(mylam > myb_linf, mylam < myb_lsup).sum()
    # Add flux in the red continuum, sum the number of pixels
    myimar = mycube.select_lambda(myr_linf*(1+myz),myr_lsup*(1+myz)).sum(axis=0)
    pixr = np.logical_and(mylam > myr_linf, mylam < myr_lsup).sum()
    myima = (myimal - (myimab*pixl/pixb + myimar*pixl/pixr) / 2.) * lamstep
    return myima

myfilecube = sys.argv[1]  #  e.g.  "NGC5253.rc.fits"
mylinesfile = sys.argv[2] # e.g. "NarrowFiltersInput.csv"
myz = float(sys.argv[3])  # e.g. -0.000046
mypathout = sys.argv[4] # "MyNFs"
mypathin = sys.argv[5] # Place with the cube
mypath = sys.argv[5] # Place where I write my pathout"."





os.chdir(mypath)


if not os.path.exists(os.path.join(mypath,mypathout)):
    os.makedirs(os.path.join(mypath,mypathout))

mycube = Cube(os.path.join(mypathin,myfilecube))
lamstart = mycube.wave.get_crval()
lamstep = mycube.wave.get_step()
npix = mycube.shape[0]
mylam = lamstart + np.arange(0,npix) * lamstep
print(mylam[0],mylam[-1])
mycube.wave.get_range()
#myhdr = mycube['DATA'].header
# Create an empty image
myima = mycube[0,:,:].clone(data_init=np.zeros)


#mynames= ["Name","lc","l_linf","l_lsup","b_linf","b_lsup",
#          "r_linf","r_lsup","Comment"]
df = pd.read_csv(os.path.join(mypath,mylinesfile), sep=",", comment="#",
                 header=0)
#df.columns = mynames


for item in list(df.index):
   print(df.iloc[item]['Name'])
   myima = domap(df.iloc[item]['l_linf'],df.iloc[item]['l_lsup'],
                 df.iloc[item]['b_linf'],df.iloc[item]['b_lsup'],
                 df.iloc[item]['r_linf'],df.iloc[item]['r_lsup'], myz)
   myimafile = "flu_" + df.iloc[item]['Name'] + ".fits"
   myima.write(os.path.join(mypathout,myimafile))
   

