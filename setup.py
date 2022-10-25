from pathlib import Path 
import numpy as np

from numpy.polynomial.chebyshev import chebfit, chebval
from numpy.polynomial import Chebyshev

import os 
import sys

from matplotlib import pyplot as plt
from matplotlib import gridspec, rcParams, rc
from matplotlib.widgets import Cursor
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from astropy.table import Table, Column
from astropy.table import QTable
from astropy.io import fits
from astropy.stats import sigma_clip, gaussian_fwhm_to_sigma
from astropy.modeling.models import Gaussian1D, Chebyshev2D
from astropy.modeling.fitting import LevMarLSQFitter

from photutils.aperture import RectangularAperture, aperture_photometry

from skimage.feature import peak_local_max
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

import astroscrappy
import glob

#from PyAstronomy import pyasl

from datetime import datetime
import locale

locale.setlocale(locale.LC_TIME, "en_GB")

import pandas as pd

from tabulate import tabulate

FONTSIZE = 12 # Change it on your computer if you wish.
rcParams.update({'font.size': FONTSIZE})

fitter = LevMarLSQFitter()

def gaussian(x, mu, sig, amp, bg):
    return bg + amp*np.exp(-0.5*(x-mu)**2/sig**2)

def pprint_df(dframe):
    print(tabulate(dframe, headers='keys', tablefmt='psql', showindex=False))

#%%
DATAPATH = Path('./')

#Make database folder if it doesn't exist already
newpath = r'./database' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

DISPAXIS = 1  # 1 = line = python_axis_1 // 2 = column = python_axis_0
COMPIMAGE = DATAPATH/'arcsub.fits' 
STDCOMPIMAGE = DATAPATH/'arcsub_std.fits' 
STDIMAGE  = DATAPATH/'std.fits'
OBJIMAGE  = DATAPATH/'obj.fits'

#Fitter
LINE_FITTER = LevMarLSQFitter()

#Detector
#Preferable get this from header. For now just hard-wire the numbers
GAIN = 0.16 #electron/ADU
RON = 4.3 #Electrons

# Parameters for IDENTIFY
FITTING_MODEL_ID = 'Chebyshev'
ORDER_ID = 6 
NSUM_ID = 10
FWHM_ID = 2.5 # rough guess of FWHM of lines in IDENTIFY (pixels)
IDtolerance = 5. #Tolerance for deleting lines in fitting arcdata.

# Parameters for REIDENTIFY
FITTING_MODEL_REID = 'Chebyshev' # 2-D fitting function
ORDER_SPATIAL_REID = 3
ORDER_WAVELEN_REID = 5
STEP_REID = 5  # Reidentification step size in pixels (spatial direction)
NSUM_REID = 5 
TOL_REID = 25 # tolerence to lose a line in pixels

# Parameters for APALL (sky fitting and aperture extract after sky subtraction)
## parameters for finding aperture
NSUM_AP = 20
FWHM_AP = 20
STEP_AP = 20  # Recentering step size in pixels (dispersion direction)

#Weight function for optimal extraction
def gaussweight(x, mu, sig):
    return np.exp(-0.5*(x-mu)**2/sig**2) / (np.sqrt(2.*np.pi)*sig)

## parameters for sky fitting
FITTING_MODEL_APSKY = 'Chebyshev'
ORDER_APSKY = 3
SIGMA_APSKY = 3
ITERS_APSKY = 5
## parameters for aperture tracing
FITTING_MODEL_APTRACE = 'Chebyshev'
ORDER_APTRACE = 3
SIGMA_APTRACE = 1
ITERS_APTRACE = 5
# The fitting is done by SIGMA_APTRACE-sigma ITERS_APTRACE-iters clipped on the
# residual of data. 

# Parameters for SENSFUNCTION
FITTING_MODEL_SF = 'Chebyshev'
ORDER_SF = 9 

#%%
lamphdu = fits.open(COMPIMAGE)
stdlamphdu = fits.open(STDCOMPIMAGE)
objhdu = fits.open(OBJIMAGE)
stdhdu = fits.open(STDIMAGE)
lampimage = lamphdu[0].data
stdlampimage = stdlamphdu[0].data
objimage  = objhdu[0].data
stdimage  = stdhdu[0].data

#check for NaNs
if np.isnan(objhdu[0].data).any(): print('There are NaNs in the object image. This may course a crash.')
if np.isnan(stdhdu[0].data).any(): print('There are NaNs in the std image. This may course a crash.')
if np.isnan(lamphdu[0].data).any(): print('There are NaNs in the arc image. This may course a crash.')
if np.isnan(stdlamphdu[0].data).any(): print('There are NaNs in the std-arc image. This may course a crash.')

if lampimage.shape != objimage.shape:
    raise ValueError('lamp and obj images should have same sizes!')

if DISPAXIS == 2:
    lampimage = lampimage.T
    objimage = objimage.T
elif DISPAXIS != 1:
    raise ValueError('DISPAXIS must be 1 or 2 (it is now {:d})'.format(DISPAXIS))

OBJNEXP = objhdu[0].header['NEXP']
OBJEXPTIME = objhdu[0].header['EXPTIME']*OBJNEXP
OBJAIRMASS = objhdu[0].header['AIRMASS']
OBJNAME = objhdu[0].header['OBJECT']
STDEXPTIME = stdhdu[0].header['EXPTIME']
STDAIRMASS = stdhdu[0].header['AIRMASS']
STDNAME = stdhdu[0].header['OBJECT']
# Now python axis 0 (Y-direction) is the spatial axis 
# and 1 (X-direciton) is the wavelength (dispersion) axis.
N_SPATIAL, N_WAVELEN = np.shape(lampimage)
N_REID = N_SPATIAL//STEP_REID # No. of reidentification
N_AP = N_WAVELEN//STEP_AP # No. of aperture finding

# ``peak_local_max`` calculates the peak location using maximum filter:
#   med1d_max = scipy.ndimage.maximum_filter(med1d, size=10, mode='constant')
# I will use this to show peaks in a primitive manner.
MINSEP_PK = 5   # minimum separation of peaks
MINAMP_PK = 0.01 # fraction of minimum amplitude (wrt maximum) to regard as peak
NMAX_PK = 50

#For setting up the output fits-spectra
def fake_multispec_data(arrlist):
   # takes a list of 1-d numpy arrays, which are
   # to be the 'bands' of a multispec, and stacks them
   # into the format expected for a multispec.  As of now
   # there can only be a single 'aperture'.
   return np.expand_dims(np.array(arrlist), 1)


print("setting done!")



