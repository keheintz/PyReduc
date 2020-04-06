from pathlib import Path
import numpy as np

from numpy.polynomial.chebyshev import chebfit, chebval

import os as os

from matplotlib import pyplot as plt
from matplotlib import gridspec, rcParams, rc
from matplotlib.widgets import Cursor

from astropy.table import Table, Column
from astropy.io import fits
from astropy.stats import sigma_clip, gaussian_fwhm_to_sigma
from astropy.modeling.models import Gaussian1D, Chebyshev2D
from astropy.modeling.fitting import LevMarLSQFitter

from photutils.aperture import RectangularAperture, aperture_photometry

from skimage.feature import peak_local_max
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

import pandas as pd

import astroscrappy
import glob

fitter = LevMarLSQFitter()

def gaussian(x, mu, sig, amp, bg):
    return bg + amp*np.exp(-0.5*(x-mu)**2/sig**2)

print("setting done!")


