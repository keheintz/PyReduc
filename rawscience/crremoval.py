import astroscrappy
import glob

import os as os

from astropy.io import fits

# This is a python program to run cosmic ray removal on science frames

# detector 
gain = 0.16 
ron = 4.3 
frac = 0.01
objlim = 15
sigclip = 5
niter = 5

# Path to folder with science frames
for nn in glob.glob("*crr*.fits"):
    os.remove(nn)
    print(nn)

files = glob.glob("*.fits")
for n in files:
    fitsfile = fits.open(str(n))
    print(n)
    print('Removing cosmics from file: '+n+'...')
    
        
    crmask, clean_arr = astroscrappy.detect_cosmics(fitsfile[1].data, sigclip=sigclip, sigfrac=frac, objlim=objlim, cleantype='medmask', niter=niter, sepmed=True, verbose=True)

# Replace data array with cleaned image
    fitsfile[1].data = clean_arr

# Try to retain info of corrected pixel if extension is present.
    try:
        fitsfile[2].data[crmask] = 16 #Flag value for removed cosmic ray
    except:
        print("No bad-pixel extension present. No flag set for corrected pixels")

# Update file
    fitsfile.writeto("crr"+n, output_verify='fix')
