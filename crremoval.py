# import cosmics
import astroscrappy
import glob
from astropy.io import fits
import os
import matplotlib.pyplot as pl

# Path to folder with science files
object_name = "testdata"

for nn in glob.glob(object_name+"/*cr*"):
    os.remove(nn)

files = glob.glob(object_name+"/*z.fits")
for n in files:
    try:
        fitsfile = fits.open(str(n))
    except:
        continue
    try:
        fitsfile[0].header['IMAGECAT'] = fitsfile[0].header['IMAGECAT']
    except:
        continue

    if fitsfile[0].header['IMAGECAT'] == 'SCIENCE':
        print('Removing cosmics from file: '+n+'...')
    
        gain = fitsfile[0].header['GAIN']
        ron = fitsfile[0].header['RDNOISE']
        frac = 0.01
        objlim = 15
        sigclip = 5
        niter = 5
        
        crmask, clean_arr = astroscrappy.detect_cosmics(fitsfile[0].data, sigclip=sigclip, sigfrac=frac, objlim=objlim, cleantype='medmask', niter=niter, sepmed=True, verbose=True)

        # Replace data array with cleaned image
        fitsfile[0].data = clean_arr

        # Try to retain info of corrected pixel if extension is present.
        try:
            fitsfile[2].data[crmask] = 16 #Flag value for removed cosmic ray
        except:
            print("No bad-pixel extension present. No flag set for corrected pixels")

        # Update file
        fitsfile.writeto(n[:-5]+"cr.fits", output_verify='fix')
