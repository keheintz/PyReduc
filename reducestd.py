# This is a python program to reduce the science frames
print('Script running')

import numpy as np
from astropy.io import fits
import os as os
import glob
import sys

ysize = 2102
xsize = 500

BIASframe = fits.open('../rawbias/BIAS.fits')
BIAS = np.array(BIASframe[0].data)
FLATframe = fits.open('../rawflats/FLAT.fits')
FLAT = np.array(FLATframe[0].data)

rawimages = sorted(glob.glob("A*.fits"))
nframes = len(rawimages)
outnames = ['sub1.fits', 'sub2.fits']
#centers = [154, 340]
centers = [251, 273]

#Read the raw file, subtract overscan, bias and divide by the flat
for n in range(0,nframes):
     spec = fits.open(str(rawimages[n]))
     print('Info on file:')
     print(spec.info())
     specdata = np.array(spec[1].data)
     mean = np.mean(specdata[2066:ysize-5,0:xsize-1])
     specdata = specdata - mean
     print('Subtracted the median value of the overscan :',mean)
     specdata = (specdata-BIAS)/FLAT
     hdr = spec[0].header
     specdata1 = specdata[50:1750,centers[n]-100:centers[n]+100] 
     print(outnames[n])
     fits.writeto(outnames[n],specdata1,hdr,overwrite=True)

#Add and rotate
sub1 = fits.open(outnames[0])
sub2 = fits.open(outnames[1])
sum = (sub1[0].data+sub2[0].data)/2.
rot = np.rot90(sum, k=3)
fits.writeto('../std.fits',rot,hdr,overwrite=True)
#os.remove(outnames[0])
#os.remove(outnames[1])

#Arcframe
arclist = glob.glob("../rawarcs/*.fits")
specdata = np.zeros((ysize,xsize),float)
for frames in arclist:
    spec = fits.open(str(frames))
    data = spec[1].data
    if ((len(data[0,:]) != xsize) or (len(data[:,0]) != ysize)): sys.exit(frame + ' has wrong image size')
    specdata += data
mean = np.mean(specdata[2066:ysize-5,0:xsize-1])
specdata = specdata - mean
print('Subtracted the median value of the overscan :',mean)
specdata = (specdata-BIAS)/FLAT
hdr = spec[0].header
center = int((centers[0])/1.)
specdata1 = specdata[50:1750,center-100:center+100]
rot = np.rot90(specdata1, k=3)
hduout = fits.PrimaryHDU(rot)
hduout.header.extend(hdr, strip=True, update=True,
        update_first=False, useblanks=True, bottom=False)
hduout.header['DISPAXIS'] = 1
hduout.header['CRVAL1'] = 1
hduout.header['CRVAL2'] = 1
hduout.header['CRPIX1'] = 1
hduout.header['CRPIX2'] = 1
hduout.header['CRVAL1'] = 1
hduout.header['CRVAL1'] = 1
hduout.header['CDELT1'] = 1
hduout.header['CDELT2'] = 1
hduout.writeto("../arcsub_std.fits", overwrite=True)

