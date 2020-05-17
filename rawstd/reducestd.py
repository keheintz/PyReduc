# This is a python program to reduce the science frames
print('Script running')

import numpy as np
from astropy.io import fits
import os as os
import glob

ysize = 1051
xsize = 400

BIASframe = fits.open('../rawbias/BIAS.fits')
BIAS = np.array(BIASframe[0].data)
FLATframe = fits.open('../rawflat/FLAT.fits')
FLAT = np.array(FLATframe[0].data)

rawimages = glob.glob("A*.fits")
nframes = len(rawimages)
outnames = ['sub1.fits','sub2.fits']
centers = [293, 107]

#Read the raw file, subtract overscan, bias and divide by the flat
for n in range(0,nframes):
     spec = fits.open(str(rawimages[n]))
     print('Info on file:')
     print(spec.info())
     specdata = np.array(spec[1].data)
     mean = np.mean(specdata[1033:ysize-5,0:xsize-1])
     specdata = specdata - mean
     print('Subtracted the median value of the overscan :',mean)
     specdata = (specdata-BIAS)/FLAT
     hdr = spec[0].header
     specdata1 = specdata[25:875,centers[n]-100:centers[n]+100] 
     print(outnames[n])
     fits.writeto(outnames[n],specdata1,hdr,overwrite=True)

#Add and rotate
sub1 = fits.open(outnames[0])
sub2 = fits.open(outnames[1])
sum = sub1[0].data+sub2[0].data
rot = np.rot90(sum, k=3)
fits.writeto('../std.fits',rot,hdr,overwrite=True)
os.remove(outnames[0])
os.remove(outnames[1])


#Arcframe
arclist = glob.glob("../rawarc/*.fits")
specdata = np.zeros((ysize,xsize),float)
for frames in arclist:
    spec = fits.open(str(frames))
    data = spec[1].data
    if ((len(data[0,:]) != xsize) or (len(data[:,0]) != ysize)): sys.exit(frame + ' has wrong image size')
    specdata += data
mean = np.mean(specdata[1033:ysize-5,0:xsize-1])
specdata = specdata - mean
print('Subtracted the median value of the overscan :',mean)
specdata = (specdata-BIAS)/FLAT
hdr = spec[0].header
center = int((centers[0]+centers[1])/2.)
specdata1 = specdata[25:875,center-100:center+100]
rot = np.rot90(specdata1, k=3)
hduout = fits.PrimaryHDU(rot)
hduout.header.extend(hdr, strip=True, update=True,
        update_first=False, useblanks=True, bottom=False)
hduout.header['DISPAXIS'] = 1
hduout.writeto("../arcsub_std.fits", overwrite=True)

