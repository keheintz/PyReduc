# This is a python program to make a specflat frame
print('Script running')
nframes = 35
ysize = 2102
xsize = 500

import astropy
import numpy
from astropy.io import fits

#Read in the raw flat frames and subtact mean of overscan region 
list = open('specflat.list')
bigflat = numpy.zeros((nframes,ysize,xsize),float)
BIASframe = fits.open('../rawbias/BIAS.fits')
BIAS = numpy.array(BIASframe[0].data)
for i in range(0,nframes):
   print('Image number:', i)
   rawflat = fits.open(str.rstrip(list.readline()))
   print('Info on file:')
   print(rawflat.info())
   data = numpy.array(rawflat[1].data)
   median = numpy.mean(data[2066:ysize-5,0:xsize-1])
   data = data - median
   print('Subtracted the median value of the overscan :',median)
   data = data - BIAS
   print('Subtracted the BIAS')
   bigflat[i-1,0:ysize-1,0:xsize-1] = data[0:ysize-1,0:xsize-1]
   norm = numpy.median(bigflat[i-1,100:400,100:1300])
   print('Normalised with the median of the frame :',norm)
   bigflat[i-1,:,:] = bigflat[i-1,:,:]/norm
list.close()

#Calculate flat is median at each pixel
medianflat = numpy.median(bigflat,axis=0)

#Normalise the flat field
lampspec = numpy.mean(medianflat,axis=1)
norm = medianflat*0.
for i in range(0,xsize-1):
   medianflat[:,i] = medianflat[:,i] / lampspec[:]

#Write out result to fitsfile
hdr = rawflat[0].header
fits.writeto('Flat.fits',medianflat,hdr,overwrite=True)
