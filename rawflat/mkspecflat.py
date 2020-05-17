# This is a python program to make a specflat frame
print('Script running')
import astropy
import numpy
from astropy.io import fits
import glob
import sys

ysize = 1051
xsize = 400

#Read in the raw flat frames and subtact mean of overscan region 
list = glob.glob("A*fits")
nframes = len(list)
bigflat = numpy.zeros((nframes,ysize,xsize),float)
BIASframe = fits.open('../rawbias/BIAS.fits')
BIAS = numpy.array(BIASframe[0].data)
i = 0
for frame in list:
   print('Image:', frame)
   rawflat = fits.open(str(frame))
   print('Info on file:')
   print(rawflat.info())
   data = numpy.array(rawflat[1].data)
   if ((len(data[0,:]) != xsize) or (len(data[:,0]) != ysize)): sys.exit(frame + ' has wrong image size')
   median = numpy.mean(data[1033:ysize-5,0:xsize-1])
   data = data - median
   print('Subtracted the median value of the overscan :',median)
   data = data - BIAS
   print('Subtracted the BIAS')
   bigflat[i-1,0:ysize-1,0:xsize-1] = data[0:ysize-1,0:xsize-1]
   norm = numpy.median(bigflat[i-1,100:200,100:300])
   print('Normalised with the median of the frame :',norm)
   bigflat[i-1,:,:] = bigflat[i-1,:,:]/norm
   i+=1

#Calculate flat is median at each pixel
medianflat = numpy.median(bigflat,axis=0)

#Normalise the flat field
lampspec = numpy.mean(medianflat,axis=1)
norm = medianflat*0.
for i in range(0,xsize-1):
   medianflat[:,i] = medianflat[:,i] / lampspec[:]

#Write out result to fitsfile
hdr = rawflat[0].header
fits.writeto('FLAT.fits',medianflat,hdr,overwrite=True)
