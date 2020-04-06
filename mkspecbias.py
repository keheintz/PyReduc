# This is a python program to make a BIAS frame
print 'Script running'
nframes = 44
ysize = 2102
xsize = 400

import astropy
import numpy
from string import rstrip
from astropy.io import fits

#Read in the raw bias frames and subtact mean of overscan region 
list = open('specbias.list')
bigbias = numpy.zeros((nframes,ysize,xsize),float)
#bigbias = numpy.zeros((nframes,3,3))
for i in range(0,nframes):
   print 'Image number:', i
   rawbias = fits.open(rstrip(list.readline()))
   print 'Info on file:'
   print rawbias.info()
   data = numpy.array(rawbias[1].data)
   median = numpy.mean(data[2070:ysize-5,0:xsize-1])
   data = data - median
   print 'Subtracted the median value of the overscan :',median
   bigbias[i-1,0:ysize-1,0:xsize-1] = data[0:ysize-1,0:xsize-1]
list.close()

##Calculate bias is median at each pixel
medianbias = numpy.median(bigbias,axis=0)

#Write out result to fitsfile
hdr = rawbias[0].header
fits.writeto('specBIAS.fits',medianbias,hdr,overwrite=True)