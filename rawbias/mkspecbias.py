# This is a python program to make a BIAS frame
print('Script running')

import astropy
import numpy
from astropy.io import fits
import glob
import sys

ysize = 1051
xsize = 400

#Read in the raw bias frames and subtact mean of overscan region 
list = glob.glob("A*fits")
nframes = len(list)
bigbias = numpy.zeros((nframes,ysize,xsize),float)
i = 0
for frame in list:
   print('Image:', frame)
   rawbias = fits.open(str(frame))
   print('Info on file:')
   print(rawbias.info())
   data = numpy.array(rawbias[1].data)
   if ((len(data[0,:]) != xsize) or (len(data[:,0]) != ysize)): sys.exit(frame + ' has wrong image size')
   mean = numpy.mean(data[1033:ysize-5,0:xsize-1])
   data = data - mean
   print('Subtracted the median value of the overscan :',mean)
   bigbias[i-1,0:ysize-1,0:xsize-1] = data[0:ysize-1,0:xsize-1]
   i+=1

##Calculate bias is median at each pixel
medianbias = numpy.median(bigbias,axis=0)

#Write out result to fitsfile
hdr = rawbias[0].header
fits.writeto('BIAS.fits',medianbias,hdr,overwrite=True)
