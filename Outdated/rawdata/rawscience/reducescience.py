#Run the setup script
exec(open("setup.py").read())

# This is a python program to reduce the science frames
print('Script running')
ysize = 1051
xsize = 400

# Science images

BIASframe = fits.open('../rawbias/specBIAS.fits')
BIAS = np.array(BIASframe[0].data)
FLATframe = fits.open('../rawflat/specflat.fits')
FLAT = np.array(FLATframe[0].data)

rawimages = ['ALDc200212.fits','ALDc200213.fits']
outnames = ['sub1.fits','sub2.fits']
centers = [107, 293]

#Read the raw file, subtract overscan, bias and divide by the flat
for n in range(0,1):
     spec = fits.open(rawimages[n])
     print('Info on file:')
     print(spec.info())
     specdata = np.array(spec[1].data)
     mean = np.mean(specdata[1033:ysize-5,0:xsize-1])
     specdata = specdata - mean
     print('Subtracted the median value of the overscan :',mean)
     specdata = (specdata-BIAS)/FLAT
     hdr = spec[0].header
     specdata1 = specdata[25:875,centers[n]-100:centers[n]+100] 

#crr here!


#Add and rotate
sub1 = fits.open(outnames[0])
sub2 = fits.open(outnames[1])
sum = sub1[0].data+sub2[0].data
rot = np.rot90(sum, k=3)
fits.writeto('spec1.fits',rot,hdr,overwrite=True)

print(centers[1])

#Arcframe
spec = fits.open('../rawarc/ALDc200214.fits')
print('Info on file:')
print(spec.info())
specdata = np.array(spec[1].data)
mean = np.mean(specdata[1033:ysize-5,0:xsize-1])
specdata = specdata - mean
print('Subtracted the median value of the overscan :',mean)
specdata = (specdata-BIAS)/FLAT
hdr = spec[0].header
center = int((centers[0]+centers[1])/2.)
specdata1 = specdata[25:875,center-100:center+100] 
rot = np.rot90(specdata1, k=3)
fits.writeto('arcsub.fits',rot,hdr,overwrite=True)

