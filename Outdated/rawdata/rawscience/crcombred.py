#Run the setup script
exec(open("setup.py").read())

# This is a python program to run cosmic ray removal on science frames

# Path to folder with science frames
sciencefolder = "../rawscience/"
for nn in glob.glob(sciencefolder+"*_crr.fits"):
    os.remove(nn)

files = glob.glob(sciencefolder+"AL*.fits")
for n in files:
    fitsfile = fits.open(str.rstrip(n))
    fitsname = n.replace('.fits','')
    print('Got here')
    fitsfile[0].header['IMAGECAT'] = fitsfile[0].header['IMAGECAT']

    print(n)
    #if fitsfile[0].header['IMAGECAT'] == 'SCIENCE':
    print('Removing cosmics from file: '+n+'...')

    gain = 0.16 # LOOK UP fitsfile[0].header['GAIN']
    ron = 4.3 # LOOK UP fitsfile[0].header['RDNOISE']
    frac = 0.01
    objlim = 15
    sigclip = 5
    niter = 5

    crmask, clean_arr = astroscrappy.detect_cosmics(fitsfile[1].data, sigclip=sigclip, sigfrac=frac, objlim=objlim, cleantype='medmask', niter=niter, sepmed=True, verbose=True)

    # Replace data array with cleaned image
    fitsfile[1].data = clean_arr

    # Try to retain info of corrected pixel if extension is present.
    try:
        fitsfile[2].data[crmask] = 16 #Flag value for removed cosmic ray
    except:
        print("No bad-pixel extension present. No flag set for corrected pixels")

    # Update file
    fitsfile.writeto(fitsname+"_crr.fits", output_verify='fix')

# This is a python program to reduce the science frames
print('Reducing spectra running')
ysize = 1051
xsize = 400

BIASframe = fits.open('../rawbias/specBIAS.fits')
BIAS = np.array(BIASframe[0].data)
FLATframe = fits.open('../rawflat/specflat.fits')
FLAT = np.array(FLATframe[0].data)

rawimages = ['ALDc200212_crr.fits','ALDc200213_crr.fits']
outnames = ['sub1.fits','sub2.fits']
centers = [107, 293]

#Read the raw file, subtract overscan, bias and divide by the flat
for n in range(0,2):
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
     print(outnames[n])
     fits.writeto(outnames[n],specdata1,hdr,overwrite=True)

#Add and rotate
sub1 = fits.open(outnames[0])
sub2 = fits.open(outnames[1])
sum = sub1[0].data+sub2[0].data
rot = np.rot90(sum, k=3)
fits.writeto('spec1.fits',rot,hdr,overwrite=True)


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

