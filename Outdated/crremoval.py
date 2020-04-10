#Run the setup script
exec(open("setup.py").read())

# This is a python program to run cosmic ray removal on science frames

# Path to folder with science frames
sciencefolder = "rawdata/rawscience/"
for nn in glob.glob(sciencefolder+"*_crr.fits"):
    os.remove(nn)

files = glob.glob(sciencefolder+"AL*.fits")
for n in files:
    fitsfile = fits.open(str.rstrip(n))
    fitsname = n.replace('.fits','')
    print('Got here')
    fitsfile[0].header['IMAGECAT'] = fitsfile[0].header['IMAGECAT']

    print(n)
    if fitsfile[0].header['IMAGECAT'] == 'SCIENCE':
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
