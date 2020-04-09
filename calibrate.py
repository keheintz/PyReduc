#The equivalent of "calibrate" in IRAF/onedspec
#https://github.com/joequan/tiraf/blob/master/noao/onedspec/doc/calibrate.hlp
#As input the script requires: 
#- sens_coeff.dat (the chebychev coefficients). 
#- the extracted, wavelength-calibrated science-file (from extract1d.py)
#
#The output file is the flux-calibrated science-file.

#Run the setup script
exec(open("setup.py").read())

#Read the sensitivity-solution
try:
    coeff_ID = np.loadtxt('database/sens_coeff.dat', delimiter="\n")
except IOError:
    print("Output from sensfunction.py is not accessible in the database")

#Read the spectrum to be calibrated
try:
    file = fits.open(OBJIMAGE.stem+".ms_1d.fits")
except IOError:
    print("Missing input spectrum to calibrate?")

hdr = file[0].header
objopt = file[0].data[0,0,:] 
objsum = file[0].data[1,0,:] 
objsky = file[0].data[2,0,:]
objnoise = file[0].data[3,0,:]
airmass = OBJAIRMASS
exptime = OBJEXPTIME
wavelength = np.arange(len(objopt))*hdr['CD1_1']+hdr['CRVAL1']
print(airmass,exptime)

#Read the atmospheric extinction data for La Palma
try:
    dataext = np.loadtxt('database/lapalma.dat')
except IOError:
    print("Extinction file is not accessible in the database")
wlext = dataext[:,0]
ext = dataext[:,1]
#interpolate the extinction file onto the wavelength grid of the object spectrum
f = interp1d(wlext, ext, kind='cubic')
extinterp1d = f(wavelength)  

ID_init = dict(wavelength=wavelength)
ID_init = Table(ID_init)

c = chebval(ID_init['wavelength'], coeff_ID)
optflux = objopt / 10**((chebval(ID_init['wavelength'], coeff_ID) - airmass*extinterp1d)/2.5) 
sumflux = objsum / 10**((chebval(ID_init['wavelength'], coeff_ID) - airmass*extinterp1d)/2.5) 
skyflux = objsky / 10**((chebval(ID_init['wavelength'], coeff_ID) - airmass*extinterp1d)/2.5) 
noiseflux = objnoise / 10**((chebval(ID_init['wavelength'], coeff_ID) - airmass*extinterp1d)/2.5) 

fig = plt.figure(figsize=(10,5))
plt.xlim(3600,9300)
goodrange = (wavelength > 3600) & (wavelength < 9300)
plt.ylim(0,np.amax(optflux[goodrange])*1.2)
plt.xlabel('lambda i Å')
plt.ylabel('Flux')
plt.xlabel('Observed wavelength [Å]')
plt.ylabel('Flux [erg/s/cm2/Å]')
plt.plot(wavelength, optflux, lw = 1,
         alpha=1.0, label='flux-calibrated 1d spectrum')
plt.plot(wavelength, noiseflux, lw = 1, color='r', label='1-sigma noise spectrum')
plt.legend()
plt.show()

#Write spectrum to an ascii file
df_frame = {'wave':wavelength, 'optflux':optflux,  'sumflux':sumflux, 'skyflux':skyflux, 'fluxnoise':noiseflux}
df = pd.DataFrame(df_frame,dtype='float32')
df.to_csv('flux_'+OBJIMAGE.stem+'.ms_1dw.dat', header=None, index=None, sep=' ')

#Write the fits-file
rightnow = datetime.now().strftime("%a %Y-%m-%dT%H:%M:%S")
hdr.add_history(f"Flux-calibrated with calibrate.py %s" % rightnow)
hdr.set('BUNIT', 'erg/cm2/s/A')

# stack in the same array configuration as a multispec
multispecdata = fake_multispec_data((optflux, sumflux, skyflux, noiseflux))
hduout = fits.PrimaryHDU(multispecdata)
hdrcopy = hdr.copy(strip = True)
hduout.header.extend(hdrcopy, strip=True, update=True,
        update_first=False, useblanks=True, bottom=False)

hduout.writeto("flux_"+OBJIMAGE.stem+".ms_1d.fits", overwrite=True)
print(" ")
print("** Wrote output file 'flux_%s.ms_1d.fits' ." % OBJIMAGE.stem)
print(" ")


