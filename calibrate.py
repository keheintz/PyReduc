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
    data = np.loadtxt('spec1_1dw.dat')
except IOError:
    print("Missing input spectrum to calibrate?")
wavelength = data[:,0]
objcounts = data[:,1] 
objnoise = data[:,2] 
airmass = OBJAIRMASS
exptime = OBJEXPTIME
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
flux = objcounts / 10**((chebval(ID_init['wavelength'], coeff_ID) - airmass*extinterp1d)/2.5) 
noise = objnoise / 10**((chebval(ID_init['wavelength'], coeff_ID) - airmass*extinterp1d)/2.5) 

fig = plt.figure(figsize=(10,5))
plt.xlim(3600,9300)
goodrange = (wavelength > 3600) & (wavelength < 9000)
plt.ylim(0,np.amax(flux[goodrange])*1.2)
plt.xlabel('lambda i Å')
plt.ylabel('Flux')
plt.xlabel('Observed wavelength [Å]')
plt.ylabel('Flux [erg/s/cm2/Å]')
plt.plot(wavelength, flux, lw = 1,
         alpha=1.0, label='flux-calibrated 1d spectrum')
plt.plot(wavelength, noise, lw = 1, color='r', label='1-sigma noise spectrum')
plt.legend()
plt.show()

#Write spectrum to a file
df_frame = {'wave':wavelength, 'flux':flux,  'noise':noise}
df = pd.DataFrame(df_frame,dtype='float32')
df.to_csv('flux_spec1.dat', header=None, index=None, sep=' ')

