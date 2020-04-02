#The equivalent of "sensfunc" in IRAF/onedspec
#https://astro.uni-bonn.de/~sysstw/lfa_html/iraf/noao.onedspec.sensfunc.html
#As output the script produces two files: sens_order.dat (the order of the chebychev fit), 
#and sens_coeff.dat (the chebychev coefficients).

#Run the setup script
exec(open("setup.py").read())

#Read output from standard.py
try:
    data = np.loadtxt('database/stdinfo')
    datastd = np.loadtxt('database/stddata')
except IOError:
    print("Output from standard.py is not accessible in the database")

#Read exptime and airmass
exptime = data[0]
airmass = data[1]
#Exptime and airmass
wlstd = datastd[:,0]
stdflux = datastd[:,1]
bandwidth = datastd[:,2]
stdcounts = datastd[:,3]

#Read the atmospheric extinction data for La Palma
try:
    dataext = np.loadtxt('database/lapalma.dat')
except IOError:
    print("Extinction file is not accessible in the database")
wlext = dataext[:,0]
ext = dataext[:,1]

#From the description of the IRAF task sensfunc:
#The calibration factor at each point is computed as 
#
# C = 2.5 log (O / (T B F)) + A E
#
#where O is the observed counts in a bandpass of an observation, T is the
#exposure time of the observation, B is the bandpass width, F is the flux per
#Angstrom at the bandpass for the standard star, A is the airmass of the
#observation, and E is the extinction at the bandpass. Thus, C is the ratio of
#the observed count rate per Angstrom corrected to some extinction curve to the
#expected flux expressed in magnitudes. The goal of the task is to fit the
#observations to the relation
#
#I have already divided with the exposure time and the bandwidth

E = np.ndarray(len(wlstd))
for n in range(0,len(wlstd)): 
     ii = (wlext > wlstd[n]-bandwidth[n]) & (wlext < wlstd[n]+bandwidth[n])
     E[n] = np.mean(ext[ii])

C = 2.5*np.log10(stdcounts/stdflux) + airmass*E 

ID_init = dict(wavelength=wlstd,
               conversion=C)
ID_init = Table(ID_init)

if FITTING_MODEL_SF.lower() == 'chebyshev':
    coeff_ID, fitfull = chebfit(ID_init['wavelength'],
                                ID_init['conversion'],
                                deg=ORDER_SF,
                                full=True)
    fitRMS = np.sqrt(fitfull[0][0]/len(ID_init))
    residual = ( ID_init['conversion']
                - chebval(ID_init['wavelength'], coeff_ID))
    res_range = np.max(np.abs(residual))
else:
    raise ValueError('Function {:s} is not implemented.'.format(FITTING_MODEL_SF))

fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(3, 1)
ax1 = plt.subplot(gs[0:2])
ax2 = plt.subplot(gs[2])
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.plot(ID_init['wavelength'], ID_init['conversion'],
         ls=':', color='k', ms=10, marker='+')
ax2.plot(ID_init['wavelength'], residual,
         ls='', color='k', ms=10, marker='+')
ax2.axhline(y=0, color='b', ls=':')
ax1.set_ylabel(r'Conversion (dex)')
ax2.set_ylabel(r'Residual (dex)')
ax2.set_xlabel('Wavelength (AA)')
ax1.grid(ls=':')
ax2.grid(ls=':')
plt.suptitle(('Sens Function (Chebyshev order {:d})\n'.format(ORDER_SF)
              + r'RMSE = {:.2f}'.format(fitRMS)))
plt.show()

#Write chebychev coefficients to a file in the database
f = open('database/sens_order.dat', 'w')
f.write("%.0i" %(ORDER_SF))
f.close
f = open('database/sens_coeff.dat', 'w')
for n in range(0,len(coeff_ID)):
      f.write("%.10e \n" %(coeff_ID[n]))
f.close

