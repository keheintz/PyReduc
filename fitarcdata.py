#Run the setup script
exec(open("setup.py").read())

# =============================================================================
# fitarc: fit the arc data from identify
# =============================================================================

# Read idacr as astropy table
try:
    pix_wl_table = np.loadtxt('database/idarc.txt')
    pixnumber = pix_wl_table[:,0]
    wavelength = pix_wl_table[:,1]
except IOError:
    print("Output from identify.py is not accessible in the database")

ID_init = dict(pixel_gauss=pixnumber,
               wavelength=wavelength)
ID_init = Table(ID_init)
 
FitContinue = True
while FitContinue:
    
    if FITTING_MODEL_ID.lower() == 'chebyshev':
        coeff_ID, fitfull = chebfit(ID_init['pixel_gauss'], 
                                    ID_init['wavelength'], 
                                    deg=ORDER_ID,
                                    full=True)
        fitRMS = np.sqrt(fitfull[0][0]/len(ID_init))
        residual = ( ID_init['wavelength'] 
                    - chebval(ID_init['pixel_gauss'], coeff_ID))
        res_range = np.max(np.abs(residual))
    else:
        raise ValueError('Function {:s} is not implemented.'.format(FITTING_MODEL_REID))
    
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(3, 1)
    ax1 = plt.subplot(gs[0:2])
    ax2 = plt.subplot(gs[2])
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.plot(ID_init['pixel_gauss'], ID_init['wavelength'],
             ls=':', color='k', ms=10, marker='+')
    ax2.plot(ID_init['pixel_gauss'], residual, 
             ls='', color='k', ms=10, marker='+')
    ax2.axhline(y=0, color='b', ls=':')
    ax2.axhline(y=-fitRMS, color='r', ls=':')
    ax2.axhline(y=+fitRMS, color='r', ls=':')
    ax2.set_ylim(-2.*res_range, +2*res_range)
    ax1.set_ylabel(r'Wavelength ($\AA$)')
    ax2.set_ylabel(r'Residual ($\AA$)')
    ax2.set_xlabel('Pixel along dispersion axis')
    ax1.grid(ls=':')
    ax2.grid(ls=':')
    plt.suptitle(('First Identify (Chebyshev order {:d})\n'.format(ORDER_ID) 
                  + r'RMSE = {:.2f} $\AA$'.format(fitRMS)))
    
    print('Double left-click on lines that should be deleted. Other mouse-clicks to exit.')
    deleted = list()
    get_new_line = True
    while get_new_line:
        points = plt.ginput(n=1, timeout=30, show_clicks=True, mouse_add=1, mouse_stop=2)
        if len(points) == 1:
            pix_ref, _ = points[0]
            select = np.abs(pixnumber - pix_ref) < IDtolerance
            wl=wavelength[select]
            deleted.append(wl[0])
            ax1.plot([pixnumber[select]],[wavelength[select]],marker="|", color='b', markersize=20)
        else:
            answer = input("End deleting lines [Y/N]?: ")
            if answer.lower() in ['yes', 'y', '']:
                 get_new_line = False
                 plt.close("all")
    
    plt.show()
    
    answer = input("Fit again [Y/N]?: ")
    if answer.lower() in ['yes', 'y', '']: FitContinue = True
    else: FitContinue = False

print(deleted)

#Update the list
pix_wl_table_new = list()
for n in range(0,len(wavelength)): 
     if (wavelength[n] not in deleted): pix_wl_table_new.append([pixnumber[n], wavelength[n]])

#Print list of pixel numbers and wavelengths to file in the database
pix_wl_table_new = np.array(pix_wl_table_new)
print(" Pixel to wavelength reference table :")
df = pd.DataFrame(pix_wl_table_new[pix_wl_table_new[:,1].argsort()],dtype='float32')
print(df)
df.to_csv('database/idarc.txt', header=None, index=None, sep=' ')

