#Run the setup script
exec(open("setup.py").read())

# =============================================================================
# Identify (1): plot for manual input
# =============================================================================
# mimics IRAF IDENTIFY
#   IDENTIIFY image.fits section='middle line' nsum=NSUM_ID
lowercut_ID = N_SPATIAL//2 - NSUM_ID//2 
uppercut_ID = N_SPATIAL//2 + NSUM_ID//2
identify_1 = np.median(lampimage[lowercut_ID:uppercut_ID, :], axis=0)

# For plot and visualization
max_intens = np.max(identify_1)


#disable_mplkeymaps()
fig = plt.figure()
ax = fig.add_subplot(111)
title_str = r'Arc line plot'
# Plot original spectrum + found peak locations
x_identify = np.arange(0, len(identify_1))
ax.plot(x_identify, identify_1, lw=1)


ax.grid(ls=':')
ax.set_xlabel('Pixel number')
ax.set_ylabel('Pixel value sum')
ax.set_xlim(0,len(identify_1))
ax.set_title(title_str.format(MINAMP_PK, MINSEP_PK))


table = []

def onclick(event):
     if event.dblclick:
        g_init = Gaussian1D(amplitude = max_intens*1.2, 
                 mean = event.xdata, 
                 stddev = FWHM_ID * gaussian_fwhm_to_sigma,
                 bounds={'amplitude': (0, max_intens*1.2),
                 'mean':(event.xdata-FWHM_ID, event.xdata+FWHM_ID),
                 'stddev':(0, FWHM_ID)})
        fitted = LINE_FITTER(g_init, x_identify, identify_1)
        ax.plot((fitted.mean.value,fitted.mean.value) , (0,2*max_intens))
        pixnumber = fitted.mean.value
        print(pixnumber)

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

# Make it as astropy table
 

