#Run the setup script
exec(open("setup.py").read())

# =============================================================================
# Identify (1): plot for manual input
# =============================================================================
# mimics IRAF IDENTIFY
#   IDENTIIFY image.fits section='middle line' nsum=NSUM_ID

wavelength=[3820.68, 3889.75, 3965.84, 4027.34, 4472.73, 4714.52, 5017.07, 5402.06,
            5877.246, 5946.48, 6404.02, 6680.15, 6931.39, 7034.36, 7440.95, 8302.61, 8379.68,
            8497.7, 8784.61, 9151.2]

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


#Print reference wavelengths
n = 1
print('Reference wavelength list:')
for wl in wavelength:
     print(n,wl)
     n=n+1


print('First zoom, then hit any key to select with right-click on the mouse:')
# Mark lines in the window and get the x values:
pix_wl_table = list()
get_new_line = True
while get_new_line:
    plt.waitforbuttonpress()
    points = plt.ginput(n=1, timeout=30, show_clicks=True, mouse_add=1, mouse_stop=2)
    if len(points) == 1:
        pix_ref, _ = points[0]
        # Fit Gaussian around +/- 10 pixels:
        mask = np.abs(x_identify - pix_ref) < FWHM_ID
        # Determine initial values:
        A0 = np.max(identify_1[int(pix_ref)])
        #bg0 = np.median([mask])
        bg0 = 0.
        # Use initial sigma of 1 pix:
        sig0 = 1.
        p0 = [pix_ref, sig0, A0, bg0]
        popt, pcov = curve_fit(gaussian, x_identify[mask], identify_1[mask], p0)
        centroid_fit = popt[0]
        plt.axvline(centroid_fit, ymin=0.75, ymax=0.95, color='r', lw=0.5)
        plt.draw()

        print("Give number in wavelength list corresponding to pix: %.2f" % centroid_fit)
        num_wl_ref = int(input(" number (-1 to skip): "))
        # Then here you can add a table look-up from a linelist...
        if num_wl_ref != -1: pix_wl_table.append([centroid_fit, wavelength[num_wl_ref-1]])

    else:
        print("End input?")
        answer = input(" [Y/N] : ")
        if answer.lower() in ['yes', 'y', '']: get_new_line = False

plt.show()

# Print to file
print(" Pixel to wavelength reference table :")
df = pd.DataFrame(pix_wl_table,dtype='float32')
print(df)
df.to_csv('database/idarc.txt', header=None, index=None, sep=' ')

