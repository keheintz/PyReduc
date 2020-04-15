import sys

#Run the setup script
exec(open("setup.py").read())

# Read idarc 
try:
    data = np.loadtxt('database/idarc.dat')
    pixnumber = data[:,0]
    wavelength = data[:,1]
except IOError:
    print("idarc.dat not accessible in the database")


ID_init = dict(peak=pixnumber,
               wavelength=wavelength)

ID_init = Table(ID_init)

peak_gauss = ID_init['peak']
ID_init["pixel_gauss"] = peak_gauss


line_REID = np.zeros((len(ID_init), N_REID-1))
spatialcoord = np.arange(0, (N_REID - 1) * STEP_REID, STEP_REID) + STEP_REID / 2

print('Reidentify each section by Chebyshev (order {:d})'.format(ORDER_ID))
print('section      |  found  |  RMS')

for i in range(0, N_REID-1):
    lower_cut, upper_cut = i*STEP_REID, (i+1)*STEP_REID
    reidentify_i = np.sum(lampimage[lower_cut:upper_cut, :], 
                          axis=0)
    peak_gauss_REID = []
    
    for peak_pix_init in ID_init['pixel_gauss']:
        search_min = int(np.around(peak_pix_init - TOL_REID))
        search_max = int(np.around(peak_pix_init + TOL_REID))
        cropped = reidentify_i[search_min:search_max]
        x_cropped = np.arange(len(cropped)) + search_min
        
        A_init = np.max(cropped)
        mean_init = peak_pix_init
        stddev_init = FWHM_ID * gaussian_fwhm_to_sigma
        g_init = Gaussian1D(amplitude=A_init, mean=mean_init, stddev=stddev_init,
                            bounds={'amplitude':(0, 2*np.max(cropped)) ,
                                    'stddev':(0, TOL_REID)})
        g_fit = fitter(g_init, x_cropped, cropped)    
        fit_center = g_fit.mean.value
        if abs(fit_center - peak_pix_init) > TOL_REID:
            peak_gauss_REID.append(np.nan)
            continue
        peak_gauss_REID.append(fit_center)

    peak_gauss_REID = np.array(peak_gauss_REID)
    nonan_REID = np.isfinite(peak_gauss_REID)
    line_REID[:, i] = peak_gauss_REID
    peak_gauss_REID_nonan = peak_gauss_REID[nonan_REID]
    n_tot = len(peak_gauss_REID)
    n_found = np.count_nonzero(nonan_REID)
    
    if FITTING_MODEL_ID.lower() == 'chebyshev':
        coeff_REID1D, fitfull = chebfit(peak_gauss_REID_nonan,
                                        ID_init['wavelength'][nonan_REID], 
                                        deg=ORDER_WAVELEN_REID,
                                        full=True)
        fitRMS = np.sqrt(fitfull[0][0]/n_found)
    
    else:
        raise ValueError('Function {:s} is not implemented.'.format(FITTING_MODEL_REID))

    print('[{:04d}:{:04d}]\t{:d}/{:d}\t{:.3f}'.format(lower_cut, upper_cut,
                                                      n_found, n_tot, fitRMS))
    
    points = np.vstack((line_REID.flatten(),
                    np.tile(spatialcoord, len(line_REID))))
points = points.T # list of ()
nanmask = ( np.isnan(points[:,0]) | np.isnan(points[:,1]) )
points = points[~nanmask]
values = np.repeat(ID_init['wavelength'], N_REID - 1)
values = np.array(values.tolist())
values = values[~nanmask]
errors = np.ones_like(values)

if FITTING_MODEL_REID.lower() == 'chebyshev':
    coeff_init = Chebyshev2D(x_degree=ORDER_WAVELEN_REID, y_degree=ORDER_SPATIAL_REID)
    fit2D_REID = fitter(coeff_init, points[:, 0], points[:, 1], values)
    ww, ss = np.mgrid[:N_WAVELEN, :N_SPATIAL]
else:
    raise ValueError('Function {:s} is not implemented.'.format(FITTING_MODEL_REID))

fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(3, 3)
ax1 = plt.subplot(gs[:2, :2])
ax2 = plt.subplot(gs[2, :2])
ax3 = plt.subplot(gs[:2, 2])

title_str = ('Reidentify and Wavelength Map\n'
             + 'func=Chebyshev, order (wavelength, dispersion) = ({:d}, {:d})')
plt.suptitle(title_str.format(ORDER_WAVELEN_REID, ORDER_SPATIAL_REID))

interp_min = line_REID[~np.isnan(line_REID)].min()
interp_max = line_REID[~np.isnan(line_REID)].max()

ax1.imshow(fit2D_REID(ww, ss).T, origin='lower')
ax1.axvline(interp_max, color='r', lw=1)
ax1.axvline(interp_min, color='r', lw=1)

ax1.plot(points[:, 0], points[:, 1], ls='', marker='+', color='r',
         alpha=0.8, ms=10)


for i in (1, 2, 3):
    vcut = N_WAVELEN * i/4
    hcut = N_SPATIAL * i/4
    vcutax  = np.arange(0, N_SPATIAL, STEP_REID) + STEP_REID/2
    hcutax  = np.arange(0, N_WAVELEN, 1)
    vcutrep = np.repeat(vcut, len(vcutax))
    hcutrep = np.repeat(hcut, len(hcutax))
    
    ax1.axvline(x=vcut, ls=':', color='k')   
    ax1.axhline(y=hcut, ls=':', color='k')
    
    ax2.plot(hcutax, fit2D_REID(hcutax, hcutrep), lw=1, 
             label="hcut {:d}".format(int(hcut)))

    vcut_profile = fit2D_REID(vcutrep, vcutax)
    vcut_normalize = vcut_profile - np.median(vcut_profile)
    ax3.plot(vcut_normalize, vcutax, lw=1,
             label="vcut {:d}".format(int(vcut)))

ax1.set_ylabel('Spatial direction')
ax2.grid(ls=':')
ax2.legend()
ax2.set_xlabel('Dispersion direction')
ax2.set_ylabel('Wavelength\n(horizontal cut)')

ax3.axvline(0, ls=':', color='k')
ax3.grid(ls=':', which='both')
ax3.set_xlabel('Wavelength change (vertical cut)')
ax3.legend()

ax1.set_ylim(0, N_SPATIAL)
ax1.set_xlim(0, N_WAVELEN)
ax2.set_xlim(0, N_WAVELEN)
ax3.set_ylim(0, N_SPATIAL)
plt.show()

# =============================================================================
# Plot a cut along the spatial direction for selection of sky and object
# =============================================================================
lower_cut = N_WAVELEN//2 - NSUM_AP//2 
upper_cut = N_WAVELEN//2 + NSUM_AP//2
apall_1 = np.sum(objimage[:, lower_cut:upper_cut], axis=1)
max_intens = np.max(apall_1)

x_apall = np.arange(0, len(apall_1))

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
title_str = r'Select the selected number of sky-regions'
ax.plot(x_apall, apall_1, lw=1)

ax.grid(ls=':')
ax.set_xlabel('Pixel number')
ax.set_ylabel('Pixel value')
ax.set_xlim(0, len(apall_1))
ax.set_title(title_str.format(np.median(apall_1), int(N_SPATIAL/100)))


# =============================================================================
# manually select trace sky regions
# =============================================================================
nsky = 2*int(input("How many sky-regions (1-4)? "))
print('Click on sky-regions (start, end), i.e. equal number. End with q')
# Mark lines in the window and get the x values:
slitregions = list(range(nsky))
get_new_line = True
nregions = 0
plt.waitforbuttonpress()

while nregions <= nsky-1:
    tpoints = plt.ginput(n=1, timeout=30, show_clicks=True, mouse_add=1, mouse_stop=2)
    pix_ref, _ = tpoints[0]
    slitregions[nregions] = int(pix_ref)
    plt.axvline(slitregions[nregions], ymin=0.75, ymax=0.95, color='r', lw=0.5)
    plt.draw()
    nregions = nregions+1

ap_sky = np.array(slitregions)

plt.show()

#I wish I was able to do this more elegantly. Also, I do not understand why the result
#is better for nsky=2 than for nsky=8.
if nsky == 2:
    x_sky = np.hstack( (np.arange(ap_sky[0], ap_sky[1])))
    sky_val = np.hstack( (apall_1[ap_sky[0]:ap_sky[1]])) 
if nsky == 4:
    x_sky = np.hstack( (np.arange(ap_sky[0], ap_sky[1]), 
                        np.arange(ap_sky[2], ap_sky[3])))
    sky_val = np.hstack( (apall_1[ap_sky[0]:ap_sky[1]], 
                          apall_1[ap_sky[2]:ap_sky[3]]))
if nsky == 6:
    x_sky = np.hstack( (np.arange(ap_sky[0], ap_sky[1]), 
                        np.arange(ap_sky[2], ap_sky[3]),
                        np.arange(ap_sky[4], ap_sky[5])))
    sky_val = np.hstack( (apall_1[ap_sky[0]:ap_sky[1]], 
                          apall_1[ap_sky[2]:ap_sky[3]],
                          apall_1[ap_sky[4]:ap_sky[5]]))
if nsky == 8:
    x_sky = np.hstack( (np.arange(ap_sky[0], ap_sky[1]), 
                        np.arange(ap_sky[2], ap_sky[3]),
                        np.arange(ap_sky[4], ap_sky[5]),
                        np.arange(ap_sky[6], ap_sky[7])))
    sky_val = np.hstack( (apall_1[ap_sky[0]:ap_sky[1]], 
                          apall_1[ap_sky[2]:ap_sky[3]],
                          apall_1[ap_sky[4]:ap_sky[5]],
                          apall_1[ap_sky[6]:ap_sky[7]]))
if nsky >= 9: 
     sys.exit('Too many sky regions.')


fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
title_str = r'Skyfit: {:s} order {:d} ({:.1f}-sigma {:d}-iters)'
ax.plot(x_apall, apall_1, lw=1)

if FITTING_MODEL_APSKY.lower() == 'chebyshev':
    clip_mask = sigma_clip(sky_val, sigma=SIGMA_APSKY, maxiters=ITERS_APSKY).mask
    coeff_apsky, fitfull = chebfit(x_sky[~clip_mask], 
                                   sky_val[~clip_mask],
                                   deg=ORDER_APSKY,
                                   full=True)
    fitRMS = np.sqrt(fitfull[0][0])
    n_sky = len(x_sky)
    n_rej = np.count_nonzero(clip_mask)
    sky_fit = chebval(x_apall, coeff_apsky) 
    ax.plot(x_apall, sky_fit, ls='--',
            label='Sky Fit ({:d}/{:d} used)'.format(n_sky - n_rej, n_sky))
    ax.plot(x_sky[clip_mask], sky_val[clip_mask], marker='x', ls='', ms=10)
    [ax.axvline(i, lw=1, color='k', ls='--') for i in ap_sky]
    ax.legend()
    ax.set_title(title_str.format(FITTING_MODEL_APSKY, ORDER_APSKY,
                                  SIGMA_APSKY, ITERS_APSKY))
    ax.set_xlabel('Pixel number')
    ax.set_ylabel('Pixel value sum')
    
else:
    raise ValueError('Function {:s} is not implemented.'.format(FITTING_MODEL_REID))
ax.grid(ls=':')
ax.set_xlabel('Pixel number')
ax.set_ylabel('Pixel value')
plt.show()

#Now try to make a 2d skysubtraction within these limits
#subtract sky
data_skysub = []    
data_sky = []
for i in range(N_WAVELEN):
    cut_i = objimage[:, i].copy()
  
    if nsky == 2:
        x_sky = np.hstack( (np.arange(ap_sky[0], ap_sky[1])))
        sky_val = np.hstack( (cut_i[ap_sky[0]:ap_sky[1]])) 
    if nsky == 4:
        x_sky = np.hstack( (np.arange(ap_sky[0], ap_sky[1]), 
                            np.arange(ap_sky[2], ap_sky[3])))
        sky_val = np.hstack( (cut_i[ap_sky[0]:ap_sky[1]], 
                              cut_i[ap_sky[2]:ap_sky[3]]))
    if nsky == 6:
        x_sky = np.hstack( (np.arange(ap_sky[0], ap_sky[1]), 
                            np.arange(ap_sky[2], ap_sky[3]),
                            np.arange(ap_sky[4], ap_sky[5])))
        sky_val = np.hstack( (cut_i[ap_sky[0]:ap_sky[1]], 
                              cut_i[ap_sky[2]:ap_sky[3]],
                              cut_i[ap_sky[4]:ap_sky[5]]))
    if nsky == 8:
        x_sky = np.hstack( (np.arange(ap_sky[0], ap_sky[1]), 
                            np.arange(ap_sky[2], ap_sky[3]),
                            np.arange(ap_sky[4], ap_sky[5]),
                            np.arange(ap_sky[6], ap_sky[7])))
        sky_val = np.hstack( (cut_i[ap_sky[0]:ap_sky[1]], 
                              cut_i[ap_sky[2]:ap_sky[3]],
                              cut_i[ap_sky[4]:ap_sky[5]],
                              cut_i[ap_sky[6]:ap_sky[7]]))
    clip_mask = sigma_clip(sky_val, sigma=SIGMA_APSKY, maxiters=ITERS_APSKY).mask
  
    coeff = chebfit(x_sky[~clip_mask],
                    sky_val[~clip_mask],
                    deg=ORDER_APSKY)

    data_skysub.append(cut_i - chebval(np.arange(cut_i.shape[0]), coeff))
    data_sky.append(chebval(np.arange(cut_i.shape[0]), coeff))

#write out the sky-subtracted and the variance images to fits files
hdr = objhdu[0].header
data_skysub = np.array(data_skysub).T
data_sky = np.array(data_sky).T
hdr.add_history(f"Sky subtracted using background.py")
_ = fits.PrimaryHDU(data=data_skysub, header=hdr)
_.data = _.data.astype('float32')
_.writeto(DATAPATH/(OBJIMAGE.stem+".skysub.fits"), overwrite=True)
_ = fits.PrimaryHDU(data=data_sky, header=hdr)
_.data = _.data.astype('float32')
_.writeto(DATAPATH/(OBJIMAGE.stem+".sky.fits"), overwrite=True)

