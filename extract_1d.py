#Run the setup script
exec(open("setup.py").read())

# =============================================================================
# reidentify
# =============================================================================
# For REIDENTIFY, I used astropy.modeling.models.Chebyshev2D

# Read idarc 
try:
    data = np.loadtxt('database/idarc.dat')
    pixnumber = data[:,0]
    wavelength = data[:,1]
except IOError:
    print("File not accessible")


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
        
        #TODO: put something like "lost_factor" to multiply to FWHM_ID in the bounds.
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
#plt.setp(ax2.get_xticklabels(), visible=False)
#plt.setp(ax3.get_yticklabels(), visible=False)

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
# apall (1): Plot a cut
# =============================================================================
lower_cut = N_WAVELEN//2 - NSUM_AP//2 
upper_cut = N_WAVELEN//2 + NSUM_AP//2
apall_1 = np.sum(objimage[:, lower_cut:upper_cut], axis=1)
max_intens = np.max(apall_1)

x_apall = np.arange(0, len(apall_1))

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
title_str = r'Select Sky_left_1, Sky_left_2, Object, Sky_right_1, Sky_rigth_2'
ax.plot(x_apall, apall_1, lw=1)

ax.grid(ls=':')
ax.set_xlabel('Pixel number')
ax.set_ylabel('Pixel value')
ax.set_xlim(0, len(apall_1))
ax.set_title(title_str.format(np.median(apall_1), int(N_SPATIAL/100)))


# =============================================================================
# apall(2): manually select trace sky regions
# =============================================================================

print('First click on sky-region left of the trace (start, end), then the trace and finally sky region right of the trace (star, end). End with q')
# Mark lines in the window and get the x values:
slitregions = list(range(5))
get_new_line = True
nregions = 0
plt.waitforbuttonpress()

while nregions <= 4:
    tpoints = plt.ginput(n=1, timeout=30, show_clicks=True, mouse_add=1, mouse_stop=2)
    pix_ref, _ = tpoints[0]
    slitregions[nregions] = int(pix_ref)
    plt.axvline(slitregions[nregions], ymin=0.75, ymax=0.95, color='r', lw=0.5)
    plt.draw()
    nregions = nregions+1

ap_init = slitregions[2]
ap_sky = np.array([slitregions[0], slitregions[1], slitregions[3], slitregions[4]])

plt.show()

# Regions to use as sky background. xl1 - 1, xu1, xl2 - 1, xu2. (0-indexing)
#   Sky region should also move with aperture center!
#   from ``ap_center - 50`` to ``ap_center - 40``, for example, should be used.

# Interactive check
x_sky = np.hstack( (np.arange(ap_sky[0], ap_sky[1]), 
                    np.arange(ap_sky[2], ap_sky[3])))
sky_val = np.hstack( (apall_1[ap_sky[0]:ap_sky[1]], 
                      apall_1[ap_sky[2]:ap_sky[3]]))

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
title_str = r'Skyfit: {:s} order {:d} ({:.1f}-sigma {:d}-iters)'
ax.plot(x_apall, apall_1, lw=1)

if FITTING_MODEL_APSKY.lower() == 'chebyshev':
    # TODO: maybe the clip is "3-sigma clip to residual and re-fit many times"?
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


# =============================================================================
# apall (3): aperture trace
# =============================================================================
# within +- 100 pixels around the aperture, the wavelength does not change much
# as can be seen from reidentify figure 
# (in NHAO case, ~ 0.01% ~ 0.1 Angstrom order).
# So it's safe to assume the wavelength is constant over around such region,
# (spatial direction) and thus I will do sky fitting from this column,
# without considering the wavelength change along a column.
# Then aperture extraction will map the pixel to wavelength using aperture
# trace solution.

aptrace = []
aptrace_fwhm = []
#coeff_apsky = []
#aptrace_apsum = []
#aptrace_wavelen = []

# TODO: This is quite slow as for loop used: improvement needed.
# I guess the problem is sigma-clipping rather than fitting process..
for i in range(N_AP - 1):
    lower_cut, upper_cut = i*STEP_AP, (i+1)*STEP_AP
    
    apall_i = np.sum(objimage[:, lower_cut:upper_cut], axis=1)
    sky_val = np.hstack( (apall_i[ap_sky[0]:ap_sky[1]], 
                          apall_i[ap_sky[2]:ap_sky[3]]))
    
    # Subtract fitted sky
    if FITTING_MODEL_APSKY.lower() == 'chebyshev':
        # TODO: maybe we can put smoothing function as IRAF APALL's b_naverage 
        clip_mask = sigma_clip(sky_val, sigma=SIGMA_APSKY, maxiters=ITERS_APSKY).mask
        coeff, fitfull = chebfit(x_sky[~clip_mask], 
                                 sky_val[~clip_mask],
                                 deg=ORDER_APSKY,
                                 full=True)
        apall_i -= chebval(x_apall, coeff)
    
    else:
        raise ValueError('Function {:s} is not implemented.'.format(FITTING_MODEL_APSKY))

    #TODO: put something like "lost_factor" to multiply to FWHM_ID in the bounds.
    search_min = int(np.around(ap_init - 3*FWHM_AP))
    search_max = int(np.around(ap_init + 3*FWHM_AP))
    cropped = apall_i[search_min:search_max]
    x_cropped = np.arange(len(cropped))
    peak_pix = peak_local_max(cropped, 
                              min_distance=FWHM_AP,
                              indices=True,
                              num_peaks=1)
    if len(peak_pix) == 0:
        aptrace.append(np.nan)
        continue
    peak_pix = peak_pix[0][0]
    
    #TODO: put something like "lost_factor" to multiply to FWHM_ID in the bounds.
    g_init = Gaussian1D(amplitude=cropped[peak_pix], 
                       mean=peak_pix, 
                       stddev=FWHM_AP * gaussian_fwhm_to_sigma,
                       bounds={'amplitude':(0, 2*cropped[peak_pix]) ,
                               'mean':(peak_pix-3*FWHM_AP, peak_pix+3*FWHM_AP),
                               'stddev':(0, FWHM_AP)})
    fitted = fitter(g_init, x_cropped, cropped)
    center_pix = fitted.mean.value + search_min
    std_pix = fitted.stddev.value
    aptrace_fwhm.append(fitted.fwhm)
    aptrace.append(center_pix)
#    coeff_apsky.append(coeff)
#    aptrace_apsum.append(apsum)
#    apsum_lower = int(np.around(center_pix - apsum_sigma_lower * std_pix))
#    apsum_upper = int(np.around(center_pix + apsum_sigma_upper * std_pix))
#    apsum = np.sum(apall_i[apsum_lower:apsum_upper])

aptrace = np.array(aptrace)
aptrace_fwhm = np.array(aptrace_fwhm)

#coeff_apsky = np.array(coeff_apsky)
#aptrace_apsum = np.array(aptrace_apsum)


# =============================================================================
# apall(4): aperture trace fit
# =============================================================================
x_aptrace = np.arange(N_AP-1) * STEP_AP

#position
coeff_aptrace = chebfit(x_aptrace, aptrace, deg=ORDER_APTRACE)
resid_mask = sigma_clip(aptrace - chebval(x_aptrace, coeff_aptrace), 
                        sigma=SIGMA_APTRACE, maxiters=ITERS_APTRACE).mask

#width
coeff_aptrace_fwhm = chebfit(x_aptrace, aptrace_fwhm, deg=ORDER_APTRACE)
resid_mask_fwhm = sigma_clip(aptrace_fwhm - chebval(x_aptrace, coeff_aptrace_fwhm), 
                        sigma=SIGMA_APTRACE, maxiters=ITERS_APTRACE).mask

#position
x_aptrace_fin = x_aptrace[~resid_mask]
aptrace_fin = aptrace[~resid_mask]
coeff_aptrace_fin = chebfit(x_aptrace_fin, aptrace_fin, deg=ORDER_APTRACE)
fit_aptrace_fin   = chebval(x_aptrace_fin, coeff_aptrace_fin)
resid_aptrace_fin = aptrace_fin - fit_aptrace_fin
del_aptrace = ~np.in1d(x_aptrace, x_aptrace_fin) # deleted points

#width
x_aptrace_fwhm_fin = x_aptrace[~resid_mask_fwhm]
aptrace_fwhm_fin = aptrace_fwhm[~resid_mask_fwhm]
coeff_aptrace_fwhm_fin = chebfit(x_aptrace_fwhm_fin, aptrace_fwhm_fin, deg=ORDER_APTRACE)
fit_aptrace_fwhm_fin   = chebval(x_aptrace_fwhm_fin, coeff_aptrace_fwhm_fin)
resid_aptrace_fwhm_fin = aptrace_fwhm_fin - fit_aptrace_fwhm_fin
del_aptrace_fwhm = ~np.in1d(x_aptrace, x_aptrace_fwhm_fin) # deleted points


#Plot the center of the trace and the fwhm of the trace and the fits to these
fig = plt.figure(figsize=(10,10))
gs = gridspec.GridSpec(4, 1)
ax2 = plt.subplot(gs[3])
ax1 = plt.subplot(gs[1:3], sharex=ax2)
ax3 = plt.subplot(gs[0], sharex=ax2)

title_str = ('Aperture Trace Fit ({:s} order {:d})\n'
            + 'Residuials {:.1f}-sigma, {:d}-iters clipped')
plt.suptitle(title_str.format(FITTING_MODEL_APTRACE, ORDER_APTRACE,
                              SIGMA_APTRACE, ITERS_APTRACE))
ax1.plot(x_aptrace, aptrace, ls='', marker='+', ms=10)
ax1.plot(x_aptrace_fin, fit_aptrace_fin, ls='--',
         label="Aperture Trace ({:d}/{:d} used)".format(len(aptrace_fin), N_AP-1))
ax1.plot(x_aptrace[del_aptrace], aptrace[del_aptrace], ls='', marker='x', ms=10)
ax1.legend()
ax2.plot(x_aptrace_fin, resid_aptrace_fin, ls='', marker='+')
ax2.axhline(+np.std(resid_aptrace_fin, ddof=1), ls=':', color='k')
ax2.axhline(-np.std(resid_aptrace_fin, ddof=1), ls=':', color='k', 
            label='residual std')
ax3.plot(x_aptrace, aptrace_fwhm, ls='', marker='+', ms=10)
ax3.plot(x_aptrace_fwhm_fin, fit_aptrace_fwhm_fin, ls='--',
         label="Aperture Width ({:d}/{:d} used)".format(len(aptrace_fwhm_fin), N_AP-1))

ax1.set_ylabel('Found object position')
ax2.set_ylabel('Residual (pixel)')
ax2.set_xlabel('Dispersion axis (pixel)')
ax3.set_ylabel('FWHM (pixel)')
ax1.grid(ls=':')
ax2.grid(ls=':')
ax3.grid(ls=':')
ax2.set_ylim(-.5, .5)
ax2.legend()
plt.show()
#plt.savefig('aptrace.png', bbox_inches='tight')


# =============================================================================
# apall(5): aperture sum
# =============================================================================
apsum_sigma_lower = 2.104 # See below
apsum_sigma_upper = 2.130 
# lower and upper limits of aperture to set from the center in gauss-sigma unit.
ap_fwhm = np.median(aptrace_fwhm[~resid_mask])
ap_sigma = ap_fwhm * gaussian_fwhm_to_sigma
apheight = (apsum_sigma_lower + apsum_sigma_upper)*ap_sigma

x_ap = np.arange(N_WAVELEN)
y_ap = chebval(x_ap, coeff_aptrace_fin)
ap_wavelen = fit2D_REID(x_ap, y_ap)
ap_sky_offset = ap_sky - ap_init

#Define limits of aperture for the APNUM1 keyword
apmin = np.min(y_ap)-apheight 
apmax = np.max(y_ap)+apheight 

data_skysub = []
data_sky = []
data_variance = []

#subtract sky
for i in range(N_WAVELEN):
    cut_i = objimage[:, i].copy()
    ap_sky_i = int(y_ap[i]) + ap_sky_offset
   
    x_sky = np.hstack( (np.arange(ap_sky_i[0], ap_sky_i[1]),
                        np.arange(ap_sky_i[2], ap_sky_i[3])))
    sky_val = np.hstack( (cut_i[ap_sky_i[0]:ap_sky_i[1]],
                          cut_i[ap_sky_i[2]:ap_sky_i[3]]))
    clip_mask = sigma_clip(sky_val, sigma=SIGMA_APSKY, maxiters=ITERS_APSKY).mask
   
    coeff = chebfit(x_sky[~clip_mask],
                    sky_val[~clip_mask],
                    deg=ORDER_APSKY)

    data_skysub.append(cut_i - chebval(np.arange(cut_i.shape[0]), coeff))
    data_sky.append(chebval(np.arange(cut_i.shape[0]), coeff))
    data_variance.append((np.abs(cut_i)/GAIN+(OBJNEXP*RON/GAIN)**2))
#To get this formular look at the noise in electrons and then convert to ADU

#write out the sky-subtracted and the variance images to fits files
hdr = objhdu[0].header
data_skysub = np.array(data_skysub).T
data_sky = np.array(data_sky).T
data_variance = np.array(data_variance).T
hdr.add_history(f"Sky subtracted using sky offset = {ap_sky_offset}, "
                + f"{SIGMA_APSKY}-sigma {ITERS_APSKY}-iter clipping "
                + f"to fit order {ORDER_APSKY} Chebyshev")
_ = fits.PrimaryHDU(data=data_skysub, header=hdr)
_.data = _.data.astype('float32')
_.writeto(DATAPATH/(OBJIMAGE.stem+".skysub.fits"), overwrite=True)
_ = fits.PrimaryHDU(data=data_variance, header=hdr)
_.data = _.data.astype('float32')
_.writeto(DATAPATH/(OBJIMAGE.stem+".variance.fits"), overwrite=True)
_ = fits.PrimaryHDU(data=data_sky, header=hdr)
_.data = _.data.astype('float32')
_.writeto(DATAPATH/(OBJIMAGE.stem+".sky.fits"), overwrite=True)


#calculate the counts of the 1d-spectrum and its 1-sigma noise (sum in aperture)
#also calculate the optimally extracted 1d-spectrum and its 1-sigma noise (Horne method)
#Finally also sum up the sky along the trace for a separate extension in the output file.

#first the sum in the aperture
pos = np.array([x_ap, y_ap]).T
aps = RectangularAperture(positions=pos, w=1, h=apheight, theta=0)
phot = aperture_photometry(data_skysub, aps, method='subpixel', subpixels=30)
ap_summed = phot['aperture_sum'] / OBJEXPTIME
var = aperture_photometry(data_variance, aps, method='subpixel', subpixels=30)
ap_variance = var['aperture_sum'] / OBJEXPTIME**2
sky = aperture_photometry(data_sky, aps, method='subpixel', subpixels=30)
ap_sky = sky['aperture_sum'] / OBJEXPTIME

#then the optimally extracted spectrum
ap_optimal = np.array(range(0,N_WAVELEN))
var_optimal = np.array(range(0,N_WAVELEN))
NROWS = objhdu[0].header['NAXIS2']
c = chebval(x_ap, coeff_aptrace_fin)
s = chebval(x_ap, coeff_aptrace_fwhm_fin)/np.sqrt(8.*np.log(2.))
y = np.array(range(0,NROWS))
for n in range(N_WAVELEN):
    weight = gaussweight(y,c[n],s[n])
    ap_optimal[n] = np.sum(data_skysub[:,n]*weight/data_variance[:,n]) / np.sum(weight**2/data_variance[:,n])
    var_optimal[n] = np.sum(weight) / np.sum(weight**2/data_variance[:,n])
ap_optimal = ap_optimal/ OBJEXPTIME
var_optimal = var_optimal/ OBJEXPTIME**2

#Plot trace in 2d before and after sky-subtraction
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False, sharey=False, gridspec_kw=None)
axs[0].imshow(objimage, vmin=0, vmax=1000, origin='lower')
axs[1].imshow(data_skysub, vmin=-50, vmax=200, origin='lower')
axs[0].set(title="Before sky subtraction")
axs[1].set(title="Sky subtracted")

for ax in [axs[0], axs[1]]:
    ax.plot(x_ap, y_ap+apsum_sigma_upper*ap_sigma, 'r-', lw=1)
    ax.plot(x_ap, y_ap-apsum_sigma_lower*ap_sigma, 'r-', lw=1)
    ax.set(ylim=(ap_init-50, ap_init+50))
    for offset in ap_sky_offset:
        ax.plot(x_ap, y_ap+offset, 'y-', lw=1)
plt.tight_layout()
plt.show()


#Now the extraction is complete. We need to write everything out to fits files.
#We want to make a 1d-file with the same number of pixels and a dispersion dictated 
#by the condition that there should be a constant dispersion per pixel.

ap_wavelen_start = ap_wavelen[0]
ap_wavelen_end = ap_wavelen[N_WAVELEN-1]
dw = np.float32((ap_wavelen_end-ap_wavelen_start)/(float(N_WAVELEN)-1.))
wavelength = np.float32(ap_wavelen_start+range(0,N_WAVELEN)*dw)
print(N_WAVELEN,len(wavelength))

#Interpolate onto the new grid
f = interp1d(ap_wavelen,ap_optimal,fill_value="extrapolate", kind='cubic')
out1 = f(wavelength)
f = interp1d(ap_wavelen,ap_summed,fill_value="extrapolate", kind='cubic')
out2 = f(wavelength)
f = interp1d(ap_wavelen,ap_sky,fill_value="extrapolate", kind='cubic')
out3 = f(wavelength)
f = interp1d(ap_wavelen,np.sqrt(var_optimal),fill_value="extrapolate", kind='cubic')
out4 = f(wavelength)

#Now follow John Thorstensen, Dartmouth College (thanks!)
#copy over the header from the original spectrum.
# stack in the same array configuration as a multispec
multispecdata = fake_multispec_data((np.float32(out1), np.float32(out2), np.float32(out3), np.float32(out4)))
hduout = fits.PrimaryHDU(multispecdata)
hdrcopy = hdr.copy(strip = True)
hduout.header.extend(hdrcopy, strip=True, update=True,
        update_first=False, useblanks=True, bottom=False)

hduout.header['HISTORY'] = "From development of extract_1d.py"
rightnow = datetime.now().strftime("%a %Y-%m-%dT%H:%M:%S")
hduout.header['HISTORY'] = "Extracted %s" % (rightnow)
hduout.header['EXTEND'] = False
hduout.header['BUNIT'] = 'erg/cm2/s/A'
hduout.header['CTYPE1'] = 'LINEAR  '
hduout.header['CTYPE2'] = 'LINEAR  '
hduout.header['CRVAL1'] = wavelength[0]
hduout.header['CUNIT1'] = 'pixel   '
hduout.header['CUNIT2'] = 'pixel   '
hduout.header['CRPIX1'] = 1. 
hduout.header['CCDSUM'] = '1 2     '
hduout.header['WCSDIM'] = 3
hduout.header['CD1_1'] = dw
hduout.header['CD2_2'] = 1.
hduout.header['LTM1_1'] = 1.
hduout.header['LTM2_2'] = 1.
hduout.header['WAT0_001'] = 'system=equispec'
hduout.header['WAT1_001'] = 'wtype=linear label=Wavelength units=angstroms'
hduout.header['WAT2_001'] = 'wtype=linear'
hduout.header['DCLOG1'] = 'Transform'
hduout.header['DC-FLAG'] = 0
hduout.header['CTYPE3'] = 'LINEAR  '
hduout.header['CD3_3'] = 1.
hduout.header['LTM3_3'] = 1.
hduout.header['WAT3_001'] = 'wtype=linear'
hduout.header['EX-FLAG'] = 0
hduout.header['CA-FLAG'] = 0
hduout.header['BANDID1'] = "Optimally extracted spectrum"
hduout.header['BANDID2'] = "Straight sum of spectrum"
hduout.header['BANDID3'] = "Background fit"
hduout.header['BANDID4'] = "Sigma per pixel."
hduout.header['APNUM1'] = '1 1 %7.2f %7.2f' % (apmin, apmax)


hduout.writeto(OBJIMAGE.stem+".ms_1d.fits", overwrite=True)
print(" ")
print("** Wrote output file '%s.ms_1d.fits' ." % OBJIMAGE.stem)
print(" ")

#Also write spectrum to an ascii-file
df_frame = {'wave':wavelength, 'optflux':out1, 'sumflux':out2, 'sky':out3, 'opt_sigma':out4}
df = pd.DataFrame(df_frame,dtype='float32')
df.to_csv(OBJIMAGE.stem+'.ms_1dw.dat', header=None, index=None, sep=' ')
print(" ")
print("** Wrote output file '%s.ms_1d.dat' ." % OBJIMAGE.stem)
print(" ")

#Finally plot the extracted spectrum
fig = plt.figure(figsize=(10,5))
plt.xlim(3600,9300)
ii = (wavelength > 3600) & (wavelength < 9300)
plt.ylim(0,np.amax(out1[ii])*1.2)
plt.xlabel('lambda i Å')
plt.ylabel('Flux')

plt.xlabel('Observed wavelength [Å]')
plt.ylabel('Flux [ADU/sec]')

plt.plot(wavelength, out2, lw = 1,
         alpha=0.5, color='r', label='1d summed spectrum')
plt.plot(wavelength, out1, lw = 1,
         alpha=0.75, label='1d Horne-extracted spectrum', color='b')
plt.plot(wavelength, out4, lw=1, color='b')
plt.legend()
plt.show()

