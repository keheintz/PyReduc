#Run the setup script
exec(open("setup.py").read())

# =============================================================================
# transform
# =============================================================================

# Read idarc from identify.py
try:
    data = np.loadtxt('database/idarc.txt')
    pixnumber = data[:,0]
    wavelength = data[:,1]
except IOError:
    print("File not accessible")

#Strategy: the 2d spectrum is divided in to bins along the spatial axis of length STEP_REID

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

#Now a strategy is to produce a 2d frame with the same number of pixels is the input image and with a dispersion dictated by the 
#condition that there should be a constant dispersion per pixel.
ap_wavelen_start = fit2D_REID([0],[N_SPATIAL/2])
ap_wavelen_end = fit2D_REID([N_WAVELEN],[N_SPATIAL/2])
dw = (ap_wavelen_end-ap_wavelen_start)/(float(N_WAVELEN)-1)
wavelength = ap_wavelen_start+range(0,N_WAVELEN)*dw

#objimage  = objhdu[0].data
outimage = objimage*0.
for nrow in range(N_SPATIAL):
     row = np.array(range(0,N_WAVELEN))*0.+nrow
     x = fit2D_REID(np.array(range(0,N_WAVELEN)),row)
     y = objimage[nrow,:]
     xisfin = np.isfinite(x)
     f = interp1d(x[xisfin],y[xisfin],fill_value="extrapolate", kind='cubic')
     outimage[nrow,:] = f(wavelength[:])

data_trans = np.array(outimage)
hdr = objhdu[0].header
hdr.add_history(f"rectified using transform.py")
#del hdr['CD2_1']
#del hdr['CD1_2']
#del hdr['LTM1_2']
#del hdr['LTM2_2']
hdr.set('CD1_1', dw[0], 'dispersion')
hdr.set('CD2_2', 1., )
hdr.set('CDELT1', dw[0], 'dispersion', after=181)
hdr.set('CTYPE1', 'LINEAR  ')
hdr.set('CTYPE2', 'LINEAR  ')
hdr.set('CRVAL1', wavelength[0], 'X at reference point', after=156)
hdr.set('CRVAL2', 1., 'Y at reference point', after=156)
hdr.set('CRPIX1', 1.)
hdr.set('CRPIX2', 1.)
hdr.set('LTM1_1', 1.)
hdr.set('LTM2_1', 1.)
hdr.set('DISPAXIS', 1)
hdr.set('WAT0_001', 'system=world')
hdr.set('WAT1_001', 'wtype=linear label=Wavelength units=angstroms')
hdr.set('WAT2_001', 'wtype=linear')
_ = fits.PrimaryHDU(data=data_trans, header=hdr)
_.data = _.data.astype('float32')
_.writeto(DATAPATH/(OBJIMAGE.stem+".trans.fits"), overwrite=True)

#Also make a version with the sky subtracted
for ncol in range(N_WAVELEN):
     outimage[:,ncol] = outimage[:,ncol]-np.median(outimage[:,ncol])

data_trans = np.array(outimage)
hdr = objhdu[0].header
hdr.add_history(f"rectified using transform.py")
#del hdr['CD2_1']
#del hdr['CD1_2']
#del hdr['LTM1_2']
#del hdr['LTM2_2']
hdr.set('CD1_1', dw[0], 'dispersion')
hdr.set('CD2_2', 1., )
hdr.set('CDELT1', dw[0], 'dispersion', after=181)
hdr.set('CTYPE1', 'LINEAR  ')
hdr.set('CTYPE2', 'LINEAR  ')
hdr.set('CRVAL1', wavelength[0], 'X at reference point', after=156)
hdr.set('CRVAL2', 1., 'Y at reference point', after=156)
hdr.set('CRPIX1', 1.)
hdr.set('CRPIX2', 1.)
hdr.set('LTM1_1', 1.)
hdr.set('LTM2_1', 1.)
hdr.set('DISPAXIS', 1)
hdr.set('WAT0_001', 'system=world')
hdr.set('WAT1_001', 'wtype=linear label=Wavelength units=angstroms')
hdr.set('WAT2_001', 'wtype=linear')
_ = fits.PrimaryHDU(data=data_trans, header=hdr)
_.data = _.data.astype('float32')
_.writeto(DATAPATH/(OBJIMAGE.stem+".trans.skysub.fits"), overwrite=True)

