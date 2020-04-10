#!/usr/bin/env python
from __future__ import print_function

## opextract.py -- Optimally extracts a 1-d stellar spectrum from a
##   bias-subbed, flatfielded CCD image.

# [Uses python3 style print statements, but has been scrubbed of the
# f-strings that are not back-compatible unless you use a rather
# gnarly compiler directive.]

# By John Thorstensen, Dartmouth College.  2019-Nov

# This task extracts a SINGLE 1-d point-source ('stellar') spec
# from a 2-d long slit spectral image, using algorithms closely
# modeled on Keith Horne's 1986 paper "An Optimal Extraction
# Algoritm for CCD Spectroscopy", PASP v. 98, p. 609.

# It is assumed the image has already been bias-subtracted and
# flatfielded using a normalized flatfield, i.e. that the flat
# has a mean close to 1 everywhere.  Otherwise, the variance
# image will be meaningless, and it's central to the procedure.

# The task assumes that IRAF has been used to find, edit,
# and trace a software aperture for the star (e.g. with apedit
# and aptrace).  It uses this information for the extraction, and
# when finished it writes the result as an IRAF "multipsec" fits
# file. It can therefore be used as a drop-in replacement for
# IRAF's "apextract" in the case of single, point-source spectra.

# There's a fair number of adjustable parameters.  As a general
# strategy I'd suggest extracting interatively with the "-d"
# flag to see intermediate results, and look at profiles in
# a range set by "-s" and "-e" esp. over bright lines or cosmic
# rays to be sure it's doing the right thing.  Once you've come up
# with appropriate defaults for your setup, you can set them as
# defaults in the "argparse" section below.

# Note that the guts of the task assume that the dispersion is
# along COLUMNS (in the Y-direction).
# This can be changed with the --axisdisp argument, which
# simply transposes the input image and changes the interpretation
# of the aperture file.

# For brevity I'll sometimes refer to the along-dispersion
# direction as "columns" or "Y" and cross-dispersion and "rows"
# or "X".

# It is assumed that an IRAF-format "aperture" file exists
# for the image that specifies a software aperture for the
# stellar image and specifies ranges on each side in which to
# determine the background.  This is assumed to be in a subdirectory
# called "database" and be named "apxxxx" where xxxx is the image
# root name (without the .fits).  The apxxxx file contains ranges
# for the object and background spectrum and an "aptrace"
# fit that traces the spectrum along the dispersion direction
# (since columns will not be perfectly square with dispersion).
# The task contains code to read IRAF "aperture" files,
# interpret the coefficients of the curves, and evaluate the
# curves.

# The basic strategy is:
# -- get parameters from the command line and (possibly) elsewhere
# -- read and interpret the aperture file
# -- read the data. If input dispersion is along rows, transpose.
# -- create an image of estiamated variance in each pixel using
#     an assumed read noise and gain.
# -- fit the background in each line and subtract it.
# -- prepare a normalized profile of the star image as a function
#     of wavelength:
#      - median-filter the sky-subbed data to remove cosmic rays
#      - fit each column within the software aperture with a polynomial
#      - normalize the aperture in each line.
# -- loop through the image line-by-line; at each line determine the
#      normalization factor that best matches the sky-subbed data to
#      the profile, weighting the result for optimal S/N.  These norm.
#      factors form the optimally weighted spectrum.
# -- also compute the standard deviation.
# -- in computing the normalization factor, iteratively ignore pixels
#      that differ significantly from the expected profile.  These
#      are generally CRs.
# -- also compute a straight sum of the flux within the aperture.
# -- .... and a running sum of data excluding CRs.
# -- When done, adjust the optimally extracted spec so its overall
#      normalization agrees with the cr-rejected straight sum.
# -- to close, stack the optimally extracted spec, straight sum,
#      background spec, and estimated sigma in each pixel into
#      the form expected by an IRAF multispec, edit the header
#      appropriately, and write the data out.
#
# Again, note that setting "-d" or "--diagnostic" shows detailed information
# and plots, and also writes out the images of the sky-subtracted data,
# variance, median-smoothed data, and normalized image profile.
# Also, giving values for "-s" and "-e" will show line-by-line plots
# over a range of dispersion lines.
# This is extremely useful in testing and tuning the free parameters.

# Remaining issues:
#  --- At very low signal levels, the S/N of the extracted spec
#      is sometimes just a bit worse than in my C program, and
#      the normalization is a tad higher.  High S/N spectra (e.g.
#      std stars) coming out nearly identical.
#  --- May want to implement a pixel rejection that doesn't assume
#      statistical errors are good at very high signal levels.
#      This hasn't been tested on very bright objects.
#  --- may also want to implement an adaptive fitter for the
#      column fits that traces high S/N more aggressively and
#      low S/N less so.
#  --- The background fit function is restricted to a legendre.
#      I've used these consistently over the years.


import sys
import numpy as np
import numpy.polynomial.legendre as leg
import numpy.polynomial.chebyshev as cheb
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import astropy.io.fits as fits
from astropy import modeling
from datetime import datetime

import argparse

# Class to read and interpret IRAF aperture database files.

class aperture_params:
    """parameters read from an IRAF ap database file."""
    def __init__(self, froot="", dispaxis=2):
        if froot:
            froot = froot.replace(".fits", "")
            reading_coeffs = False  # coeffs get parsed differently.
            ncoeffs = 0
            self.coeffs = []
            reading_background = False
            with open("database/ap%s" % froot, "r") as apf:
               for l in apf:
                  if l[0] != '#':
                     if 'background' in l:
                        reading_background = True
                     if 'axis' in l:
                         reading_background = False
                     if 'curve' in l:
                         reading_coeffs = True
                         reading_background = False # just in case.
                     x = l.split()
                     if x: # no blank lines
                         if 'image' in x[0]:
                            self.imroot = x[1]
                         if 'aperture' in x[0]:
                             self.apnum = int(x[1])
                             if self.apnum > 1:
                                 print("Warning -- apnum > 1!")
                                 print("Multiple apertures are not implemented.")
                         if 'enter' in x[0]:
                             if dispaxis == 2:
                                 self.center = float(x[1])
                                 self.displine = float(x[2])
                             else:
                                 self.center = float(x[2])
                                 self.displine = float(x[1])
                         if 'low' in x[0] and 'reject' not in x[0]:
                             # print(l, x)
                             if dispaxis == 2:
                                 self.aplow = float(x[1])
                                 self.lowline = float(x[2])
                             else:
                                 self.aplow = float(x[2])
                                 self.lowline = float(x[1])
                         if 'high' in x[0] and 'reject' not in x[0]:
                             if dispaxis == 2:
                                 self.aphigh = float(x[1])
                                 self.highline = float(x[2])
                             else:
                                 self.aphigh = float(x[2])
                                 self.highline = float(x[1])
                         if reading_background:
                             if 'sample' in x[0]:  # this is not consistently formatted.  Ugh.
                                 self.bckgrintervals = []
                                 if len(x) == 2:   # lower and upper joined by comma and no space
                                     y = x[1].split(',')
                                 else :            # lower and upper parts space-separated.
                                     y = x[1:]
                                 for yy in y:
                                     z = yy.split(':')
                                     # print(z)
                                     self.bckgrintervals.append([float(z[0]), float(z[1])])
                             if 'function' in x[0]:
                                 self.bckgrfunc = x[1]
                             if 'order' in x[0]:
                                 self.bckgrfunc_iraforder = int(x[1])
                             if 'niterate' in x[0]:
                                 self.bckgr_niterate = int(x[1])
                             # rejecting both low and high pixels is a bit awkward later.
                             # low pixels are not much of a problem, usually, so didn't
                             # implement a low-reject scheme.
                             #if 'low_reject' in x[0]:
                             #    self.bckgr_low_reject = float(x[1])
                             if 'high_reject' in x[0]:
                                 self.bckgr_high_reject = float(x[1])
                         if reading_coeffs and 'curve' not in l:
                             self.coeffs.append(float(x[0]))

        else:
           print("Need a valid image name.")

    # These were all done neatly with f-string but reverted for
    # compatibility with python 2.7.  Even after
    # from  __future__ import print_statement
    # f-strings will not parse in pyhon 2.7.

    def repeat_back(self):
        print("root ", self.imroot, ", aperture ", self.apnum)
        print("center ", self.center, " displine ", self.displine)
        print("aplow ", self.aplow, ", aphigh ", self.aphigh)
        print("lowline ", self.lowline, "  highline ", self.highline)
        print("bckgrfunc ", self.bckgrfunc, "iraf_order ", self.bckgrfunc_iraforder)
        print("niterate ", self.bckgr_niterate, " high_reject ", self.bckgr_high_reject)
        print("bckgr intervals:")
        for b in self.bckgrintervals:
           print(" start ", b[0], " end ", b[1])
        print("coeffs:")
        for c in self.coeffs:
           print(c)

    def evaluate_curve(self, pixlims=None):
        ic = irafcurve(self.coeffs)
        # ic.repeat_back()
        y = ic.evaluate_by1(pixlims)
        # curve is relative to center so need to add center to get real value.
        return self.center + y

# May want to reuse irafcurve if I write code to do ident,
# reident, or other things.

class irafcurve:
    """A fit generated by the iraf curvefit routines, e.g. a
       an aptrace and possibly a wavelenght fit (not tested yet tho')."""

    def __init__(self, fitparams):
        # handed a list or tuple of firparams (which includes flags
        # for type of fit) sets up the irafcurve instance.

        # should be an int but it i'nt sometimes
        typecode = int(fitparams[0] + 0.001)

        if typecode == 1: self.curvetype = 'chebyshev'
        elif typecode == 2: self.curvetype = 'legendre'
        elif typecode == 3: self.curvetype = 'spline3'
        else:
            print("Unknown fit type: ", fitparams[0])

        # the 'iraforder' is not the conventional order; it's
        # the number of coefficients for a polynomial, so a
        # a straight line has iraforder = 2.  For a spline3 it is
        # the number of spline segments.

        self.iraforder = int(fitparams[1] + 0.001)

        self.xrange = (fitparams[2], fitparams[3])
        self.span = self.xrange[1] - self.xrange[0]
        self.sumrange = self.xrange[0] + self.xrange[1]

        self.fitcoeffs = np.array(fitparams[4:])

        # Numpy provides built-in legendre and chebyshev apparatuses that make
        # this trivial.  The numpy spline3 is apparently oriented toward interpolation,
        # and wasn't as easy to adapt, though I'm probably missing something.  My own
        # spline3 stuff works correctly though it's more awkward.

        if self.curvetype == 'legendre':
            self.lpoly = leg.Legendre(self.fitcoeffs, domain = [self.xrange[0], self.xrange[1]])

        if self.curvetype == 'chebyshev':
            self.chpoly = cheb.Chebyshev(self.fitcoeffs, domain = [self.xrange[0], self.xrange[1]])

    def repeat_back(self):
        # be sure the fit read correctly
        print("curvetype ", self.curvetype, " iraforder ", self.iraforder)
        print("xrange ", self.xrange)
        print("span, sumrange ", self.span, self.sumrange)
        print("coeffs ", self.fitcoeffs)

    def evalfit(self, x):  # evaluate fit for an array of x-values.
        # translated from C, taking advantage of array arithmetic.

        if self.curvetype == 'spline3':

            # this is by far the most complicated case.

            xnorm = (2. * x - self.sumrange) / self.span
            splcoo = self.iraforder * (x - self.xrange[0]) / self.sumrange
            jlo = splcoo.astype(int)  # this is 0 for pixels in first segment, 1 for 2nd etc
            a = (jlo + 1) - splcoo    # these are basically x-values referred to segment boundaries
            b = splcoo - jlo

            # make four blank arrays

            coef0 = np.zeros(xnorm.shape)
            coef1 = np.zeros(xnorm.shape)
            coef2 = np.zeros(xnorm.shape)
            coef3 = np.zeros(xnorm.shape)

            # fill the arrays piecewise with the appropriate
            # spline coefficients.  Then the evaluation can be
            # done entirely with array arithmentic.

            for i in range(self.iraforder):
                np.place(coef0, jlo == i, self.fitcoeffs[i])
                np.place(coef1, jlo == i, self.fitcoeffs[i+1])
                np.place(coef2, jlo == i, self.fitcoeffs[i+2])
                np.place(coef3, jlo == i, self.fitcoeffs[i+3])

            y = coef0 * a ** 3 + coef1 * (1. + 3. * a * (1 + a * b)) + \
                                 coef2 * (1. + 3. * b * (1 + a * b)) + \
                                 coef3 * b ** 3

            return y

        elif self.curvetype == "legendre":
            return self.lpoly(x)

        elif self.curvetype == "chebyshev":
            return self.chpoly(x)

    def evaluate_by1(self, pixlims=None): # evaluates curve for every pixel in range.
        if pixlims == None:
            firstpix = int(self.xrange[0] + 0.001)
            lastpix = int(self.xrange[1] + 0.001)
        else:
            firstpix = pixlims[0]
            lastpix = pixlims[1]
        pixarr = np.arange(firstpix, lastpix + 1, 1)
        return self.evalfit(pixarr)

####

def fake_multispec_data(arrlist):
   # takes a list of 1-d numpy arrays, which are
   # to be the 'bands' of a multispec, and stacks them
   # into the format expected for a multispec.  As of now
   # there can only be a single 'aperture'.

   return np.expand_dims(np.array(arrlist), 1)


### START MAIN TASK.  Get the input file.
def opextract(imroot, firstlinetoplot, lastlinetoplot, plot_sample, DISPAXIS, readnoise, gain, apmedfiltlength,
              codfitorder, scattercut, colfit_endmask=10, diagnostic=False, production=False):

    if '.fits' in imroot:
        imroot = imroot.replace(".fits", "")
    if '.fit' in imroot:
        imroot = imroot.replace(".fit", "")

    apparams = aperture_params(froot=imroot, dispaxis=DISPAXIS)

    if diagnostic:
        apparams.repeat_back()

    hdu = fits.open(imroot + '.fits')
    hdr = hdu[0].header
    rawdata = hdu[0].data

    # If dispersion does not run along the columns, transpose
    # the data array.  Doing this once here means we can assume
    # dispersion is along columns for the rest of the program.

    if DISPAXIS != 2:
        rawdata = rawdata.T

    # compute a variance image from the original counts, using
    # the specified read noise (in electrons) and gain
    # (in electrons per ADU).  These can be set with command
    # line arguments or (more likely) set as defaults.

    # The algorithm wants variance in data numbers, so divide
    # and square.

    readvar = (readnoise / gain) ** 2  # read noise as variance in units of ADU

    # variance image data is derived from bias-subbed data.
    varimage = readvar + rawdata / gain

    # Creating zero arrays for processed data should be much
    # faster than building them row by row, e.g. for the
    # background-subtracted data:
    subbeddata = np.zeros(rawdata.shape)

    rootpi = np.sqrt(np.pi)  # useful later.

    # Compute aperture and background limits using the
    # parameters read from the database file.

    apcent = apparams.evaluate_curve(pixlims=(0, rawdata.shape[0] - 1))

    # IRAF is one-indexed, so subtract 1 to make it zero-indexed.
    apcent -= 1

    # four arrays give limits of background sections

    bckgrlim1 = apcent + apparams.bckgrintervals[0][0]
    bckgrlim2 = apcent + apparams.bckgrintervals[0][1]
    bckgrlim3 = apcent + apparams.bckgrintervals[1][0]
    bckgrlim4 = apcent + apparams.bckgrintervals[1][1]

    # arrays of limits for aperture
    aplimlow = apcent + apparams.aplow
    aplimhigh = apcent + apparams.aphigh
    # convert to integers for later use
    aplimlowint = np.round(aplimlow).astype(int)
    aplimhighint = np.round(aplimhigh).astype(int)

    lowestap = aplimlowint.min()
    highestap = aplimhighint.max()   # extreme ends of aperture range

    if diagnostic:
        print("lowestap ", lowestap, " highestap ", highestap)

    # Now compute and load the background spectrum by fitting
    # rows one by one.  Start with a zero array:

    ### NOTE that only Legendre fits are implemented in this version. ###

    bckgrspec = np.zeros(apcent.shape)

    # crossdisp is the grid of pixel numbers for
    # fit to background, and to form the x-array
    # for optional plots.

    crossdisp = np.array(range(rawdata.shape[1]))

    # take background fit parameters from input
    # file if they loaded right.

    try:
        niterations = apparams.bckgr_niterate
        low_rej = apparams.bckgr_low_reject
        high_rej = apparams.bckgr_high_reject
    except:
        niterations = 3
        low_rej = 2.
        high_rej = 2.

    # fit and subtract the background.  The region
    # fitted is on each side of the program object.
    # Only legendre fits have been tested.

    for lineindex in range(rawdata.shape[0]):

        ldata = rawdata[:][lineindex]

        # index limits for low side and high side background windows
        ind1  = int(bckgrlim1[lineindex])
        ind2  = int(bckgrlim2[lineindex])
        ind3  = int(bckgrlim3[lineindex])
        ind4  = int(bckgrlim4[lineindex])

        # grab subarrays for low and high side and join

        xlo = crossdisp[ind1:ind2]
        ylo = ldata[ind1:ind2]
        xhi = crossdisp[ind3:ind4]
        yhi = ldata[ind3:ind4]

        xtofit = np.hstack((xlo, xhi))
        ytofit = np.hstack((ylo, yhi))

        # fit and iterate to get rid of bad pixels.

        for iteration in range(niterations):

            # use legendre order from input if function is a leg

            if apparams.bckgrfunc == 'legendre':
                legcoefs = leg.legfit(xtofit, ytofit,
                   apparams.bckgrfunc_iraforder - 1)
            else:  # or default to 2nd order leg if func is something else.
                legcoefs = leg.legfit(xtofit, ytofit, 2)
            fit = leg.legval(xtofit, legcoefs)
            residuals = ytofit - fit
            stdev = np.std(residuals)
            # fancy indexing!
            keepindices = abs(residuals) < high_rej * stdev
            xtofit = xtofit[keepindices]
            ytofit = ytofit[keepindices]

        # Subtract the fit from this line, and store in subbeddta

        subbeddata[lineindex] = rawdata[lineindex] - leg.legval(crossdisp, legcoefs)

        # Keep the 1-d background spec at the center of the image.
        # later this is scaled up to the 'effective' width of the optimally
        # extracted spectrum and written out to the multispec.

        bckgrspec[lineindex] = leg.legval(apcent[lineindex], legcoefs)

    # If keeping diagnostics, write a sky-subtracted image.

    if diagnostic:
        # create a new hdu object around subbeddata
        hduout = fits.PrimaryHDU(subbeddata)
        # copy header stuff
        hdrcopy = hdr.copy(strip = True)
        hduout.header.extend(hdrcopy, strip=True, update=True,
            update_first=False, useblanks=True, bottom=False)
        # and write it.
        hduout.writeto(imroot + "_sub.fits", overwrite=True)

        print("Background-subbed image written to '%s_sub.fits'" % (imroot))

        # Now that we have hduout, write the variance image
        # simply by substituting the data and writing.

        hduout.data = varimage
        hduout.writeto(imroot + "_var.fits", overwrite=True)

        print("Variance image written to '%s_var.fits'" % (imroot))

    # PROFILE FINDING

    # Creates an image of the stellar spectrum profile
    # normalized row by row, i.e, Stetson's "P_lambda".

    # Start by median-filtering the subbed array parallel to the
    # dispersion; this will remove CRs and smooth a bit.

    smootheddata = nd.median_filter(subbeddata, size=(apmedfiltlength, 1), mode='nearest')

    if diagnostic:
        # write out median-smoothed array for diagnostic.
        hduout.data = smootheddata
        hduout.writeto(imroot + "_medfilt.fits", overwrite=True)
        print("Medium-filtered image written to '%s_medfilt.fits'" % (imroot))

    # Find the whole x-range over which we'll extract.  We'll
    # fit only those columns.

    aprange = range(lowestap, highestap+1, 1)

    # Get the range of pixels along the dispersion.
    # OSMOS data needs extreme ends masked a bit.

    pixrange = np.arange(0, smootheddata.shape[0], 1.)

    firstfitpix = colfit_endmask
    lastfitpix = smootheddata.shape[0] - colfit_endmask
    fitrange = np.arange(firstfitpix, lastfitpix, 1.)

    # This array will contain the normalized 2-d aperture.

    apdata = np.zeros(smootheddata.shape)

    # go column by column (parallel to dispersion), and fit
    # a polynomial to the smoothed spec in that column.

    # the order needs to be set high enough to follow
    # odd bumps for the echelle chip.
    # now removed to hard-coded param list
    # colfitorder = 15

    for i in aprange:

        # Diagnostics gives a nice plot of the columns and their fits, which can
        # be very enlightening.  First, the smoothed data:

        if diagnostic:
            plt.plot(pixrange, smootheddata[:, i])

        legcoefs = leg.legfit(fitrange, smootheddata[firstfitpix:lastfitpix, i], colfitorder)

        thisfit = leg.legval(pixrange, legcoefs)
        # plot fit for diagnostic.
        if diagnostic:
            plt.plot(pixrange, thisfit)
        apdata[:, i] = thisfit

    # mask values less than zero
    # this may bias spec in very low s/n situations, but
    # it saves lots of trouble.

    apdata[apdata < 0.] = 0.

    # normalize across dispersion to create Horne's profile
    # estimate 'P'.  This is redundant as the aperture is later
    # restricted to those within the pixel limits for that row,
    # but 'norm' is useful later.

    norm = np.sum(apdata, axis=1)

    # if there are no pixels, norm is zero.  Replace those with
    # ones to avoid NaNs in the divided array.

    norm = np.where(norm == 0., 1., norm)

    # show accumulated graphs, which are of the fits of aperture
    # along the dispersion together with the median-smoothed
    # spectrum.

    if diagnostic:
        plt.title("Smoothed column data and poly fits.")
        plt.xlabel("Pixel along dispersion")
        plt.ylabel("Counts (not normalized)")
        plt.show()

    # finally, normalize the aperture so sum across
    # dispersion is 1, making Horne's "P".
    # (again, this is redundant)

    nn = norm.reshape(rawdata.shape[0], 1)
    apdata = apdata / nn

    # Do something rational to normalize the
    # sky background.  Let's try this:
    # - get the width of the spectrum as a sigma (from a
    #     gaussian fit)   at a spot where it's likely to be
    #     strong -- the displine from the aperture file is
    #     likely to be good.
    # - multiply sigma by root-pi.  This is like an 'effective
    #     width' of the aperture -- it's basically how much
    #     sky effectively falls in the aperture.
    # - Use this width to renormalize the sky spec, which is
    #     per-pixel.
    # - This whole exercise may be silly, but it's reasonably
    #   cheap.

    goodapline = apdata[int(apparams.displine), aprange]

    # use the nice astropy Gaussian fit routines to fit profile.

    fitter = modeling.fitting.LevMarLSQFitter()
    model = modeling.models.Gaussian1D(amplitude = np.max(goodapline), mean = np.median(aprange), stddev=1.)
    fittedgau = fitter(model, aprange, goodapline)
    skyscalefac = rootpi * fittedgau.stddev.value

    if diagnostic:

        # diagnostic to show fidicual profile.

        plt.plot(aprange, goodapline)
        plt.plot(aprange, fittedgau(aprange))
        plt.title("Fiducial profile (from row %5.0f)." % (apparams.displine))
        plt.show()

    #
    # Here comes the MAIN EVENT, namely extraction of the
    # spectrum and so on.
    #

    # Create 1-d arrays for the results.

    optimally_extracted = np.zeros(rawdata.shape[0])
    straight_sum = np.zeros(rawdata.shape[0])
    sigma_spec = np.zeros(rawdata.shape[0])

    # keep a summation of the non-optimally-extracted spectra
    # for later renormalization.

    cr_corrected_overall_flux = 0.

    # keep track of how many pixels are rejected.
    nrej_pixel = 0       # total number of 2d pixels affected
    corrected_specpts = 0   # total number of spectral points affected
    n_profiles_substituted = 0 # and number of profiles replaced with
               # Gaussian fits because of terrible S/N.

    # This optimal extraction process is very tricky.  For
    # development it's useful to look at the detailed calculation
    # for several rows.  I've built in the ability to
    # plot a range of rows and print out some data on them.
    # This is invoked now with command-line arguments.

    # We'll compute a statistic for how badly a pixel deviates from
    # the expected profile and ignore those that are bad enough.

    # This maximum accepted 'badness' statistic is set elsewhere now.
    # scattercut = 25.   # Horne's suggested value is 25.

    # Finally, extract line-by-line:

    for lineindex in range(rawdata.shape[0]):

        # are we plotting this line out (etc.)?
        # This logic is separate from "--diagnostic".
        showdiagn = False
        if plot_sample and lineindex >= firstlinetoplot and lineindex <= lastlinetoplot:
            showdiagn = True

        # Compute straight sum of sky-subbed data in aperture, without pixel
        # rejection.  In principle we could edit the CRs out of the straight sum but
        # that would require interpolating across them somehow.

        # Keep a copy of data in the aperture for later renormalization of the optimal
        # extraction.

        in_ap_data = subbeddata[lineindex, aplimlowint[lineindex]:aplimhighint[lineindex]]

        straight_sum[lineindex] = np.sum(in_ap_data)

        # I'm getting bad results in very poor S/N parts of the spectrum
        # where the aperture isn't well-defined even after all that smoothing.
        # Here's an attempt to fix this -- in cases where the S/N is terrible,
        # replace the aperture with the gaussian fit to the supposedly good line,
        # recentered.  Here we can re-use the gaussian profile fit computed
        # earlier for the "fiducial" line, shifted to the local aperture
        # center from aptrace.  This ignores any profile variations along the
        # dispersion, but the spec is barely detected anyway.

        # The norm array from before has preserved the cross-dispersion sums
        # pre-normalization.  Use this to decide whether to substitute the
        # fit for the empirical profile.

        if norm[lineindex] < readnoise / gain:  # basically nothing there.
            apmodel = modeling.models.Gaussian1D(amplitude =
                 rootpi/fittedgau.stddev.value, mean = apcent[lineindex],
                 stddev = fittedgau.stddev.value)
            apdata[lineindex] = apmodel(range(apdata.shape[1]))
            n_profiles_substituted = n_profiles_substituted + 1

        # 'pixind' is the array of pixel numbers to be included.
        # When pixels are rejected, the mechanism used is to delete
        # their index from pixind.

        # To start, include only pixels where the aperture is
        # at least positive and a bit.

        pixind0 = np.where(apdata[lineindex] > 0.001)
        # this weirdly returned a tuple that would not do fancy indexing:
        pixinds = np.array(pixind0)

        # Include only those pixels that are within the aperture
        # in this row.
        pixinds = pixinds[pixinds >= aplimlowint[lineindex]]
        pixinds = pixinds[pixinds <= aplimhighint[lineindex]]

        # renormalize apdata to the values within the aperture.
        # This is assumed to contain 'all the light'.

        validsum = np.sum(apdata[lineindex, pixinds])
        apdata[lineindex, pixinds] = apdata[lineindex, pixinds] / validsum

        worst_scatter = 10000. # initialize high to make the loop start.

        largest_valid_stat = 40. # this isn't used yet.
        iteration = 0

        # "while" loop to iterate CR rejection. Second
        # condtion guards against case of no valid aperture points.

        while worst_scatter > scattercut and pixinds.size > 0:

            # Horne eq'n (8):
            numerator = np.sum(apdata[lineindex, pixinds] * subbeddata[lineindex, pixinds] / varimage[lineindex, pixinds])
            denominator = np.sum(apdata[lineindex, pixinds] ** 2 / varimage[lineindex, pixinds])
            optimally_extracted[lineindex] = numerator/denominator

            # Horne eq'n (9) for variance, square-rooted to get sigma:
            sigma_spec[lineindex] = np.sqrt(1. / (np.sum(apdata[lineindex, pixinds] ** 2 / varimage[lineindex, pixinds])))

            # The procedure for eliminating cosmic rays and other discrepant profile points
            # follows; it's taken from Horne's article, page 614, top right.

            # compute Horne's measure of anomalous pixels due to CRs or whatever.

            # NOTE that an inaccurate profile estimate will lead to spurious 'bad' pixels.
            # May want to put in something to relax the rejection criterion for bright objects.

            scatter_array = ((subbeddata[lineindex, pixinds] - optimally_extracted[lineindex] * apdata[lineindex, pixinds])**2 / varimage[lineindex, pixinds])

            # array of S/Ns to assess validity of stat model - not yet used.
            sn_array = subbeddata[lineindex, pixinds] / np.sqrt(varimage[lineindex, pixinds])

            if showdiagn:   # examine the fit in this row in detail.
                print("scatter_array ", scatter_array, " shape ", scatter_array.shape)
                print("sn_array", sn_array)

            worst_scatter = np.max(scatter_array)

            if worst_scatter > scattercut:   # reject bad pixels

                # find and delete bad pixel.  This will fail if there are two equal
                # values of scatter_array, but they are floats so the chance of this
                # happening is minuscule.

                index_of_worst = np.where(scatter_array == worst_scatter)[0][0]
                pixinds = np.delete(pixinds, index_of_worst)

                if showdiagn:
                    print("worst: ", worst_scatter, "killed index ", index_of_worst)

                # Also edit out the high point from the in_ap_data so it doesn't skew the
                # later overall renormalization too badly.

                bad_point_value = subbeddata[lineindex, index_of_worst]
                in_ap_data = in_ap_data[in_ap_data != bad_point_value]

                # re-normalize the remaining aperture points.
                # *** This was an error!! ***  Just omit the point, and keep normalization.
                # validsum = np.sum(apdata[lineindex, pixinds])
                # apdata[lineindex, pixinds] = apdata[lineindex, pixinds] / validsum

                # keep track of how many pixels were rejected, and how
                # many spectral points are affected.

                nrej_pixel += 1
                if iteration == 0:
                    corrected_specpts += 1

            if len(pixinds) == 0:  # Uh-oh -- out of pixels!
                worst_scatter = 0.  # will kick us out of loop.
                optimally_extracted[lineindex] = 0.

            iteration += 1

        if len(pixinds) == 0:  # can be zero because aperture is all zero.
            optimally_extracted[lineindex] = 0.
            sigma_spec[lineindex] = 10.  # arbitrary

        # accumulate sum of flux in non-rejected straight sum points.
        cr_corrected_overall_flux += np.sum(in_ap_data)

        # plot some sample lines for diagnostic if indicated.

        if showdiagn:
            lowx = aplimlowint[lineindex]   #brevity
            highx = aplimhighint[lineindex]
            plrange = range(lowx - 15, highx + 15)
            # plot aperture profile * estimate
            plt.plot(plrange, apdata[lineindex, plrange] * optimally_extracted[lineindex])
            # and also the actual sky-subtracted data.
            plt.plot(plrange, subbeddata[lineindex, plrange])

            # also plot vertical bars at pixel limits, and dots at pixels that were used.
            plt.plot((lowx, lowx), (-10, 50))
            plt.plot((highx, highx), (-10, 50))
            pixpl = np.zeros(pixinds.shape[0])
            plt.plot(pixinds, pixpl, 'bo')
            plt.title("Line %d  optextr %8.2f " % (lineindex, optimally_extracted[lineindex]))
            plt.show()

    if diagnostic:
        # write aperture image (as amended by extraction) for a diagnostic.
        hduout.data = apdata
        hduout.writeto(imroot + "_aperture.fits", overwrite=True)
        print("Normalized aperture image written to '%s_aperture.fits'" % imroot)
        print("(These diagnostic images are purely for your dining and dancing")
        print("pleasure, and can be safely deleted.)")
        print(" ")

    # Finally, normalize the optimally extracted spec to
    # the cr-rejected straight sum.

    normfac = cr_corrected_overall_flux / np.sum(optimally_extracted)
    if diagnostic:
        print("overall flux %8.0f, sum of optimal extr. %8.0f, norm. fac %7.5f" %
            (cr_corrected_overall_flux, np.sum(optimally_extracted), normfac))
    optimally_extracted *= normfac

    # EXTRACTION IS COMPLETE!

    # WRITE OUT AS A MULTISPEC FITS FILE.

    ultimate = rawdata.shape[0] - 1      # last and second-to-last indices
    penultimate = rawdata.shape[0] - 2

    if DISPAXIS == 2:

        # For modspec data -- and presumably for other column-dispersion ---

        # Comparison with previous extractions show an off-by-one!
        # Never could figure out why, so shift output arrays by one
        # pixel with np.roll, and repeat the last pixel.
        # Could also do this with indexing I'm thinking that
        # the implementation of np.roll is likely to be faster (?).

        ultimate = rawdata.shape[0] - 1
        penultimate = rawdata.shape[0] - 2

        out1 = np.roll(optimally_extracted, -1)
        out1[ultimate] = out1[penultimate]

        out2 = np.roll(straight_sum, -1)
        out2[ultimate] = out2[penultimate]

        out3 = np.roll(bckgrspec * skyscalefac, -1)
        out3[ultimate] = out3[penultimate]

        out4 = np.roll(sigma_spec, -1)
        out4[ultimate] = out4[penultimate]

    else:  # OSMOS data (dispaxis = 1) doesn't have this issue
        # Code to fix a bad pixel at high end left in place commented
        # out in case it's needed
        out1 = optimally_extracted
        # out1[ultimate] = out1[penultimate] # fudge the last pixel.
        out2 = straight_sum
        # out2[ultimate] = out2[penultimate] # fudge the last pixel.
        out3 = bckgrspec * skyscalefac
        out4 = sigma_spec

    # stack in the same array configuration as a multispec

    multispecdata = fake_multispec_data((out1, out2, out3, out4))

    # And write it to a fits spectrum faked to look exactly
    # as if it were written by IRAF apsum.

    hduout = fits.PrimaryHDU(multispecdata)

    # copy over the header from the original spectrum.

    hdrcopy = hdr.copy(strip = True)
    hduout.header.extend(hdrcopy, strip=True, update=True,
            update_first=False, useblanks=True, bottom=False)

    # Add an whole pile of other information including a
    # faked WCS.

    hduout.header['HISTORY'] = "From development of opextract.py"
    hduout.header['HISTORY'] = "ap center %6.2f defined at line %6.2f" % \
              (apparams.center, apparams.displine)
    hduout.header['HISTORY'] = "sky limits  %6.2f %6.2f %6.2f %6.2f" % \
          (apparams.bckgrintervals[0][0], apparams.bckgrintervals[0][1],
           apparams.bckgrintervals[1][0], apparams.bckgrintervals[1][1])
    rightnow = datetime.now().strftime("%a %Y-%m-%dT%H:%M:%S")
    hduout.header['HISTORY'] = "Extracted %s" % (rightnow)
    hduout.header['WCSDIM'] = 3
    hduout.header['WAT0_001'] = 'system=equispec'
    hduout.header['WAT1_001'] = 'wtype=linear label=Pixel'
    hduout.header['WAT2_001'] = 'wtype=linear'
    hduout.header['CRVAL1'] = 1
    hduout.header['CRPIX1'] = 1
    hduout.header['CD1_1'] = 1
    hduout.header['CD1_2'] = 1
    hduout.header['CD1_3'] = 1
    if DISPAXIS != 2:  # splot complains about OSMOS spectra unless these
                        # are set.
        hduout.header['LTM1_1'] = 1
        hduout.header['LTM2_2'] = 1
    hduout.header['LTM3_3'] = 1
    hduout.header['WAT3_001'] = 'wtype=linear'

    hduout.header['BANDID1'] = "Optimally extracted spectrum"
    hduout.header['BANDID2'] = "Straight sum of spectrum; no CR cleaning"
    hduout.header['BANDID3'] = "Background fit (per cross-dispersed pixel)"
    hduout.header['BANDID4'] = "Sigma per pixel."
    hduout.header['APNUM1'] = '1 1 %7.2f %7.2f' % (apparams.aplow, apparams.aphigh)

    if production:
        hduout.writeto(imroot + ".ms.fits", overwrite=True)
    else:
        hduout.writeto(imroot + ".ms_test.fits", overwrite=True)
        print(" ")
        print("** Wrote output file '%s.ms_test.fits' ." % (imroot))
        print("** To turn off the  '_test' suffix for production data, specify '-p'. ** ")
        print(" ")

    # especially in production it's good to be able to see progress.  An
    # excessive number of bad pixels also tends to indicate something wrong,
    # so good to expose it.

    print(imroot, ": rejected ", nrej_pixel, " pixels, affecting ", corrected_specpts, " spectral points.")

    # Admire the result.

    if diagnostic:
        try:
            objname = hdr['OBJECT']
        except:
            objname = ''
        plt.plot(pixrange, optimally_extracted)
        plt.plot(pixrange, sigma_spec)
        # plt.plot(pixrange, straight_sum)
        plt.title("%s - %s opextract spec. and background." % (imroot, objname))
        plt.xlabel("Pixel")
        plt.ylabel("Counts")
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Use IRAF-defined aperture to optimally extract multispec from 2-d spectrum.")
    parser.add_argument('imroot', help="Root image name without '.fits'")

    ### ADJUSTABLE PARAMETER DEFAULTS ARE SET HERE.
    ### Edit as needed once you've determined what's best for your data.

    parser.add_argument('--axisdisp', '-a', help="1=dispersion along row, 2 = along col.", type=int, default=1)
    parser.add_argument('--readnoise', '-r', help="read noise in electrons per pixel.", type=float, default=6)
    parser.add_argument('--gain', '-g', help="CCD gain in electrons per ADU.", type=float, default=2.4)
    parser.add_argument('--medfiltlen', '-m', help="median filter len for column fit", type=int, default=61)
    parser.add_argument('--colfitord', '-c', help="polynomial order for column fit", type=int, default=15)
    parser.add_argument('--badpixthresh', '-b', help="bad pixel rejection threshold", type=float, default=25.)
    parser.add_argument('--production', '-p', help="Write .ms file (not .ms_test).", action='store_true')

    ### These parameters control diagnostic views and writing images of subtracted data,
    ### smoothed data, and sofware aperture.

    parser.add_argument('--diagnostic', '-d', help="Make diagnostic images and plots", action='store_true')
    # You can also view line-by-line plots of the data, profile, and included pixels.
    parser.add_argument('--startplot', '-s', help="Starting line for cross-section diagnostic",
        type=int, default=0)
    parser.add_argument('--endplot', '-e', help="ending line for cross-section diagnostic",
         type=int, default=0)

    ###

    args = parser.parse_args()

    # a little logic to decode line-by-line plot stuff.
    firstlinetoplot = int(args.startplot)
    lastlinetoplot = int(args.endplot)
    if firstlinetoplot != 0:
       plot_sample = True
       if lastlinetoplot == 0: # plot 10 lines if end not specified.
           lastlinetoplot = firstlinetoplot + 10
    else:
       plot_sample = False

    DISPAXIS = args.axisdisp  # 1 along rows (along X), 2 along cols (along Y)

    readnoise = args.readnoise
    gain = args.gain

    apmedfiltlength = args.medfiltlen    # median filter length along dispersion
    colfitorder = args.colfitord      # order of polynomial fit along columns.
    scattercut = args.badpixthresh   # pixels that do not fit the profile (with
                                     # badness of fit larger than this) are ignored.
                                     # Horne's suggested value is 25.

    ## Only one relatively minor HARD-CODED PARAMETER.

    colfit_endmask = 10   # How many pixels to ignore at the ends of column

    ## END OF HARD-CODED PARAMETERS ######

    opextract(args.imroot, firstlinetoplot, lastlinetoplot, plot_sample, DISPAXIS, readnoise, gain, apmedfiltlength,
              colfitorder, scattercut,
              colfit_endmask=colfit_endmask, diagnostic=args.diagnostic, production=args.production)
