This repo is dedicated to provide an IRAF-like reduction scheme for long-slit spectra using Python. This is not because IRAF is bad (it is not), but because it is no longer supported so we need to find other solutions that easy to install and easy to introduce, e.g. as part of teaching.

The philosophy: let's first make something quickly that works. Then we can refine it. Otherwise, I fear it will just go nowhere. We will assume that the science-file and arc-lamp file are made before running these reduction scripts. These files are called:

spec1.fits - science file

arcsub.fits - arc-line file

(in this case a grism 4 spectrum of a quasar taken with the instrument Alfosc mounted on the Nordic Optical Telescope).

They should be  fits-files with the same size and with dispersion along rows.

There are also files for a standard star:

std.fits - standard star observation 

(http://www.ing.iac.es/Astronomy/observing/manuals/html_manuals/tech_notes/tn065-100/hd849.html)

arcsub_std.fits - arc-line file 


We have started from Yoonsoo Bach's notebook on github:
https://nbviewer.jupyter.org/github/ysBach/SNU_AOclass/blob/master/Notebooks/Spectroscopy_in_Python.ipynb?fbclid=IwAR22YsWpk-uNw7Iz9LGolRD6kbtpcTeqmYDKgfeRIQHQ42M8OLfRbRzJmeY

That we have used as a skeleton to start from.


Like iraf we try to keep output from scripts in the database folder.

The order of running the scripts are:

The script setup.py defines all the packages and gobal parameters. It is run by all the other scripts.


1) identify.py: identify arclines. 


2) fitarcdata.py: fit a checychef polynomium to the pixel to wavelength relation. 


3) extract1d.py: 2d wavelength calibration and extraction of the wavelength calibrated 1d spectrum.



Missing (obviously a lot):
1) Ideally, the identify.py and fitarcdata.py should be combined such that is possible to iteratively add and delete lines from the idarc list, fit the lines, go back to add/remove more lines, etc.

2) the equivalents of 'fitcoords', 'transform', 'standard', 'sensfunc' and 'calibrate' in IRAF. A version of standard.py has now been written, but that still needs to be improved.

3) implementation of optimal extraction (ala Keith Horne).

4) implementation of output 1-sigma noise-spectrum.
