This repo is dedicated to provide an IRAF-like reduction scheme for long-slit spectra using Python.
The science-file and arc-lamp file are supposed to be made before as fits-files with the same size and dispersion along rows.

Like iraf we try to keep output from scripts in the database folder.


The order of running the scripts are:

The script setup.py defines all the packages and gobal parameters. It is run by all the other scripts.

1) identify.py: identify arclines. 

2) fitarcdata.py: fit a checychef polynomium to the pixel to wavelength relation. 

3) extract1d.py: 2d wavelength calibration and extraction of the wavelength calibrated 1d spectrum

Missing:
1) It should be possible to delete lines from the idarc list.
2) the equivalents of 'fitcoords', 'transform', 'standard', 'sensfunc' and 'calibrate' in IRAF.
3) implementation of optimal extraction (ala Keith Horne).
4) implementation of output noise-spectrum.
