This repo is dedicated to provide an IRAF-like reduction scheme for long-slit spectra using Python.
The science-file and arc-lamp file are supposed to be made before as fits-files with the same size and dispersion along rows.

Like iraf we try to keep output from scripts in the database folder.


The order of running the scripts are:

The script setup.py defines all the packages and gobal parameters. It is run by all the other scripts.

1) identify.py: identify arclines. 

2) fitarcdata.py: fit a checychef polynomium to the pixel to wavelength relation. 

3) extract1d.py: 2d wavelength calibration and extraction of the wavelength calibrated 1d spectrum

Missing:
1) Update identify such that it becomes iterative (i.e. reads an existing set of measurements and updtes to that. Also, it should be possible to delete lines from the list.
2) the equivalents of transform, standard and sensfunc in IRAF
3) implementation of optical extraction
4) implementation of output noise-spectrum
