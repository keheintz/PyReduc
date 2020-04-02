#The equivalent of "calibrate" in IRAF/onedspec
#https://github.com/joequant/iraf/blob/master/noao/onedspec/doc/calibrate.hlp
#As input the script requires: 
#- sens_order.dat (the order of the chebychev fit)
#- sens_coeff.dat (the chebychev coefficients). 
#- the extracted, wavelength calibrated science-file (from extract1d.py)
#
#The output file is the flux-calibrated science-file.

#Run the setup script
exec(open("setup.py").read())

#Read the sensitivity-solution
chebord = int(np.loadtxt('database/sens_order.dat'))
datacheb = np.loadtxt('database/sens_coeff.dat', delimiter="\n")
print(datacheb)
