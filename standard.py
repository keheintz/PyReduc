#The equivalent of "standard" in IRAF/onedspec
#https://astro.uni-bonn.de/~sysstw/lfa_html/iraf/noao.onedspec.standard.html

#Run the setup script
exec(open("setup.py").read())

#Open file for output
f = open('database/stdinfo', 'w')

#name of standard star
#stdnm = 'gd71'
#stdnm = 'feige110'
stdnm = 'bd33d2642'

#get information from header
std_file = fits.open(STDIMAGE)
hdr = std_file[0].header
exptime = hdr['EXPTIME']
airmass = hdr['AIRMASS']

#read in the standard star measurements
std_data = np.loadtxt('std.ms_1d.dat')
lam = std_data[:,0]
stdcounts = std_data[:,1]

f.write("%s  %s  %s\n" % ('#', stdnm, 'std.ms_1d.dat'))
f.write("%5.1f %4.3f\n" % (exptime, airmass))
f.close

#Read the file with the flux measurements
#SET .Z.UNITS = "micro-Janskys"
#SET .Z.LABEL = "Flux"
#SET .TABLE.BANDWIDTH = 40
std_ref_data = np.loadtxt('m'+stdnm+'.dat')
reflam = std_ref_data[:,0]
refflux = std_ref_data[:,1]
bandwidth = 40.

#Plot measured standard spectrum
plt.figure()
plt.xlim(3300,9500)
plt.ylim(0,np.amax(stdcounts)*1.2)
plt.xlabel('lambda i Å')
plt.ylabel('Counts')
plt.xlabel('Observed wavelength [Å]')
plt.ylabel('Counts per sec')
plt.plot(lam,stdcounts, lw = 1,
         alpha=0.5, label='1d extracted standard star spectrum')
plt.legend()

#Overplot boxes with reference flux measurements
for n in range(0,len(reflam)):
     wl = reflam[n]
     if (wl > np.amin(lam)) & (wl < np.amax(lam)):
           window = (lam > wl-0.5*bandwidth) & (lam < wl+0.5*bandwidth)
           maxflux = np.amax(stdcounts[window])
           minflux = np.amin(stdcounts[window])
           plt.plot([wl-0.5*bandwidth,wl-0.5*bandwidth],[minflux,maxflux],color='r', lw=1.5)
           plt.plot([wl+0.5*bandwidth,wl+0.5*bandwidth],[minflux,maxflux],color='r', lw=1.5)
           plt.plot(lam[window],lam[window]/lam[window]+minflux,color='r', lw=1.5)
           plt.plot(lam[window],lam[window]/lam[window]+maxflux,color='r', lw=1.5)

#Click on red boxes that should be deleted.
print('Double left-click on red boxes that should be deleted. Other mouse-clicks to exit.')
deleted = list()
get_new_line = True
while get_new_line:
    points = plt.ginput(n=1, timeout=30, show_clicks=True, mouse_add=1, mouse_stop=2)
    if len(points) == 1:
        pix_ref, _ = points[0]
        select = np.abs(reflam - pix_ref) < bandwidth/2
        for wl in (reflam[select]): 
              deleted.append([wl])
              window = (lam > wl-0.5*bandwidth) & (lam < wl+0.5*bandwidth)
              maxflux = np.amax(stdcounts[window])
              minflux = np.amin(stdcounts[window])
              plt.plot([wl,wl],[minflux,maxflux],marker="|", color='b', markersize=20)  
    else:
        answer = input("End [Y/N]?: ")
        if answer.lower() in ['yes', 'y', '']:
             get_new_line = False
             plt.close("all")
 
plt.show()

#Write to file
#Convert micro-Jansky to erg/s/cm/AA (https://en.wikipedia.org/wiki/AB_magnitude)
#flam = refflux/1.e6/3.34e4/reflam**2
flam = 2.998e18*10**(-(refflux+48.6)/2.5)/reflam**2

f = open('database/stddata', 'w')
for n in range(0,len(reflam)):
     wl = reflam[n]
     if (wl > np.amin(lam)) & (wl < np.amax(lam)) & (wl not in deleted) & (np.abs(wl-7650.) > 80) & (np.abs(wl-6900.) > 60:
           window = (lam > wl-0.5*bandwidth) & (lam < wl+0.5*bandwidth)
           f.write("%.0f   %.3e   %.2f    %.1f\n" %(wl,flam[n],bandwidth,np.mean(stdcounts[window])))
print('Output files stdinfo and stddata have been written to the database.')

f.close
