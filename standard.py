#The equivalent of "standard" in IRAF/onedspec

#Run the setup script
exec(open("setup.py").read())

#Open file for output
f = open('std', 'w')

#name of standard star
stdnm = 'hd849'

#get information from header
std_file = fits.open('std.fits')
hdr = std_file[0].header
exptime = hdr['EXPTIME']
airmass = hdr['AIRMASS']

#read in the standard star measurements
std_data = np.loadtxt('std_1dw.dat')
lam = std_data[:,0]
stdcounts = std_data[:,1]/exptime

f.write("%s  %s  %5.1f %4.3f\n" % (stdnm, 'std_1dw.dat', exptime, airmass))

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
         alpha=0.5, label='1d extracted spectrum')
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
deleted = list()
get_new_line = True
while get_new_line:
    points = plt.ginput(n=1, timeout=30, show_clicks=True, mouse_add=1, mouse_stop=2)
    if len(points) == 1:
        pix_ref, _ = points[0]
        select = np.abs(reflam - pix_ref) < bandwidth/2
        print(reflam[select])
        for wl in (reflam[select]): 
              deleted.append([wl])
              window = (lam > wl-0.5*bandwidth) & (lam < wl+0.5*bandwidth)
              maxflux = np.amax(stdcounts[window])
              minflux = np.amin(stdcounts[window])
              plt.plot([wl],[(minflux+maxflux)/2.],marker="X", color='b', markersize=20)  
    else:
        print("End?")
        answer = input(" [Y/N] : ")
        if answer.lower() in ['yes', 'y', '']:
             get_new_line = False
             plt.close("all")
 
plt.show()
print(deleted)

#Write to file
for n in range(0,len(reflam)):
     wl = reflam[n]
     if (wl > np.amin(lam)) & (wl < np.amax(lam)) & (wl not in deleted):
           window = (lam > wl-0.5*bandwidth) & (lam < wl+0.5*bandwidth)
           f.write("%.1f   %.1f   %.1f\n" %(wl,refflux[n],np.mean(stdcounts[window])))

f.close
