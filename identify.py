from IPython.display import Image

#Run the setup script
exec(open("setup.py").read())

#Structure: We aim for a structure with an outer loop where we go through
#adding lines and fitting until the user wants to stop. Within the outer loop
#there are two steps: 1) adding lines, 2) fitting / deleting lines. The
#script will produce an output file in the database idarc.dat The script
#requires as input a file with reference lines from the arc-lamp used for the
#observation. List with reference wavelengths can typically be downloaded from
#observatory web-pages.

FindContinue = True
     
while FindContinue:
#Reference wavelengths for the arclamp
#The expected format: wavelength and text (ID of the line)
     wavelengthtable = list()
     wavelength = list()
     infile = open('database/mylines_vac.dat', 'r')
     n=1
     for line in infile:
           words = line.split()
           wl = float(words[0])
           id = str()
           for i in range(1,len(words)): 
               id+=words[i]
               id+=' '
           wavelengthtable.append([int(n), float(wl), id])
           wavelength.append(wl)
           n+=1
     wavelengthtable = np.array(wavelengthtable)
     wavelength = np.array(wavelength)

#Read idarc if it exists already
     if os.path.isfile('database/idarc.dat'):
         print ("idarc exists in the database. I will add to the existing file.")
         pix_wl_table = np.loadtxt('database/idarc.dat')
     else:
         print ("idarc does not exist in the database. Will make a new file.")
         pix_wl_table = list()
     
     
     #Print numbered list of reference wavelengths
     #Ideally I would like to print this out in a separate scroll-able window
     print()
     print('Number', 'Wavelength', 'ID')
     print('====================')
     for n in range(0,len(wavelength)): print(n+1,'    ',wavelength[n],'  ',wavelengthtable[n,2])
     #t = QTable([np.array(wavelengthtable[:,0]),np.array(wavelengthtable[:,1]),np.array(wavelengthtable[:,2])],
     #             names=('Number', 'Wavelength [Å]', 'ID'),
     #             meta={'name': 'first table'})
     
     
     #Cut out the region of the arc-file to fit to (defined in setup.py)
     lowercut_ID = N_SPATIAL//2 - NSUM_ID//2
     uppercut_ID = N_SPATIAL//2 + NSUM_ID//2
     identify_1 = np.median(lampimage[lowercut_ID:uppercut_ID, :], axis=0)
     
     # Plot the arclines as a function of pixel number
     max_intens = np.max(identify_1)
     fig = plt.figure()
     ax = fig.add_subplot(111)
     title_str = r'First zoom, then hit any key and select with right-click on the mouse'
     x_identify = np.arange(0, len(identify_1))
     ax.plot(x_identify, identify_1, lw=1)
     ax.grid(ls=':')
     ax.set_xlabel('Pixel number')
     ax.set_ylabel('Pixel value sum')
     ax.set_xlim(0,len(identify_1))
     ax.set_title(title_str.format(MINAMP_PK, MINSEP_PK), fontsize=10)
     
     #If idarc already exists the overplot existing lines
     if os.path.isfile('database/idarc.dat'):
          for pixval in pix_wl_table[:,0]:
               plt.axvline(pixval, ymin=0.90, ymax=0.95, color='r', lw=1.5)
          pix_wl_table = list(pix_wl_table)
     
     print('First zoom, then hit any key to select with right-click on the mouse:')
     # Mark lines in the window and get the x values:
     get_new_line = True
     while get_new_line:
         plt.waitforbuttonpress()
         points = plt.ginput(n=1, timeout=30, show_clicks=True, mouse_add=1, mouse_stop=2)
         if len(points) == 1:
             pix_ref, _ = points[0]
             # Fit Gaussian around +/- 10 pixels:
             mask = np.abs(x_identify - pix_ref) < FWHM_ID
             # Determine initial values:
             A0 = np.max(identify_1[int(pix_ref)])
             #bg0 = np.median([mask])
             bg0 = 0.
             # Use initial sigma of 1 pix:
             sig0 = 1.
             p0 = [pix_ref, sig0, A0, bg0]
             popt, pcov = curve_fit(gaussian, x_identify[mask], identify_1[mask], p0)
             centroid_fit = popt[0]
             plt.axvline(centroid_fit, ymin=0.90, ymax=0.95, color='r', lw=1.5)
             plt.draw()
     
             print("Give the number of the selected calibration lines in the wavelength list:" % centroid_fit)
             num_wl_ref = int(input(" number (-1 to skip): "))
             # Then here you can add a table look-up from a linelist...
             if num_wl_ref != -1: pix_wl_table.append([centroid_fit, wavelength[num_wl_ref-1]])
     
         else:
             answer = input("End input [Y/N]?: ")
             if answer.lower() in ['yes', 'y', '']:
                  get_new_line = False
                  plt.close("all")
     
     plt.show()
     
     #Print list of pixel numbers and wavelengths to file in the database
     pix_wl_table = np.array(pix_wl_table)
     print(" Pixel to wavelength reference table :")
     df = pd.DataFrame(pix_wl_table[pix_wl_table[:,1].argsort()],dtype='float32')
     print(df)
     df.to_csv('database/idarc.dat', header=None, index=None, sep=' ')
     
     #If there are more than ORDER_ID then run the fitarcdata script
     if len(pix_wl_table[:,0]) >= ORDER_ID+2: 
            print('There are more than %s lines in the list so there is enough to fit' % ORDER_ID+2)
            exec(open("fitarcdata.py").read())
     else: print('Too few lines to fit. Add more lines.')

     ask = input('Add more lines (Y/N)?')
     if ask == 'N': FindContinue = False
     
