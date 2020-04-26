"""
Graphic Interface for spectral line identification
"""

__author__ = "Jens-Kristian Krogager"
__email__ = "krogager.jk@gmail.com"
__credits__ = ["Jens-Kristian Krogager", "Johan Fynbo"]
__version__ = '1.0'

#Run the setup script
exec(open("setup.py").read())


def NN_gaussian(x, bg, mu, sigma, logamp):
    """ One-dimensional modified non-negative Gaussian profile."""
    amp = 10**logamp
    return bg + amp * np.exp(-0.5*(x-mu)**4/sigma**2)

def fit_gaussian_center(x, y):
    bg = np.median(y)
    logamp = np.log10(np.nanmax(y))
    sig = 1.5
    mu = x[len(x)//2]
    p0 = np.array([bg, mu, sig, logamp])
    popt, _ = curve_fit(NN_gaussian, x, y, p0)
    return popt[1]


class ResidualView(object):
    def __init__(self, axis, mean=0., scatter=0., visible=False):
        self.zeroline = axis.axhline(0., color='0.1', ls=':', alpha=0.8)
        self.med_line = axis.axhline(mean, color='RoyalBlue', ls='--')
        self.u68_line = axis.axhline(mean + scatter, color='crimson', ls=':')
        self.l68_line = axis.axhline(mean - scatter, color='crimson', ls=':')
        self.mean = mean
        self.scatter = scatter

        if visible:
            self.set_visible(True)
        else:
            self.set_visible(False)

    def get_lines(self):
        return [self.zeroline, self.med_line, self.u68_line, self.l68_line]

    def set_visible(self, visible=True):
        for line in self.get_lines():
            line.set_visible(visible)

    def set_scatter(self, sig):
        self.u68_line.set_ydata(self.mean + sig)
        self.l68_line.set_ydata(self.mean - sig)
        self.scatter = scatter

    def set_mean(self, mean):
        self.u68_line.set_ydata(mean + self.scatter)
        self.l68_line.set_ydata(mean - self.scatter)
        self.mean = mean

    def set_values(self, mean, scatter):
        self.mean = mean
        self.scatter = scatter
        self.u68_line.set_ydata(mean + scatter)
        self.l68_line.set_ydata(mean - scatter)


def load_linelist(fname):
    with open(fname) as raw:
        all_lines = raw.readlines()

    linelist = list()
    for line in all_lines:
        line = line.strip()
        if line[0] == '#':
            continue

        l_ref = line.split()[0]
        comment = line.replace(l_ref, '').strip()
        linelist.append([float(l_ref), comment])

    sorted_list = sorted(linelist, key=lambda x: x[0])
    return sorted_list


class GraphicInterface(QMainWindow):
    def __init__(self, arc_fname='', grism_name='', pixtable='', linelist_fname='', dispaxis=2, parent=None):
        QMainWindow.__init__(self, parent)
        self.setWindowTitle('Identify Arc Lines')
        self._main = QWidget()
        self.setCentralWidget(self._main)
        self.pix = np.array([])
        self.arc1d = np.array([])
        self.arc_fname = arc_fname
        self.grism_name = grism_name
        self.dispaxis = dispaxis
        self._fit_ref = None
        self.cheb_fit = None
        self._scatter = None
        self.vlines = list()
        self.pixel_list = list()
        self.linelist = np.array([])
        self._full_linelist = [['', '']]

        # Create Toolbar and Menubar
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar_fontsize = QFont()
        toolbar_fontsize.setPointSize(14)

        quit_action = QAction("Close", self)
        quit_action.setFont(toolbar_fontsize)
        quit_action.setStatusTip("Quit the application")
        quit_action.triggered.connect(self.close)
        quit_action.setShortcut("ctrl+Q")
        toolbar.addAction(quit_action)
        toolbar.addSeparator()

        load_file_action = QAction("Load Spectrum", self)
        load_file_action.triggered.connect(self.load_spectrum)
        load_file_action.setShortcut("ctrl+N")
        toolbar.addAction(load_file_action)

        save_pixtab_action = QAction("Save PixTable", self)
        save_pixtab_action.setShortcut("ctrl+S")
        save_pixtab_action.triggered.connect(self.save_pixtable)
        toolbar.addAction(save_pixtab_action)

        load_ref_action = QAction("Load LineList", self)
        load_ref_action.triggered.connect(self.load_linelist_fname)
        toolbar.addAction(load_ref_action)

        load_pixtab_action = QAction("Load PixTable", self)
        load_pixtab_action.triggered.connect(self.load_pixtable)
        # toolbar.addAction(load_pixtab_action)

        toolbar.addSeparator()

        add_action = QAction("Add Line (a)", self)
        add_action.setShortcut("ctrl+A")
        add_action.setStatusTip("Identify new line")
        add_action.setFont(toolbar_fontsize)
        add_action.triggered.connect(self.add_line)
        toolbar.addAction(add_action)

        del_action = QAction("Delete Line (d)", self)
        del_action.setShortcut("ctrl+D")
        del_action.setStatusTip("Delete line")
        del_action.setFont(toolbar_fontsize)
        del_action.triggered.connect(self.remove_line)
        toolbar.addAction(del_action)

        clear_action = QAction("Clear Lines", self)
        clear_action.setStatusTip("Clear all identified lines")
        clear_action.setFont(toolbar_fontsize)
        clear_action.triggered.connect(self.clear_lines)
        toolbar.addAction(clear_action)

        refit_action = QAction("Refit Line (r)", self)
        refit_action.setShortcut("ctrl+R")
        refit_action.setFont(toolbar_fontsize)
        refit_action.triggered.connect(self.refit_line)
        toolbar.addAction(refit_action)

        refit_all_action = QAction("Refit All", self)
        refit_all_action.setFont(toolbar_fontsize)
        refit_all_action.triggered.connect(self.refit_all)
        toolbar.addAction(refit_all_action)
        self.addToolBar(toolbar)

        main_menu = self.menuBar()
        file_menu = main_menu.addMenu("File")
        file_menu.addAction(load_file_action)
        file_menu.addAction(load_ref_action)
        file_menu.addSeparator()
        file_menu.addAction(save_pixtab_action)
        file_menu.addAction(load_pixtab_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        edit_menu = main_menu.addMenu("Edit")
        edit_menu.addAction(add_action)
        edit_menu.addAction(del_action)
        edit_menu.addAction(clear_action)
        edit_menu.addSeparator()
        edit_menu.addAction(refit_action)
        edit_menu.addAction(refit_all_action)
        edit_menu.addSeparator()
        fit_action = QAction("Fit wavelength solution", self)
        fit_action.triggered.connect(self.fit)
        clear_fit_action = QAction("Clear wavelength solution", self)
        clear_fit_action.triggered.connect(self.clear_fit)
        save_fit_action = QAction("Save polynomial coefficients", self)
        save_fit_action.triggered.connect(self.save_wave)
        edit_menu.addAction(fit_action)
        edit_menu.addAction(clear_fit_action)
        edit_menu.addAction(save_fit_action)


        # =============================================================
        # Start Layout:
        layout = QHBoxLayout(self._main)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create Table for Reference Linelist:
        ref_layout = QVBoxLayout()
        label_ref_header = QLabel("Reference Linelist\n")
        label_ref_header.setAlignment(Qt.AlignCenter)
        self.reftable = QTableWidget()
        self.reftable.setColumnCount(2)
        self.reftable.setHorizontalHeaderLabels(["Wavelength", "Ion"])
        self.reftable.setColumnWidth(0, 80)
        self.reftable.setColumnWidth(1, 60)
        self.reftable.setFixedWidth(180)
        ref_layout.addWidget(label_ref_header)
        ref_layout.addWidget(self.reftable)
        layout.addLayout(ref_layout)

        # Create Table for Pixel Identifications:
        pixtab_layout = QVBoxLayout()
        label_pixtab_header = QLabel("Pixel Table\n")
        label_pixtab_header.setAlignment(Qt.AlignCenter)
        self.linetable = QTableWidget()
        self.linetable.verticalHeader().hide()
        self.linetable.setColumnCount(2)
        self.linetable.setHorizontalHeaderLabels(["Pixel", "Wavelength"])
        self.linetable.setColumnWidth(0, 80)
        self.linetable.setColumnWidth(1, 90)
        self.linetable.setFixedWidth(180)
        pixtab_layout.addWidget(label_pixtab_header)
        pixtab_layout.addWidget(self.linetable)
        layout.addLayout(pixtab_layout)



        # Create Right-hand Side Layout: (buttons, figure, buttons and options for fitting)
        right_layout = QVBoxLayout()

        # Top row of options and bottons:
        bottom_hbox = QHBoxLayout()
        button_fit = QPushButton("Fit")
        button_fit.clicked.connect(self.fit)
        button_clear_fit = QPushButton("Clear fit")
        button_clear_fit.clicked.connect(self.clear_fit)
        button_show_resid = QPushButton("Resid/Data")
        button_show_resid.clicked.connect(self.toggle_residview)
        self.poly_order = QLineEdit(str(ORDER_ID))
        self.poly_order.setValidator(QIntValidator(1, 100))
        self.poly_order.returnPressed.connect(self.fit)
        button_save_wave = QPushButton("Save Fit")
        button_save_wave.clicked.connect(self.save_wave)
        bottom_hbox.addWidget(button_fit)
        bottom_hbox.addWidget(QLabel('Poly Order: '))
        bottom_hbox.addWidget(self.poly_order)
        bottom_hbox.addStretch(1)
        bottom_hbox.addWidget(button_show_resid)
        bottom_hbox.addStretch(1)
        bottom_hbox.addWidget(button_save_wave)
        bottom_hbox.addWidget(button_clear_fit)

        right_layout.addLayout(bottom_hbox)

        # Figure in the middle:
        self.fig = Figure(figsize=(6, 8))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        # self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.canvas.setFocus()
        right_layout.addWidget(self.canvas, 1)
        self.mpl_toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.mpl_toolbar)

        layout.addLayout(right_layout, 1)


        # -- Draw initial data:
        self.ax = self.fig.add_axes([0.15, 0.40, 0.82, 0.54])
        self.ax.set_ylabel("Intensity")
        self.ax.set_title("Add line by pressing 'a'")
        self.ax.plot(self.pix, self.arc1d)
        self.ax.set_xlim(0, 2048)
        self.ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        self.ax2 = self.fig.add_axes([0.15, 0.10, 0.82, 0.25])
        self.ax2.plot([], [], 'k+')
        self.ax2.set_xlim(0, 2048)
        self.ax2.set_ylim(0, 1)
        self.ax2.set_xlabel("Pixel Coordinate")
        self.ax2.set_ylabel("Ref. Wavelength")
        self.resid_view = ResidualView(self.ax2)
        self._fit_view = 'data'

        if os.path.exists(arc_fname):
            self.load_spectrum(arc_fname)

        if os.path.exists(pixtable):
            self.load_pixtable(pixtable)

        if os.path.exists(linelist_fname):
            self.load_linelist_fname(linelist_fname)


    def load_linelist_fname(self, linelist_fname=None):
        if linelist_fname is False:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            filters = "All files (*)"
            linelist_fname = QFileDialog.getOpenFileName(self, 'Open Linelist', current_dir, filters)
            linelist_fname = str(linelist_fname[0])

        if linelist_fname:
            self.linelist = np.loadtxt(linelist_fname, usecols=(0,))
            self._full_linelist = load_linelist(linelist_fname)
            self.set_reftable_data()

    def set_reftable_data(self):
        for line in self._full_linelist:
            rowPosition = self.reftable.rowCount()
            self.reftable.insertRow(rowPosition)
            wl_ref, comment = line
            item = QTableWidgetItem("%.2f" % wl_ref)
            item.setFlags(Qt.ItemIsEnabled)
            item.setBackground(QColor('lightgray'))
            self.reftable.setItem(rowPosition, 0, item)

            item2 = QTableWidgetItem(comment)
            item2.setFlags(Qt.ItemIsEnabled)
            item2.setBackground(QColor('lightgray'))
            self.reftable.setItem(rowPosition, 1, item2)

    def load_spectrum(self, arc_fname=None):
        if arc_fname is False:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            filters = "FITS files (*.fits | *.fit)"
            arc_fname = QFileDialog.getOpenFileName(self, 'Open Pixeltable', current_dir, filters)
            arc_fname = str(arc_fname[0])

        if arc_fname:
            self.arc_fname = arc_fname
            hdu = fits.open(arc_fname)
            primhdr = hdu[0].header
            raw_data = fits.getdata(arc_fname)
            if len(hdu) > 1:
                imghdr = hdu[1].header
                primhdr.update(imghdr)
            hdu.close()
            if 'DISPAXIS' in primhdr.keys():
                self.dispaxis = primhdr['DISPAXIS']
            if self.dispaxis == 1:
                raw_data = raw_data.T
            ilow = raw_data.shape[1]//2 - 20
            ihigh = raw_data.shape[1]//2 + 20
            self.arc1d = np.median(raw_data[:, ilow:ihigh], axis=1)
            self.pix = np.arange(raw_data.shape[0])

            self.ax.lines[0].set_data(self.pix, self.arc1d)
            self.ax.relim()
            self.ax.autoscale()
            self.ax.set_xlim(np.min(self.pix), np.max(self.pix))
            self.ax2.set_xlim(np.min(self.pix), np.max(self.pix))
            self.canvas.draw()

    def load_pixtable(self, filename=None):
        if filename is False:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            filters = "All files (*)"
            filename = QFileDialog.getOpenFileName(self, 'Open Pixeltable', current_dir, filters)
            filename = str(filename[0])

        if filename:
            self._main.setUpdatesEnabled(False)
            pixtable = np.loadtxt(filename)
            for x, wl in pixtable:
                self.append_table(x, wl)
                vline = self.ax.axvline(x, color='r', ls=':', lw=1.0)
                self.vlines.append(vline)
                self.pixel_list.append(x)
            self.update_plot()
            self._main.setUpdatesEnabled(True)
            self.canvas.setFocus()

    def save_pixtable(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filters = "All files (*)"
        path = QFileDialog.getSaveFileName(self, 'Save Pixeltable', current_dir, filters)
        if path[0]:
            with open(str(path[0]), 'w') as tab_file:
                pixvals, wavelengths = self.get_table_values()
                tab_file.write("# Pixel Table: %s\n" % self.grism_name)
                tab_file.write("# Pixel    Wavelength [Å]\n")
                np.savetxt(tab_file, np.column_stack([pixvals, wavelengths]),
                           fmt=" %8.2f   %8.2f")

    def save_wave(self):
        if self.cheb_fit is None:
            return
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = QFileDialog.getSaveFileName(self, 'Save Polynomial Model', current_dir)
        poly_fit = self.cheb_fit.convert(kind=np.polynomial.Polynomial)
        if path[0]:
            with open(str(path[0]), 'w') as output:
                output.write("# Wavelength solution for grism: %s\n" % self.grism_name)
                output.write("# Raw arc-frame filename: %s\n" % self.arc_fname)
                output.write("# Wavelength residual = %.2f Å\n" % self._scatter)
                output.write("# Polynomial coefficients:\n")
                poly_fit = self.cheb_fit.convert(kind=np.polynomial.Polynomial)
                for i, c_i in enumerate(poly_fit.coef):
                    output.write(" % .3e\n" % c_i)

    def remove_line(self):
        self.ax.set_title("Click near the line you want to remove...")
        self.canvas.draw()
        point = self.fig.ginput(1, 0)
        if len(point) == 0:
            self.ax.set_title("")
            self.canvas.draw()
            return
        x0, _ = point[0]
        idx = np.argmin(np.abs(self.pixel_list - x0))
        self.vlines[idx].remove()
        self.vlines.pop(idx)
        self.pixel_list.pop(idx)
        self.linetable.removeRow(idx)
        self.ax.set_title("")
        self.update_plot()

    def append_table(self, x0, wl0=None):
        rowPosition = self.linetable.rowCount()
        self.linetable.insertRow(rowPosition)
        item = QTableWidgetItem("%.2f" % x0)
        item.setFlags(Qt.ItemIsEnabled)
        item.setBackground(QColor('lightgray'))
        self.linetable.setItem(rowPosition, 0, item)

        wl_item = QLineEdit("")
        wl_item.setValidator(QRegExpValidator(QRegExp(r"^\s*\d*\s*\.?\d*$")))
        # wl_item.setValidator(QDoubleValidator())
        if wl0 is not None:
            wl_item.setText("%.2f" % wl0)
        wl_item.returnPressed.connect(self.look_up)
        # wl_item.editingFinished.connect(self.look_up)
        self.linetable.setCellWidget(rowPosition, 1, wl_item)

    def add_line(self):
        self.ax.set_title("Click to fit Gaussian centroid...")
        self.canvas.draw()
        point = self.fig.ginput(1, 0)
        if len(point) == 0:
            self.ax.set_title("")
            self.canvas.draw()
            return
        x0, _ = point[0]
        # -- Fit Gaussian
        cut = np.abs(self.pix - x0) <= 10
        pix_cut = self.pix[cut]
        arc_cut = self.arc1d[cut]
        x_cen = fit_gaussian_center(pix_cut, arc_cut)

        # Update Table:
        if self.cheb_fit is None:
            rowPosition = self.linetable.rowCount()
            self.append_table(x0)
            self.linetable.cellWidget(rowPosition, 1).setFocus()
        else:
            wl_predicted = self.cheb_fit(x0)
            line_idx = np.argmin(np.abs(self.linelist - wl_predicted))
            wl_sep = self.linelist[line_idx] - wl_predicted
            if wl_sep > 5.:
                msg = "No line in linelist matches the expected wavelength!"
                QMessageBox.critical(None, 'No line found', msg)
            else:
                wl_predicted = self.linelist[line_idx]
            self.append_table(x0, wl_predicted)
            self.update_plot()

        # Update Plot:
        vline = self.ax.axvline(x_cen, color='r', ls=':', lw=1.0)
        self.vlines.append(vline)
        self.pixel_list.append(x_cen)
        self.ax.set_title("")
        self.canvas.draw()

    def refit_line(self, default=False, idx=None):
        if idx is None:
            self.ax.set_title("Click near a line you want to refit...")
            self.canvas.draw()
            point = self.fig.ginput(1, 0)
            if len(point) == 0:
                self.ax.set_title("")
                self.canvas.draw()
                return
            x0, _ = point[0]
            idx = np.argmin(np.abs(self.pixel_list - x0))
            self.vlines[idx].set_linestyle('-')
            self.ax.set_title("Now click near the new centroid...")
            self.canvas.draw()
            point2 = self.fig.ginput(1, 0)
            x_old, _ = point2[0]
            cut = np.abs(self.pix - x_old) <= 7
            self.vlines[idx].set_linestyle(':')
            self.ax.set_title("")
            self.canvas.draw()
        else:
            old_item = self.linetable.item(idx, 0)
            x_old = float(old_item.text())
            cut1 = np.abs(self.pix - x_old) <= 25
            idx_max = np.argmax(self.arc1d * cut1)
            cut = np.abs(self.pix - self.pix[idx_max]) <= 10
        pix_cut = self.pix[cut]
        arc_cut = self.arc1d[cut]
        try:
            x_cen = fit_gaussian_center(pix_cut, arc_cut)
        except RuntimeError:
            x_cen = x_old

        item = QTableWidgetItem("%.2f" % x_cen)
        item.setFlags(Qt.ItemIsEnabled)
        item.setBackground(QColor('lightgray'))
        self.linetable.setItem(idx, 0, item)
        self.pixel_list[idx] = x_cen
        self.vlines[idx].set_xdata(x_cen)
        self.update_plot()

    def refit_all(self):
        n_rows = self.linetable.rowCount()
        if n_rows == 0:
            return

        for idx in range(n_rows):
            self.refit_line(idx=idx)
        self.update_plot()

    def look_up(self):
        wl_editor = self.focusWidget()
        if wl_editor.text().strip() == '':
            pass
        else:
            wl_in = float(wl_editor.text())
            line_idx = np.argmin(np.abs(self.linelist - wl_in))
            wl_sep = self.linelist[line_idx] - wl_in
            if wl_sep > 5.:
                msg = "No line in linelist within ±5 Å, using raw input!"
                QMessageBox.critical(None, 'No line found', msg)
                wl_editor.setText("%.2f" % wl_in)
            else:
                wl_editor.setText("%.2f" % self.linelist[line_idx])
        wl_editor.clearFocus()
        self.canvas.setFocus()
        self.update_plot()

    def on_key_press(self, event):
        if event.key == 'a':
            self.add_line()
        elif event.key == 'd':
            self.remove_line()
        elif event.key == 'r':
            self.refit_line()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_A:
            self.add_line()
        elif e.key() == Qt.Key_D:
            self.remove_line()
        elif e.key() == Qt.Key_R:
            self.refit_line()

    def get_table_values(self):
        pixvals = list()
        wavelengths = list()
        n_rows = self.linetable.rowCount()

        for n in range(n_rows):
            x = self.linetable.item(n, 0)
            wl = self.linetable.cellWidget(n, 1)
            pixvals.append(float(x.text()))

            if wl:
                wl_text = wl.text()
                # tmp = wl_text.replace('.', '')
                if wl_text == '':
                    wavelengths.append(np.nan)
                else:
                    wavelengths.append(float(wl_text))
            else:
                wavelengths.append(np.nan)
        wavelengths = np.array(wavelengths)
        pixvals = np.array(pixvals)
        return (pixvals, wavelengths)

    def set_dataview(self):
        self.ax2.set_ylabel("Ref. Wavelength")
        self._fit_view = 'data'
        self.resid_view.set_visible(False)
        pixvals, wavelengths = self.get_table_values()
        self.ax2.lines[0].set_data(pixvals, wavelengths)
        self.set_ylimits()
        self.fig.canvas.draw()
        self.canvas.setFocus()

    def set_residview(self):
        pixvals, wavelengths = self.get_table_values()
        if self.cheb_fit is not None:
            self.ax2.set_ylabel("Residual Wavelength")
            self._fit_view = 'resid'
            self.resid_view.set_visible(True)
            # Set data to residuals:
            residuals = wavelengths - self.cheb_fit(pixvals)
            mean = np.nanmean(residuals)
            scatter = np.nanstd(residuals)
            self.resid_view.set_values(mean, scatter)
            self.ax2.lines[0].set_ydata(residuals)
            self.set_ylimits()
            self.fig.canvas.draw()
            self.canvas.setFocus()
        else:
            msg = "Cannot calculate residuals. Data have not been fitted yet."
            QMessageBox.critical(None, "Cannot show residuals", msg)

    def set_ylimits(self):
        yvals = self.ax2.lines[0].get_ydata()
        if np.sum(~np.isnan(yvals)) > 0:
            ymin = np.nanmin(yvals)
            ymax = np.nanmax(yvals)
            delta_wl = ymax - ymin
            if self._fit_view == 'data':
                if delta_wl == 0.:
                    self.ax2.set_ylim(ymax*0.5, ymax*1.5)
                else:
                    self.ax2.set_ylim(max(0., ymin-delta_wl*0.2), ymax+delta_wl*0.2)
            else:
                self.ax2.set_ylim(-3*self._scatter, 3*self._scatter)

    def toggle_residview(self):
        if self._fit_view == 'data':
            self._fit_view = 'resid'
            self.set_residview()
        elif self._fit_view == 'resid':
            self.set_dataview()
            self._fit_view = 'data'
        else:
            print(" [ERROR] - Unknown value of _fit_view: %r" % self._fit_view)
            print("  How did that happen??!!")
            self.set_dataview()
            self._fit_view = 'data'

    def update_plot(self):
        self.set_dataview()

    def fit(self):
        pixvals, wavelengths = self.get_table_values()
        mask = ~np.isnan(wavelengths)
        order = int(self.poly_order.text())
        if np.sum(~np.isnan(wavelengths)) < order:
            msg = "Not enough data points to perform fit!\n"
            msg += "Choose a lower polynomial order or identify more lines."
            QMessageBox.critical(None, 'Not enough data to fit', msg)
        else:
            p_fit = Chebyshev.fit(pixvals[mask], wavelengths[mask], order, domain=[self.pix.min(), self.pix.max()])
            wave_solution = p_fit(self.pix)
            scatter = np.std(wavelengths[mask] - p_fit(pixvals[mask]))
            scatter_label = r"$\sigma_{\lambda} = %.2f$ Å" % scatter
            self.cheb_fit = p_fit
            self._scatter = scatter
            self._residuals = wavelengths - p_fit(pixvals)
            if self._fit_ref is None:
                fit_ref = self.ax2.plot(self.pix, wave_solution,
                                        color='RoyalBlue', label=scatter_label)
                self._fit_ref = fit_ref[0]
            else:
                self._fit_ref.set_ydata(wave_solution)
                self._fit_ref.set_label(scatter_label)
            self.print_fit()
            self.ax2.legend(handlelength=0.5, frameon=False)
            self.canvas.draw()
            self.canvas.setFocus()

    def print_fit(self):
        if self.cheb_fit is None:
            return

        order = int(self.poly_order.text())
        print("\n\n------------------------------------------------")
        print(" Fitting wavelength solution")
        print(" Using polynomium of order: %i" % order)
        print("")
        print(" Wavelength residual = %.2f Å" % self._scatter)
        print(" Polynomial coefficients:")
        poly_fit = self.cheb_fit.convert(kind=np.polynomial.Polynomial)
        for i, c_i in enumerate(poly_fit.coef):
            print(" c_%i = % .3e" % (i, c_i))
        print("\n\n------------------------------------------------")

    def clear_fit(self):
        if self._fit_ref is not None:
            self._scatter = None
            self._residuals = None
            self.cheb_fit = None
            self.ax2.get_legend().remove()
            self._fit_ref.remove()
            self._fit_ref = None
            self.canvas.draw()
            self.canvas.setFocus()
            self.set_dataview()
            self._fit_view = 'data'

    def clear_lines(self):
        n_rows = self.linetable.rowCount()
        for idx in range(n_rows)[::-1]:
            self.vlines[idx].remove()
            self.vlines.pop(idx)
            self.pixel_list.pop(idx)
            self.linetable.removeRow(idx)
            self.clear_fit()
        self.set_dataview()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Spectral Line Identification')
    parser.add_argument("filename", type=str, nargs='?', default='',
                        help="Raw arc-line spectrum")
    parser.add_argument("--lines", type=str, default='',
                        help="Linelist filename containing reference wavelengths")
    parser.add_argument("--axis", type=int, default=2,
                        help="Dispersion axis 1: horizontal, 2: vertical  [default=2]")
    args = parser.parse_args()

    #if args.filename == 'test':
    #    # Load Test Data:
    #    tab_fname = 'calib/grism4_pixeltable.dat'
    #    fname = '/Users/krogager/Data/NOT/MALS/ALzh010234.fits'
    #    grism_name = 'grism4'
    #    linelist_fname = 'calib/HeNe_linelist.dat'
    #    dispaxis = 2
    #else:
    #    fname = args.filename
    #    linelist_fname = args.lines
    #    tab_fname = ''
    #    grism_name = ''
    #    dispaxis = args.axis

    #If the pixel file exist then read it.
    fname = COMPIMAGE
    grism_name = ''
    linelist_fname = ''
    dispaxis = 1
    if os.path.exists('database/idarc.dat'): tab_fname='database/idarc.dat'

    # Launch App:
    qapp = QApplication(sys.argv)
    app = GraphicInterface(fname, grism_name=grism_name, pixtable=tab_fname, linelist_fname=linelist_fname, dispaxis=dispaxis)
    app.show()
    qapp.exec_()
