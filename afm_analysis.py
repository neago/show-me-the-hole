# coding: utf-8

from __future__ import division
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RectBivariateSpline



class AFM():
    def __init__(self, txtfile=None):
        if txtfile:
            self.load_txtfile(txtfile)
            self.filename = txtfile

    def load_txtfile(self, txtfile):
        try:
            # load data and scale to µm instead of m
            self.Zraw = 1e6 * np.loadtxt(txtfile)
            self.Z = np.copy(self.Zraw)
            self.shape = self.Zraw.shape
            
            # get lateral scan dimensions from text file header comments
            with open(txtfile) as f:
                for line in f:
                    if 'Width' in line: 
                        self.xdim = float(re.findall('[\d,.]+', line)[0]) 
                    if 'Height' in line:
                        self.ydim = float(re.findall('[\d,.]+', line)[0]) 
                    if line[0] != '#':
                        break

            self.dx = self.xdim / (self.shape[0] - 1)
            self.dy = self.ydim / (self.shape[1] - 1)
            #x = np.linspace(0, xdim, zdata.shape[0])
            #y = np.linspace(0, ydim, zdata.shape[1])
            self.X, self.Y = np.mgrid[0:self.xdim:1j*self.shape[0], 
                                      0:self.ydim:1j*self.shape[1]]
            
        except IOError:
            print(txtfile + ' does not exist')

    def interpolate_profile(self, s, e, method='rbs', *args, **kwargs):
        Npoints = np.sqrt((e[0] - s[0])**2 + (e[1] - s[1])**2) + 1
        sx, sy = s[0]/self.shape[0] * self.xdim, s[1]/self.shape[1] * self.ydim
        ex, ey = e[0]/self.shape[0] * self.xdim, e[1]/self.shape[1] * self.ydim
        points = np.c_[np.linspace(sx, ex, Npoints), 
                       np.linspace(sy, ey, Npoints)]
        
        if method == 'gd':
            return points, griddata(c_[self.X.ravel(), self.Y.ravel()], 
                                    self.Z.ravel(), 
                                    (points[:,0], points[:,1]), *args, **kwargs)
        elif method == 'rbs':
            spline = RectBivariateSpline(self.X[:,0], self.Y[0], self.Z, 
                                         *args, **kwargs)
            return points, spline.ev(points[:,0], points[:,1])

    def subtract_plane(self, margins=0):
        """Subtract a plane fit through the data. Store result
        in Z attribute.
        
        Parameters
        ----------
        margins : float or list of 4 floats
            Determines the amount of margin to exclude from the
            edges of the scan when fitting the plane.
            Between 0 (include all) and 0.5 (exclude all).
        """

        Xc, Yc, Zc = map(np.ravel, self.crop_rect_ratio(margins))
        
        # Use SVD for plane fitting <http://stackoverflow.com/a/10904220/2927184>
        G = np.c_[Xc, Yc, Zc, np.ones(Zc.shape)]
        u, s, v = np.linalg.svd(G, False)
        A, B, C, D = v[-1]
        self.Zplane = -1/C * (A * self.X + B * self.Y + D)
        self.Z -= self.Zplane
        
    def offset_bottom(self, margins=0):
        """Offset the data to put the bottom at 0.
        
        Parameters
        ----------
        margins : float or list of 4 floats
            Determines the amount of margin to exclude from the
            edges of the scan when finding minimum.
            Useful for excluding corners.
            Between 0 (include all) and 0.5 (exclude all).
        """
        
        Xc, Yc, Zc = self.crop_rect_ratio(margins)
        
        self.Z -= Zc.min()
        
    def crop_rect_ratio(self, margins=0):
        """Crop X,Y,Z arrays to a rectangular region.
        
        Parameters
        ----------
        margins : float or list of 4 floats
            Amount of margin to crop away, ratio of image size.
            [left, right, bottom, top] or a single float if all
            are equal.
        
        Returns
        -------
        (Xc, Yc, Zc)
        
        Examples
        --------
        >>> s.crop_rect_ratio([.1, .1, .2, .2])
        Crops away 10% of the left and right edges and 20% of the
        top and bottom.
        """
        if np.isscalar(margins):
            margins = [margins, margins, margins, margins]
        else:
            if len(margins) != 4:
                print('margins should be float or list of 4 floats')
                return None            
        
        def crop(M):
            x1 = round(margins[0] * self.shape[0])
            x2 = round((1 - margins[1]) * self.shape[0])
            y1 = round(margins[2] * self.shape[1])
            y2 = round((1 - margins[3]) * self.shape[1])
            
            return M[x1:x2, y1:y2]

        # cropmask = ((self.X > self.xdim * margins[0]) & 
        #             (self.X < self.xdim * (1-margins[1])) &
        #             (self.Y > self.ydim * margins[2]) &
        #             (self.Y < self.ydim * (1-margins[3])))
        
        # return self.X[cropmask], self.Y[cropmask], self.Z[cropmask]
        return (crop(self.X), crop(self.Y), crop(self.Z))

    def show(self, attr='Z', vrange=[None, None], profiles=[]):
        vmin = vrange[0] if vrange[0] is not None else getattr(self, attr).min()
        vmax = vrange[1] if vrange[1] is not None else getattr(self, attr).max()
        plt.imshow(getattr(self, attr).T, cmap=plt.cm.GnBu_r, 
                   interpolation='nearest', origin='lower',
                   extent=[0, self.xdim, 0, self.ydim], vmin=vmin, vmax=vmax)
        plt.colorbar()


class AFMhole(AFM):
    def __init__(self, txtfile):
        AFM.__init__(self, txtfile)
        self.set_minpos()

    def set_minpos(self, margins=0):
        self.minval = self.crop_rect_ratio(margins)[2].min()
        self.minpos = np.array(np.where(self.Z==self.minval)).ravel()
        
    def profiles_around_min(self, angle=0, span=5, margins=0):
        """
        """
        profile_centers = self.minpos + [[i*np.sin(angle), -i*np.cos(angle)] 
                                         for i in range(-span, span+1)]

        # need to figure out how to remove div-by-zero error here
        # radius = min(self.minpos[0] / np.cos(angle),
        #              self.minpos[1] / np.sin(angle),
        #              (self.shape[0] - self.minpos[0]) / np.cos(angle),
        #              (self.shape[1] - self.minpos[1]) / np.sin(angle))
        radius = min(self.minpos[0], self.shape[0] - self.minpos[0],
                     self.minpos[1], self.shape[1] - self.minpos[1])
        radius -= max(span * np.cos(angle), span * np.sin(angle))
        ss = profile_centers - [np.cos(angle) * radius, np.sin(angle) * radius]
        ee = profile_centers + [np.cos(angle) * radius, np.sin(angle) * radius]

        profiles = [self.interpolate_profile(ss[i], ee[i]) 
                    for i in range(len(ss))]

        ddiag = np.sqrt((np.cos(angle) * self.dx)**2 + 
                        (np.sin(angle) * self.dy)**2)
        xvals = np.linspace(-radius * ddiag, radius * ddiag, len(profiles[0][1]))
        # halflength = min(self.shape)/2
        # co = [c + np.arange(-halflength, halflength+1)[:,np.newaxis]
        #       * [np.cos(angle), np.sin(angle)] for c in profile_centers]
        # co = np.array(co)
        # inside = np.all((co[:,:,0] >= 0) & (co[:,:,0] <= self.shape[0]-1) &
        #                 (co[:,:,1] >= 0) & (co[:,:,1] <= self.shape[1]-1), 0)
        # co = co[:, inside]

        return [Profile(xvals, p[1], p[0]) for p in profiles]

    def profile_analysis(self, margins=.2, degree=10, angles=36):
        self.set_minpos(margins)
        self.angles = np.linspace(0, np.pi, angles+1, endpoint=True)
        self.profs = [self.profiles_around_min(a, 0)[0] for a in self.angles]
        for p in self.profs:
            p.fit_poly(degree)

        self.ROCmins = [p.ROCmin for p in self.profs]
        self.diameters = [p.diameter for p in self.profs]
        self.major = np.argmax(self.ROCmins)
        self.minor = np.argmin(self.ROCmins)
        # first and last profile are the same, so skip one for averages
        self.ROCmean = np.mean(self.ROCmins[:-1])
        self.ROCaspect = max(self.ROCmins) / min(self.ROCmins)
        self.diam_mean = np.mean(self.diameters[:-1])
        self.diam_aspect = max(self.diameters) / min(self.diameters)
        self.depth = self.Z.max() - self.minval
        
        #print(ROCmean, ROCaspect, diam_mean, diam_aspect, depth)

    def print_data(self):
        print(self.filename)
        print('-' * len(self.filename))
        print('ROC: {:.1f} µm - {:.1f} µm; mean {:.1f}'.format(min(self.ROCmins),
                                                            max(self.ROCmins),
                                                            self.ROCmean))
        print('diameter: {:.1f} µm - {:.1f} µm; mean {:.1f}'.format(
                min(self.diameters), max(self.diameters), self.diam_mean))
        print('aspect ratio: {:.2f} (diam) / {:.2f} (ROC)'.format(
                self.diam_aspect, self.ROCaspect))
        print('depth: {:.2f} µm'.format(self.depth))


class Profile():
    def __init__(self, x, y, coords):
        self.x = x
        self.y = y
        self.coords = coords

    def fit_poly(self, degree=10):
        # fit a degree-order polynomial to the profile
        self._poly = np.polyfit(self.x, self.y, degree)
        self.poly = np.polyval(self._poly, self.x)
        # calculate first and second derivatives
        self._poly_d = np.arange(degree, 0, -1) * self._poly[:-1]
        self._poly_d2 = np.arange(degree-1, 0, -1) * self._poly_d[:-1]
        self.ROC = ((1  + np.polyval(self._poly_d, self.x)**2)**(3/2) 
                    / np.polyval(self._poly_d2, self.x))

        # calculate diameter of hole as distance between turning points
        # (where ROC goes from positive to negative)
        neg_curv = np.where(self.ROC < 0)[0]
        ip = np.searchsorted(neg_curv, len(self.y)/2)
        self.inflections = [neg_curv[ip-1], neg_curv[ip]]
        self.diameter = (self.x[self.inflections[1] - 1] - 
                         self.x[self.inflections[0]])

        self.ROCmin = self.ROC[self.inflections[0]+1:self.inflections[1]].min()

        # very simple depth measure
        self.depth = self.y.max() - self.y.min()