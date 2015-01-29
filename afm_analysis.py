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
            return griddata(c_[self.X.ravel(), self.Y.ravel()], self.Z.ravel(), 
                            (points[:,0], points[:,1]), *args, **kwargs)
        elif method == 'rbs':
            spline = RectBivariateSpline(self.X[:,0], self.Y[0], self.Z, 
                                         *args, **kwargs)
            return spline.ev(points[:,0], points[:,1])

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
        return map(crop, (self.X, self.Y, self.Z))


class AFMhole(AFM):
    def find_minimum(self):
        pass
        
    def profiles_around_min(self, angle=0, span=5, margins=0):
        """
        """
        minx, miny = map(int, where(d.Z==d.crop_rect_ratio(margins)[2].min()))
        

