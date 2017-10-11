import numpy as np
import matplotlib.pyplot as pl
import everest
from everest.math import SavGol
from intrapix import PixelFlux
from random import randint
from astropy.io import fits
import pyfits
from everest import Transit
import k2plr
from k2plr.config import KPLR_ROOT
from everest.config import KEPPRF_DIR
import os


class Target(object):

    def __init__(self, ID, per = 15, dur = 0.5, depth = 0.01, factor = 1800, ftpf = None):

        # initialize variables
        self.ID = ID
        self.per = per
        self.dur = dur
        self.depth = depth
        self.ftpf = ftpf

        # set aperture size (number of pixels to a side)
        self.aps = 19

    def Transit(self):
        '''
        Generates a transit model with the EVEREST Transit() function.
        Takes parameters: period (per), duration (dur), and depth.
        Returns: transit model
        '''

        self.fpix = np.zeros((1000,self.aps,self.aps))
        self.t = np.linspace(0,90,len(self.fpix))
        self.trn = Transit(self.t, t0 = 5.0, per = self.per, dur = self.dur, depth = self.depth)

        return self.trn

    def GeneratePSF(self, mag, motion_mag = 1.0):
        '''
        Generate a PSF model that includes modeled inter-pixel sensitivity variation.
        Model includes photon noise and background nosie.
        Returns: fpix and ferr
        '''

        self.A = self.Amplitude(mag)

        # read in relevant data
        if self.ftpf is None:
            client = k2plr.API()
            star = client.k2_star(self.ID)
            tpf = star.get_target_pixel_files(fetch = True)[0]
            ftpf = os.path.join(KPLR_ROOT, 'data', 'k2', 'target_pixel_files', '%d' % self.ID, tpf._filename)
        else:
            ftpf = self.ftpf
        with pyfits.open(ftpf) as f:
            self.xpos = f[1].data['pos_corr1']
            self.ypos = f[1].data['pos_corr2']

        # throw out outliers
        for i in range(len(self.xpos)):
            if abs(self.xpos[i]) >= 50 or abs(self.ypos[i]) >= 50:
                self.xpos[i] = 0
                self.ypos[i] = 0
            if np.isnan(self.xpos[i]):
                self.xpos[i] = 0
            if np.isnan(self.ypos[i]):
                self.ypos[i] = 0

        # import pdb; pdb.set_trace()
        self.xpos = self.xpos[1000:2000] * motion_mag
        self.ypos = self.ypos[1000:2000] * motion_mag

        # define intra-pixel sensitivity variation
        intra = np.zeros((self.aps,self.aps))
        for i in range(self.aps):
            for j in range(self.aps):
                intra[i][j] = (0.995 + np.random.randint(10) / 1000)

        # mask transits
        self.naninds = np.where(self.trn < 1)
        self.M = lambda x: np.delete(x, self.naninds, axis = 0)

        # generate PRF model with inter-pixel sensitivity variation
        cx = [1.0,0.0,-0.2]
        cy = [1.0,0.0,-0.2]
        x0 = (self.aps / 2.0) + 0.2 * np.random.randn() # + 0.25
        y0 = (self.aps / 2.0) + 0.2 * np.random.randn() # + 0.25
        sx = [0.5 + 0.05 * np.random.randn()]
        sy = [0.5 + 0.05 * np.random.randn()]
        rho = [0.05 + 0.02 * np.random.randn()]
        neighborcoords = [4.,4.]

        magdiff = 1.0
        r = 10**(magdiff / 2.5)

        self.target = np.zeros((len(self.fpix),self.aps,self.aps))
        self.ferr = np.zeros((len(self.fpix),self.aps,self.aps))
        background_level = 800

        is_neighbor = False

        for c in range(100):
            for i in range(self.aps):
                for j in range(self.aps):
                    # contribution from target
                    target_val = self.trn[c]*PixelFlux(cx,cy,[self.A],[x0-i+self.xpos[c]],[y0-j+self.ypos[c]],sx,sy,rho)

                    if is_neighbor:
                        # contribution from neighbor
                        val = target_val + (1/r)*PixelFlux(cx,cy,[self.A],[neighborcoords[0]-i+self.xpos[c]],[neighborcoords[1]-j+self.ypos[c]],sx,sy,rho)
                        self.fpix[c][i][j] = val
                    else:
                        self.fpix[c][i][j] = target_val

                    self.target[c][i][j] = target_val

                    factor = self.NoiseFactor(self.fpix[c][i][j])
                    # add photon noise
                    if self.fpix[c][i][j] < 0:
                        self.fpix[c][i][j] = 0
                    self.ferr[c][i][j] = np.sqrt(self.fpix[c][i][j]) * factor
                    randnum = np.random.randn()
                    self.fpix[c][i][j] += self.ferr[c][i][j] * randnum
                    self.target[c][i][j] += self.ferr[c][i][j] * randnum

                    '''
                    # add background noise
                    noise = np.sqrt(background_level) * np.random.randn()
                    self.fpix[c][i][j] += noise
                    self.target[c][i][j] += noise
                    '''
            # multiply by intra-pixel variation
            self.fpix[c] *= intra
            self.target[c] *= intra

        return self.fpix, self.target, self.ferr


    def CenterOfFlux(self, fpix):
        '''
        Finds the center of flux for the PSF (similar to a center of mass calculation)
        Returns: arrays for x0 and y0 positions over the full light curve
        '''

        ncad, ny, nx = fpix.shape
        x0 = np.zeros(ncad)
        y0 = np.zeros(ncad)
        for n in range(ncad):
            x0[n] = np.sum([(i+0.5)*fpix[n][:,i] for i in range(nx)]) / np.sum(fpix[n])
            y0[n] = np.sum([(ny-j-0.5)*fpix[n][j,:] for j in range(ny)]) / np.sum(fpix[n])

        return x0,y0

    def NoiseFactor(self,counts):
        '''
        Returns the factor by which to multiply the square root of the number
        of counts to get the approximate noise level for a given pixel.
        '''

        x = counts

        a, b, c, d, e = 0.19558, 506.59143, 0.05561, 6131.06708, 0.03664
        return a * np.exp(-x / b) + c * np.exp(-x / d) + e

    def Amplitude(self,mag):
        '''
        Returns the amplitude of the PSF for a star of a given magnitude.
        '''

        x = mag
        a,b,c,d = 1.64864230e+07, .93, -7.35240947e+00, -6.42400461e+03
        return a * np.exp(-b * (x+c)) + d
