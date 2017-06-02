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

    def __init__(self, ID, amplitude, per = 15, dur = 0.5, depth = 0.01):

        # initialize variables
        self.A = amplitude
        self.ID = ID
        self.per = per
        self.dur = dur
        self.depth = depth

    def Transit(self):
        '''
        Generates a transit model with the EVEREST Transit() function.
        Takes parameters: period (per), duration (dur), and depth.
        Returns: transit model
        '''

        self.fpix = np.zeros((1000,5,5))
        self.t = np.linspace(0,90,len(self.fpix))
        self.trn = Transit(self.t, t0 = 5.0, per = self.per, dur = self.dur, depth = self.depth)

        return self.trn

    def GeneratePSF(self):
        '''
        Generate a PSF model that includes modeled inter-pixel sensitivity variation.
        Model includes photon noise and background nosie.
        Returns: fpix and ferr
        '''

        # read in relevant data
        ID = 205998445
        client = k2plr.API()
        star = client.k2_star(self.ID)
        tpf = star.get_target_pixel_files(fetch = True)[0]
        ftpf = os.path.join(KPLR_ROOT, 'data', 'k2', 'target_pixel_files', '%d' % self.ID, tpf._filename)
        with pyfits.open(ftpf) as f:
            xpos = f[1].data['pos_corr1']
            ypos = f[1].data['pos_corr2']

        # throw out outliers
        for i in range(len(xpos)):
            if abs(xpos[i]) >= 50 or abs(ypos[i]) >= 50:
                xpos[i] = 0
                ypos[i] = 0
            if np.isnan(xpos[i]):
                xpos[i] = 0
            if np.isnan(ypos[i]):
                ypos[i] = 0

        xpos = xpos[:1000]
        ypos = ypos[:1000]

        # define intra-pixel sensitivity variation
        intra = np.zeros((5,5))
        for i in range(5):
            for j in range(5):
                intra[i][j] = (0.995 + np.random.randint(10) / 1000)

        # mask transits
        self.naninds = np.where(self.trn < 1)
        self.M = lambda x: np.delete(x, self.naninds, axis = 0)

        # generate PRF model with inter-pixel sensitivity variation

        cx = [1.0,0.0,0.0]
        cy = [1.0,0.0,0.0]
        x0 = 2.5
        y0 = 2.5
        sx = [0.7]
        sy = [0.4]
        rho = [0.05]
        neighborcoords = [4.,4.]

        magdiff = 1.0
        r = 10**(magdiff / 2.5)

        self.target = np.zeros((len(self.fpix),5,5))
        self.ferr = np.zeros((len(self.fpix),5,5))
        background_level = 800

        for c in range(1000):
            for i in range(5):
                for j in range(5):
                    # contribution from target
                    target_val = self.trn[c]*PixelFlux(cx,cy,[self.A],[x0-i+xpos[c]],[y0-j+ypos[c]],sx,sy,rho)

                    # contribution from neighbor
                    val = target_val + (1/r)*PixelFlux(cx,cy,[self.A],[neighborcoords[0]-i+xpos[c]],[neighborcoords[1]-j+ypos[c]],sx,sy,rho)

                    self.target[c][i][j] = target_val
                    self.fpix[c][i][j] = val

                    # add photon noise
                    self.ferr[c][i][j] = np.sqrt(self.fpix[c][i][j])
                    randnum = np.random.randn()
                    self.fpix[c][i][j] += self.ferr[c][i][j] * randnum
                    self.target[c][i][j] += self.ferr[c][i][j] * randnum

                    # add background noise
                    noise = np.sqrt(background_level) * np.random.randn()
                    self.fpix[c][i][j] += background_level + noise
                    self.target[c][i][j] += background_level + noise

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
