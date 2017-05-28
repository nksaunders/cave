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


class ApertureTest(object):

    def __init__(self, ID, amplitude):

        self.A = amplitude
        self.ID = ID

    def Transit(self, per, dur, depth):

        self.depth = depth

        self.fpix = np.zeros((1000,5,5))
        self.t = np.linspace(0,90,len(self.fpix))
        self.trn = Transit(self.t, t0 = 5.0, per = per, dur = dur, depth = depth)

        return self.trn

    def GeneratePSF(self):

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

        cx = [1.0,1.0,1.0]
        cy = [1.0,1.0,1.0]
        x0 = 2.5
        y0 = 2.5
        sx = [0.5]
        sy = [0.5]
        rho = [0]

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
                    val = target_val + (1/r)*PixelFlux(cx,cy,[self.A],[4.-i+xpos[c]],[4.-j+ypos[c]],sx,sy,rho)

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

        return self.fpix, self.ferr


    def CenterOfFlux(self, fpix):
        ncad, ny, nx = fpix.shape
        x0 = np.zeros(ncad)
        y0 = np.zeros(ncad)
        for n in range(ncad):
            x0[n] = np.sum([(i+0.5)*fpix[n][:,i] for i in range(nx)]) / np.sum(fpix[n])
            y0[n] = np.sum([(ny-j-0.5)*fpix[n][j,:] for j in range(ny)]) / np.sum(fpix[n])

        return x0,y0

    def Crowding(self, fpix):
        # crowding parameter for each pixel
        self.c_pix = np.zeros((len(fpix),5,5))

        # crowding parameter for entire detector
        self.c_det = np.zeros((len(fpix)))

        for c in range(len(fpix)):
            for i in range(5):
                for j in range(5):
                    if np.isnan(fpix[c][i][j]):
                        continue
                    else:
                        self.c_pix[c][i][j] = self.target[c][i][j] / fpix[c][i][j]

            self.c_det[c] = np.nansum(self.target[c]) / np.nansum(fpix[c])

        return self.c_det, self.c_pix

    def FirstOrderPLD(self, fpix):

        fpix_rs = fpix.reshape(len(fpix),-1)
        flux = np.sum(fpix_rs,axis=1)

        # mask transits
        X = fpix_rs / flux.reshape(-1,1)
        MX = self.M(fpix_rs) / self.M(flux).reshape(-1,1)

        # perform first order PLD
        A = np.dot(MX.T, MX)
        B = np.dot(MX.T, self.M(flux))
        C = np.linalg.solve(A, B)

        # compute detrended light curve
        model = np.dot(X, C)
        detrended = flux - model + np.nanmean(flux)

        D = (detrended - np.dot(C[1:], X[:,1:].T) + np.nanmedian(detrended)) / np.nanmedian(detrended)
        T = (t - 5.0 - per / 2.) % per - per / 2.

        return detrended, flux

    def RecoverTransit(self, lightcurve_in):
        '''
        Solve for depth of transit in detrended lightcurve
        '''

        detrended = lightcurve_in

        # normalize transit model
        transit_model = (self.trn - 1) / self.depth

        # create relevant arrays
        X = np.array(([],[]), dtype = float).T
        for i in range(len(self.fpix)):
            rowx = np.array([[1.,transit_model[i]]])
            X = np.vstack((X, rowx))
        Y = detrended / np.nanmedian(detrended)

        # solve for recovered transit depth
        A = np.dot(X.T, X)
        B = np.dot(X.T, Y)
        C = np.linalg.solve(A, B)
        rec_depth = C[1]

        return rec_depth

    def AperturePLD(self, fpix, aperture):

        aperture = [aperture for i in range(len(fpix))]

        # import pdb; pdb.set_trace()
        fpix_rs = (fpix*aperture).reshape(len(fpix),-1)
        fpix_ap = np.zeros((len(fpix),len(np.delete(fpix_rs[0],np.where(np.isnan(fpix_rs[0]))))))

        for c in range(len(fpix_rs)):
            naninds = np.where(np.isnan(fpix_rs[c]))
            fpix_ap[c] = np.delete(fpix_rs[c],naninds)

        flux = np.sum(fpix_ap,axis=1)
        X = fpix_ap / flux.reshape(-1,1)
        MX = self.M(fpix_ap) / self.M(flux).reshape(-1,1)

        # perform first order PLD
        A = np.dot(MX.T, MX)
        B = np.dot(MX.T, self.M(flux))
        C = np.linalg.solve(A, B)

        # compute detrended light curve
        model = np.dot(X, C)
        detrended = flux - model + np.nanmean(flux)

        return aperture,detrended,flux
