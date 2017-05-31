import numpy as np
import matplotlib.pyplot as pl
import everest
from everest.math import SavGol
from intrapix import PixelFlux
import simulateK2
from random import randint
from astropy.io import fits
import pyfits
from everest import Transit
import k2plr
from k2plr.config import KPLR_ROOT
from everest.config import KEPPRF_DIR
import os

class ApertureFit(object):

    def __init__(self, ID, amplitude):

        # initialize variables
        sK2 = simulateK2.Target(ID, amplitude)
        self.trn = sK2.Transit()
        self.fpix, self.ferr = sK2.GeneratePSF()

    def Crowding(self, fpix):
        '''
        Calculates and returns pixel crowding (c_pix) and detector crowding (c_det)
        Crowding defined by F_target / F_total
        '''

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
        '''
        Perform first order PLD on a light curve
        Returns: detrended light curve, raw light curve
        '''

        #  generate flux light curve
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
        Returns: recovered depth of transit in light curve
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
        '''
        Performs PLD on only a desired region of the detector
        Takes parameters: light curve (fpix), and aperture containing desired region
        Returns: aperture, detrendended light curve, raw light curve in aperture
        '''

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
