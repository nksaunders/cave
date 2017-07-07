import aperturefit as af
import numpy as np
import matplotlib.pyplot as pl
import psffit as pf
import simulateK2
from datetime import datetime
from tqdm import tqdm
from everest import detrender
from everest.math import SavGol, Scatter, Downbin

class MotionNoise(object):
    '''
    Simulates a star and transiting exoplanet
    Creates light curves for various coefficients of the K2 motion vectors
    Calculates CDPP and normalized CDPP for light curves
    '''

    def __init__(self):
        '''

        '''

        self.ID = 205998445
        self.startTime = datetime.now()

        # simulated a star, takes an ID and flux value (corresponding to magnitude)
        self.sK2 = simulateK2.Target(int(self.ID), 159000.0)
        self.trn = self.sK2.Transit()
        self.aft = af.ApertureFit(self.trn)

    def SimulateStar(self, f):
        '''
        returns raw flux and detrended flux light curves
        parameter int 'f': coefficient on motion vectors
        '''

        # generate a simulated PSF
        self.fpix, self.target, self.ferr = self.sK2.GeneratePSF(motion_mag = f)
        self.t = np.linspace(0,90,len(self.fpix))

        # raw flux light curve
        fpix_rs = self.fpix.reshape(len(self.fpix),-1)
        raw_flux = np.sum(fpix_rs,axis=1)

        # mask outliers due to flux loss off aperture
        self.trnvals = np.where(self.trn < 1)
        self.trnM = lambda x: np.delete(x, self.trnvals, axis = 0)
        self.maskvals = np.where((np.abs(raw_flux - (np.nanmean(raw_flux)) > (np.nanmean(raw_flux)*0.005))) & (self.trn == 1))
        self.M = lambda x: np.delete(x, self.maskvals, axis = 0)

        raw_flux = self.M(raw_flux)
        # motion vectors
        self.xpos = self.sK2.xpos
        self.ypos = self.sK2.ypos

        # reduce aperture size to 5x5, run first order PLD
        self.fpix_crop = np.array([fp[3:8,3:8] for fp in self.fpix])
        dtrn, flux = self.aft.FirstOrderPLD(self.M(self.fpix_crop))

        return raw_flux, dtrn

    def Create(self, f_n = 5):
        '''
        calculates CDPP for light curves for coefficients 'f' up to 'f_n'
        parameter 'f_n': number of coefficients to test
        '''

        self.fset = [(i+1) for i in range(f_n)]

        self.flux_set = []
        self.CDPP_set = []
        self.f_n = f_n

        print("Testing Motion Magnitudes...")

        # calculate no-motion case
        rf1, f1 = self.SimulateStar(0)
        self.true_cdpp = self.CDPP(f1)

        # iterate through 'f' values
        for f in tqdm(self.fset):
            temp_CDPP_set = []

            # take mean of 5 runs
            for i in tqdm(range(1)):
                raw_flux, flux = self.SimulateStar(f)
                cdpp = self.CDPP(flux)
                temp_CDPP_set.append(cdpp)
                if i == 0:
                    self.flux_set.append(flux)

            self.CDPP_set.append(np.mean(temp_CDPP_set))

    def CDPP(self, flux, mask = [], cadence = 'lc'):
        '''
        Compute the proxy 6-hr CDPP metric.

        :param array_like flux: The flux array to compute the CDPP for
        :param array_like mask: The indices to be masked
        :param str cadence: The light curve cadence. Default `lc`
        '''

        mask = self.trnvals

        # 13 cadences is 6.5 hours
        rmswin = 13
        # Smooth the data on a 2 day timescale
        svgwin = 49

        # If short cadence, need to downbin
        if cadence == 'sc':
            newsize = len(flux) // 30
            flux = Downbin(flux, newsize, operation = 'mean')

        flux_savgol = SavGol(np.delete(flux, mask), win = svgwin)
        if len(flux_savgol):
            return Scatter(flux_savgol / np.nanmedian(flux_savgol), remove_outliers = True, win = rmswin)
        else:
            return np.nan

    def Plot(self):
        '''
        plot 1: light curves from each coefficient 'f'
        plot 2: Normalized CDPP vs. coefficient 'f'
        '''

        f_n = self.f_n
        fig, ax = pl.subplots(f_n,1, sharex=True)

        # plot light curve for each coefficient of 'f'
        for f in range(f_n):
            ax[f].plot(self.flux_set[f],'k.')
            ax[f].set_title("f = %.1f" % (f+1))
            ax[f].set_ylabel("Flux (counts)")
            ax[f].annotate(r'$\mathrm{Mean\ CDPP}: %.2f$' % (self.CDPP_set[f]),
                            xy = (0.85, 0.05),xycoords='axes fraction',
                            color='k', fontsize=12);

        ax[f_n-1].set_xlabel("Time (days)")

        fig2 = pl.figure()

        self.CDPP_set_norm = [(n / self.true_cdpp) for n in self.CDPP_set]

        print(self.CDPP_set)
        print(self.CDPP_set_norm)

        pl.plot(self.fset,self.CDPP_set_norm,'r')
        pl.xlabel("f")
        pl.ylabel("Normalized CDPP")
        pl.title("Normalized CDPP vs. Motion Magnitude")
        pl.show()
        import pdb; pdb.set_trace()

MN = MotionNoise()
MN.Create()
MN.Plot()
