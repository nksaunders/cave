import aperturefit as af
import numpy as np
import matplotlib.pyplot as pl
import psffit as pf
import simulateK2
from datetime import datetime
from tqdm import tqdm
from everest import detrender
from everest.math import SavGol, Scatter, Downbin
from astropy.stats import median_absolute_deviation as mad

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
        self.sK2 = simulateK2.Target(int(self.ID))
        self.trn = self.sK2.Transit()
        self.aft = af.ApertureFit(self.trn)

    def DetrendFpix(self, mag, motion):


        path = 'stars/mag' + str(mag) + 'motion' + str(motion) + '.npz'
        fpix = np.load(path)['fpix']

        fpix_rs = fpix.reshape(len(fpix),-1)
        tempflux = np.sum(fpix_rs,axis=1)

        crop = np.where(tempflux < (0.99*np.mean(tempflux)))[0]
        M = lambda x: np.delete(x, crop, axis=0)
        fpix = M(fpix)

        x0, y0 = self.sK2.CenterOfFlux(fpix)
        crop2 = []
        for n in range(len(fpix)):
            if (np.sqrt((x0[n]-15/2)**2+(y0[n]-15/2)**2) > 6.5):
                crop2.append(n)

        M2 = lambda x: np.delete(x, crop2, axis=0)
        fpix = M2(fpix)
        flux, rawflux = self.aft.PLD(fpix)

        return flux, rawflux

    def Create(self, mag, f_n = 20):
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
        f1, rf1 = self.DetrendFpix(mag, 0)
        self.true_cdpp = self.CDPP(f1)

        # iterate through 'f' values
        for f in tqdm(self.fset):
            temp_CDPP_set = []


            # take mean of 5 runs
            for i in tqdm(range(5)):
                flux, rawflux = self.DetrendFpix(mag, f)
                cdpp = self.CDPP(flux)
                temp_CDPP_set.append(cdpp)
                if i == 0:
                    self.flux_set.append(flux)

            cdppval = np.mean(temp_CDPP_set)
            self.CDPP_set.append(cdppval)


    def CDPP(self, flux, mask = [], cadence = 'lc'):
        '''
        Compute the proxy 6-hr CDPP metric.

        :param array_like flux: The flux array to compute the CDPP for
        :param array_like mask: The indices to be masked
        :param str cadence: The light curve cadence. Default `lc`
        '''

        self.trnvals = np.where(self.trn < 1.0)
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
        fig, ax = pl.subplots(f_n,1, sharex=True, sharey=True)

        # plot light curve for each coefficient of 'f'
        for f in range(f_n):
            ax[f].plot(self.flux_set[f],'k.')
            ax[f].set_title("f = %.1f" % (f+1))
            ax[f].set_ylabel("Flux (counts)")
            ax[f].annotate(r'$\mathrm{Mean\ CDPP}: %.2f$' % (self.CDPP_set[f]),
                            xy = (0.85, 0.05),xycoords='axes fraction',
                            color='k', fontsize=12);

        ax[f_n-1].set_xlabel("Time (cadences)")

        fig2 = pl.figure()

        self.CDPP_set_norm = [(n / self.true_cdpp) for n in self.CDPP_set]

        print(self.CDPP_set)
        print(self.CDPP_set_norm)

        pl.plot(self.fset,self.CDPP_set_norm,'r')
        pl.xlabel("f")
        pl.ylabel("Normalized CDPP")
        pl.title("Normalized CDPP vs. Motion Magnitude")
        pl.show()

MN = MotionNoise()
MN.Create(14)
MN.Plot()
