import numpy as np
import matplotlib.pyplot as pl
import everest
from intrapix import PixelFlux
from random import randint
from astropy.io import fits
import pyfits
from everest import Transit
import k2plr
import os
from scipy.optimize import fmin_powell

class PSFFit(object):

    def __init__(self, fpix, ferr):

        self.nsrc = 2
        self.xtol = 0.0001
        self.ftol = 0.0001
        self.fpix = fpix
        self.ferr = ferr

    def PSF(self, params):
        amp1,amp2,x01,x02,y01,y02,rho = params

        cx_1 = [1.,0.,0.]
        cx_2 = [1.,0.,0.]
        cy_1 = [1.,0.,0.]
        cy_2 = [1.,0.,0.]
        sx = 0.5
        sy = 0.5
        # rho = 0.0


        model = np.zeros((5,5))
        for i in range(5):
            for j in range(5):
                model[i][j] = PixelFlux(cx_1, cy_1, [amp1], [x01-i], [y01-j], [sx], [sy], [rho]) + \
                              PixelFlux(cx_2, cy_2, [amp2], [x02-i], [y02-j], [sx], [sy], [rho])
        return model

    def Fit(self, params):
        '''
        Returns a simple :py:class:`Fit` object that
        contains information about the PSF fit.

        '''

        # read variables
        amp1,amp2,x01,x02,y01,y02,rho = params

        fit = self.PSF(params)

        # instance the class
        return type('Fit', (object,), {'amp1' : amp1, 'amp2' : amp2, 'x01' : x01, 'x02' : x02, 'y01' : y01, 'y02' : y02,
                                       'rho' : rho, 'fit': fit})


    # fit PSF for target and neighbor simultaneously

    def LnLikelihood(self, params):

        amp1,amp2,x01,x02,y01,y02,rho = params

        if rho >= 1 or rho <= -1:
            return 1.0e30

        # constrain position values
        if ((2.5 - x01)**2 + (2.5 - y01)**2) > 16:
            return 1.0e30
        if ((4. - x02)**2 + (4. - y02)**2) > 16:
            return 1.0e30


        #if (sx > 2 * np.pi) or (sy > 2 * np.pi):
        #    PSFres = 1.0e300

        # Reject negative values
        for elem in [amp1,amp2,x01,x02,y01,y02]:
            if elem < 0:
                return 1.0e30

        fit = self.Fit(params)
        PSFfit = fit.fit
        index = self.index
        dtol = 0.01
        x01 = fit.x01
        y01 = fit.y01
        x02 = fit.x02
        y02 = fit.x02
        rho = fit.rho

        meanfpix = np.mean(self.fpix)
        meanferr = np.mean(self.ferr)

        # sum squared difference between data and model
        PSFres = np.nansum(((self.fpix[index] - PSFfit) / self.ferr[index]) ** 2)
        s_s = 1.
        '''
        PSFres += np.nansum(((np.array([2.5,4.]) - np.array([x01,x02])) / dtol) ** 2 + \
                            ((np.array([2.5,4.]) - np.array([y01,y02])) / dtol) ** 2)
        if max(np.abs(2.5 - x01), np.abs(2.5 - y01)) > 10.0:
        PRFres = 1.0e300


        # Prior likelihood
        sx0 = 0.5 + s_s * np.random.randn()
        sy0 = 0.5 + s_s * np.random.randn()
        PSFres += ((sx - sx0) / s_s)**2
        PSFres += ((sy - sy0) / s_s)**2

        '''

        print("L = %.2e, x1 = %.2f, x2 = %.2f, y1 = %.2f, y2 = %.2f, rho = %.2f, a1 = %.2f, a2 = %.2f" % (PSFres, x01, x02, y01, y02, rho, amp1, amp2))

        return PSFres

    def FindSolution(self, guess, index=100):

        self.guess = guess
        self.index = index


        answer, chisq, _, iter, funcalls, warn = fmin_powell(self.LnLikelihood, self.guess, xtol = self.xtol, ftol = self.ftol,
                                                             disp = False, full_output = True)

        bic = chisq + len(answer) * np.log(len(self.fpix))

        return answer
