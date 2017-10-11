import numpy as np
import sputter as sp
import matplotlib.pyplot as pl
import aperturefit as aft


def MakePlots(nmags = 5, nmot = 22, pwd = 'stars/larger_aperture/'):

    CDPPs = []
    rCDPPs = []
    MN = sp.MotionNoise()

    for n in range(nmags):

        mag = n + 10
        mcdpp = []
        rcdpp = []

        for mot in range(nmot):
            flux, rawflux = MN.DetrendFpix(mag, mot, pwd = pwd)

            mcdpp.append(MN.CDPP(flux))
            rcdpp.append(MN.CDPP(rawflux))

        CDPPs.append(mcdpp)
        rCDPPs.append(rcdpp)

    for i,c in enumerate(CDPPs):

        fig = pl.figure()

        pl.plot(c,'r')
        pl.plot(rCDPPs[i],'k')

        pl.title(r'$K_p\ Mag = $' + str(i+10))
        pl.xlabel('K2 Roll Coefficient')
        pl.ylabel('CDPP (ppm)')

    pl.show()

MakePlots(nmags=5,nmot=12)
