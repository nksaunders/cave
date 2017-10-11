import numpy as np
import sputter as sp
import matplotlib.pyplot as pl
import aperturefit as aft
from tqdm import tqdm

# Magnitude and motion arrays
mags = np.arange(10., 16., 1.)
m_mags = [1, 2, 5, 10, 20]

def MakePlots(nmags = 5, nmot = 22, pwd = 'batch/'):

    CDPPs = []
    rCDPPs = []
    MN = sp.MotionNoise()

    for mag in mags:
    
        mcdpp = []
        rcdpp = []

        print("Running magnitude %.1f..." % mag)

        for mot in m_mags:
            flux, rawflux = MN.DetrendFpix(mag, mot, pwd = pwd, star = 0, neighbors = [1])
            mcdpp.append(MN.CDPP(flux))
            rcdpp.append(MN.CDPP(rawflux))
        CDPPs.append(mcdpp)
        rCDPPs.append(rcdpp)
    
    # Save
    np.savez('batch/detrended_cdpps.npz', CDPPs = CDPPs, m_mags = m_mags, mags = mags)
    
    for i, c in enumerate(CDPPs):

        fig = pl.figure()

        pl.plot(m_mags, c,'r')
        pl.plot(m_mags, rCDPPs[i],'k')

        pl.title(r'$K_p\ Mag = $' + str(mags[i]))
        pl.xlabel('K2 Roll Coefficient')
        pl.ylabel('CDPP (ppm)')

    pl.show()

MakePlots()
