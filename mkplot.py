import matplotlib.pyplot as pl
import numpy as np
import simulateK2 as sK2
import aperturefit as af
import sputter as sp
from tqdm import tqdm

def mkplot(mag, motion):

    MN = sp.MotionNoise()
    flux, rawflux = MN.DetrendFpix(mag, motion)

    pl.plot(rawflux,'r.',alpha = 0.5,label='Raw Flux')
    pl.plot(flux,'k.',alpha = 0.75,label='Detrended')
    pl.xlabel('Time (days)')
    pl.ylabel('Flux (counts)')
    legend = pl.legend(loc=0)

    pl.show()

def rawCDPP(mag,motion):

    MN = sp.MotionNoise()
    flux,rawflux = MN.DetrendFpix(mag,motion)

    rawCDPP = MN.CDPP(rawflux)

    return rawCDPP
'''
sk2 = sK2.Target(205998445)
sk2.Transit()
sk2.GeneratePSF(159000.0)

fullcdpp = []
for m in tqdm(range(5)):
    mag = m + 10
    cdpp = []
    for mot in tqdm(range(20)):
        motion = mot+1
        cdpp.append(rawCDPP(mag,motion))
    fullcdpp.append(cdpp)

print(fullcdpp)
'''

mkplot(12,19)
