import matplotlib.pyplot as pl
import numpy as np
import simulateK2 as sK2
import aperturefit as af
import sputter as sp

def mkplot(mag, motion):

    MN = sp.MotionNoise()
    flux, rawflux = MN.DetrendFpix(mag, motion)

    pl.plot(rawflux,'r.',alpha = 0.5,label='Raw Flux')
    pl.plot(flux,'k.',alpha = 0.75,label='Detrended')
    pl.xlabel('Time (days)')
    pl.ylabel('Flux (counts)')
    legend = pl.legend(loc=0)

    pl.show()

mkplot(10, 20)
