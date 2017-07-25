import matplotlib.pyplot as pl
import numpy as np
import simulateK2 as sK2
import aperturefit as af
import sputter as sp

def mkplot(f_mag, motion):
    
    MN = sp.MotionNoise(f_mag)
    raw,flux = MN.SimulateStar(motion)
    t = np.linspace(0,90,len(flux))

    pl.plot(t,raw,'r.',alpha = 0.5,label='Raw Flux')
    pl.plot(t,flux,'k.',alpha = 0.75,label='Detrended')
    pl.xlabel('Time (days)')
    pl.ylabel('Flux (counts)')
    legend = pl.legend(loc=0)

    pl.show()

mkplot(159000.0, 3.0)
