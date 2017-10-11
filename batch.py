import numpy as np
import simulateK2
from tqdm import tqdm
import itertools
from everest.pool import Pool
from everest.missions.k2 import CDPP
from everest.config import EVEREST_SRC
import os

def Simulate(arg):
    mag, m_mag = arg
    print("Running mag = %.2f, m_mag = %.2f" % (mag, m_mag))
    sK2 = simulateK2.Target(205998445, ftpf = os.path.expanduser('~/.kplr/data/k2/target_pixel_files/205998445/ktwo205998445-c03_lpd-targ.fits.gz'))
    sK2.Transit()
    fpix, target, ferr = sK2.GeneratePSF(mag, motion_mag = m_mag)
    np.savez('batch/mag%.2fmotion%.2f' % (mag, m_mag), fpix = fpix)

def Benchmark():
    '''
    
    '''
    
    import matplotlib.pyplot as pl
    
    mags = np.arange(10., 17.5, 0.5)
    cdpp = np.zeros_like(mags)
    
    # Compare zero-motion synthetic data to original Kepler raw CDPP
    for i, mag in enumerate(mags):
        fpix = np.load('batch/mag%.2fmotion%.2f.npz' % (mag, 0.))['fpix']
        flux = np.sum(fpix, axis = (1,2))
        cdpp[i] = CDPP(flux)
    _, kepler_kp, kepler_cdpp6 = np.loadtxt(os.path.join(EVEREST_SRC, 'missions', 'k2', 'tables', 'kepler.cdpp'), unpack = True)
    fig, ax = pl.subplots(1)
    ax.plot(kepler_kp, kepler_cdpp6, 'y.', alpha = 0.01, zorder = -1)
    ax.set_rasterization_zorder(-1)
    bins = np.arange(7.5,18.5,0.5)
    by = np.zeros_like(bins) * np.nan
    for b, bin in enumerate(bins):
        i = np.where((kepler_cdpp6 > -np.inf) & (kepler_cdpp6 < np.inf) & (kepler_kp >= bin - 0.5) & (kepler_kp < bin + 0.5))[0]
        if len(i) > 10:
            by[b] = np.median(kepler_cdpp6[i])
    ax.plot(bins, by, 'yo', label = 'Kepler', markeredgecolor = 'k')
    ax.plot(mags, cdpp, 'bo', label = 'Synthetic (0x motion)')
    ax.set_xlabel('Kepler Magnitude')
    ax.set_ylabel('CDPP [ppm]')
    ax.set_ylim(-10, 500)
    ax.set_xlim(8, 18)
    ax.legend(loc = 'best')

    # Compare 1x motion to K2 raw CDPP from campaign 3
    for i, mag in enumerate(mags):
        fpix = np.load('batch/mag%.2fmotion%.2f.npz' % (mag, 1.))['fpix']
        flux = np.sum(fpix, axis = (1,2))
        cdpp[i] = CDPP(flux)
    _, kp, cdpp6r, _, _, _, _, _, _ = np.loadtxt(os.path.join(EVEREST_SRC, 'missions', 'k2', 'tables', 'c03_nPLD.cdpp'), unpack = True, skiprows = 2)
    fig, ax = pl.subplots(1)
    ax.plot(kp, cdpp6r, 'r.', alpha = 0.05, zorder = -1)
    ax.set_rasterization_zorder(-1)
    bins = np.arange(7.5,18.5,0.5)
    by = np.zeros_like(bins) * np.nan
    for b, bin in enumerate(bins):
        i = np.where((cdpp6r > -np.inf) & (cdpp6r < np.inf) & (kp >= bin - 0.5) & (kp < bin + 0.5))[0]
        if len(i) > 10:
            by[b] = np.median(cdpp6r[i])
    ax.plot(bins, by, 'ro', label = 'Raw K2', markeredgecolor = 'k')
    ax.plot(mags, cdpp, 'bo', label = 'Synthetic (1x motion)')
    ax.set_xlabel('Kepler Magnitude')
    ax.set_ylabel('CDPP [ppm]')
    ax.set_ylim(-30, 1500)
    ax.set_xlim(8, 18)
    ax.legend(loc = 'best')
    
    pl.show()
    
if __name__ == '__main__':

    # Run!   
    mags = np.arange(10., 17.5, 0.5)
    m_mags = np.arange(0., 22, 1)
    combs = list(itertools.product(mags, m_mags))
    with Pool() as pool:
        pool.map(Simulate, combs)