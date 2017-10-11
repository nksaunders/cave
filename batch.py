import numpy as np
import simulateK2
from tqdm import tqdm
import itertools
from everest.pool import Pool
from everest.missions.k2 import CDPP
from everest.config import EVEREST_SRC
import os

# Number of targets to simulate
niter = 5

# Magnitude and motion arrays
mags = np.arange(10., 17.5, 0.5)
m_mags = np.arange(0., 22, 1)

# DEBUG
niter = 2
mags = [10., 11., 12., 13.]
m_mags  = [0., 1.]

def Simulate(arg):
    iter, mag, m_mag = arg
    print("Running mag = %.2f, m_mag = %.2f" % (mag, m_mag))
    sK2 = simulateK2.Target(205998445, depth = 0., npts = 100, ftpf = os.path.expanduser('~/.kplr/data/k2/target_pixel_files/205998445/ktwo205998445-c03_lpd-targ.fits.gz'))
    sK2.Transit()
    fpix, target, ferr = sK2.GeneratePSF(mag, motion_mag = m_mag)
    np.savez('batch/%2dmag%.2fmotion%.2f' % (iter, mag, m_mag), fpix = fpix)

def Benchmark():
    '''
    
    '''
    
    import matplotlib.pyplot as pl
    
    # Compare zero-motion synthetic data to original Kepler raw CDPP
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
    for iter in range(niter):
        cdpp = np.zeros_like(mags)
        for i, mag in enumerate(mags):
            fpix = np.load('batch/%2dmag%.2fmotion%.2f.npz' % (iter, mag, 0.))['fpix']
            flux = np.nansum(fpix, axis = (1,2))
            cdpp[i] = CDPP(flux)
        if iter == 0:
            ax.plot(mags, cdpp, 'b.', label = 'Synthetic (0x motion)')
        else:
            ax.plot(mags, cdpp, 'b.')
    ax.set_xlabel('Kepler Magnitude')
    ax.set_ylabel('CDPP [ppm]')
    ax.set_ylim(-10, 500)
    ax.set_xlim(8, 18)
    ax.legend(loc = 'best')
    
    # Compare 1x motion to K2 raw CDPP from campaign 3
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
    for iter in range(niter):
        cdpp = np.zeros_like(mags)
        for i, mag in enumerate(mags):
            fpix = np.load('batch/%2dmag%.2fmotion%.2f.npz' % (iter, mag, 1.))['fpix']
            flux = np.nansum(fpix, axis = (1,2))
            cdpp[i] = CDPP(flux)
        if iter == 0:
            ax.plot(mags, cdpp, 'b.', label = 'Synthetic (1x motion)')
        else:
            ax.plot(mags, cdpp, 'b.')
    ax.set_xlabel('Kepler Magnitude')
    ax.set_ylabel('CDPP [ppm]')
    ax.set_ylim(-30, 1500)
    ax.set_xlim(8, 18)
    ax.legend(loc = 'best')
    
    pl.show()
    
if __name__ == '__main__':

    # Run!   
    combs = list(itertools.product(range(niter), mags, m_mags))
    with Pool() as pool:
        pool.map(Simulate, combs)