import numpy as np
import simulateK2
from tqdm import tqdm
import itertools
from everest.pool import Pool
import os

def Simulate(arg):
    mag, m_mag = arg
    print("Running mag = %.2f, m_mag = %.2f" % (mag, m_mag))
    sK2 = simulateK2.Target(205998445, ftpf = os.path.expanduser('~/.kplr/data/k2/target_pixel_files/205998445/ktwo205998445-c03_lpd-targ.fits.gz'))
    sK2.Transit()
    fpix, target, ferr = sK2.GeneratePSF(mag, motion_mag = m_mag)
    np.savez('batch/mag%.2fmotion%.2f' % (mag, m_mag), fpix = fpix)

# Run!   
mags = np.arange(10., 17.5, 0.5)
m_mags = np.arange(0., 22, 1)
combs = list(itertools.product(mags, m_mags))
with Pool() as pool:
    pool.map(Simulate, combs)