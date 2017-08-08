import numpy as np
import simulateK2
from tqdm import tqdm

def Simulate(ID, f_mag, m_mag):

    sK2 = simulateK2.Target(ID)
    sK2.Transit()

    # generate a simulated PSF
    fpix, target, ferr = sK2.GeneratePSF(f_mag, motion_mag = m_mag)

    return fpix
'''
fval = [159000.0, 68000.0, 28000.0]
mags = [12, 13, 14]

for index,m in tqdm(enumerate(mags)):
    for i in tqdm(range(26)):

        fpix = Simulate(205998445, fval[index], i)
        np.savez(('stars/larger_aperture/neighbor_mag%imotion%.i'%(m,i)),fpix=fpix)
'''

for i in tqdm(range(26)):

    fpix = Simulate(205998445, 28000.0, i)
    np.savez(('stars/larger_aperture/mag14motion%i'%i),fpix=fpix)
