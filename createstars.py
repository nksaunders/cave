import numpy as np
import simulateK2
from tqdm import tqdm

def Simulate(ID, f_mag, m_mag):

    sK2 = simulateK2.Target(ID)
    sK2.Transit()

    # generate a simulated PSF
    fpix, target, ferr = sK2.GeneratePSF(f_mag, motion_mag = m_mag)

    return fpix

for i in tqdm(range(21)):
    fpix = Simulate(205998445, 1150000.0, i)
    np.savez(('stars/neighbor_mag10motion%.i'%i),fpix=fpix)
