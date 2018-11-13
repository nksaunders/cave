import numpy as np
import simulateK2
from tqdm import tqdm

def Simulate(ID, mag, m_mag):

    sK2 = simulateK2.Target(ID,npts = 50)
    sK2.Transit()

    # generate a simulated PSF
    fpix, target, ferr = sK2.GeneratePSF(mag, motion_mag = m_mag)

    return fpix
'''
fval = [159000.0, 68000.0, 28000.0]
mags = [12, 13, 14]

for index,m in tqdm(enumerate(mags)):
    for i in tqdm(range(26)):

        fpix = Simulate(205998445, fval[index], i)
        np.savez(('stars/larger_aperture/neighbor_mag%imotion%.i'%(m,i)),fpix=fpix)
'''

fpix = Simulate(205998445, 12., 5)



# np.savez('stars/tests/mag11motion0',fpix=fpix)

import pdb; pdb.set_trace()
'''
import simulateK2target as sK2
import matplotlib.pyplot as pl
import numpy as np
t = sK2.Target(205998445, transit=True)
fp, ta, fe = t.GeneratePSF(12.)

flux = np.sum(fp.reshape(len(fp),-1),axis=1)
pl.plot(flux,'k.');pl.show()
'''
