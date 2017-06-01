import aperturefit as af
import numpy as np
import matplotlib.pyplot as pl
import psffit as pf
import simulateK2
from datetime import datetime

sK2 = simulateK2.Target(205998445, 355000.0)
trn = sK2.Transit()
fpix,target, ferr = sK2.GeneratePSF()

t = af.ApertureFit(fpix,target,ferr,trn)
c_pix, c_det = t.Crowding()

# define aperture
aperture1 = np.zeros((5,5))
aperture2 = np.zeros((5,5))

for i in range(5):
    for j in range(5):
        if i == 3 and j == 3:
            aperture1[i][j] = 1.0
            aperture2[i][j] = np.nan
        elif -1 < i < 4 and -1 < j < 4:
            aperture1[i][j] = 1.0
            aperture2[i][j] = 1.0
        else:
            aperture1[i][j] = np.nan
            aperture2[i][j] = np.nan

ap1, apdetrended1, apflux1 = t.AperturePLD(aperture1)
ap2, apdetrended2, apflux2 = t.AperturePLD(aperture2)
rd_ap1 = t.RecoverTransit(apdetrended1)
rd_ap2 = t.RecoverTransit(apdetrended2)

fig1 = pl.figure(figsize=(16,5))
time = np.linspace(0,90,len(fpix))
pl.plot(time,apdetrended2,'k.');
pl.plot(time,np.mean(apflux2)*trn,'r');

fig, ax = pl.subplots(1,3, sharey=True)
fig.set_size_inches(17,5)

ax[0].imshow(fpix[-1],interpolation='nearest',origin='lower',cmap='viridis')
ax[1].imshow(ap1[-1]*fpix[-1],interpolation='nearest',origin='lower',cmap='viridis')
ax[2].imshow(ap2[-1]*fpix[-1],interpolation='nearest',origin='lower',cmap='viridis')

ax[1].annotate(r'$\mathrm{Recovered\ Depth}: %.4f$' % (rd_ap1),
                xy = (0.05, 0.1),xycoords='axes fraction',
                color='w', fontsize=12);
ax[2].annotate(r'$\mathrm{Recovered\ Depth}: %.4f$' % (rd_ap2),
                xy = (0.05, 0.1),xycoords='axes fraction',
                color='w', fontsize=12);

print(datetime.now() - startTime)
pl.show()
