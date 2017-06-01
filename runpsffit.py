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

fit = pf.PSFFit(fpix,ferr)

amp = [355000.0,(355000.0 / 2)]
x0 = [2.5,4.]
y0 = [2.5,4.]
sx = [.5]
sy = [.5]
rho = [0.01]
background = [800]

guess = np.concatenate([amp,x0,y0,sx,sy,rho,background])
answer = fit.FindSolution(guess, index=200)
neighborvals = np.zeros((len(answer)))
for i in range(len(answer)):
    if i == 0:
        neighborvals[i] = 0
    else:
        neighborvals[i] = answer[i]

fit1 = fit.PSF(answer)
neighborfit = fit.PSF(neighborvals)
residual = fit1 - neighborfit

fig, ax = pl.subplots(1,3, sharey=True)
fig.set_size_inches(17,5)

meanfpix = np.mean(fpix,axis=0)
ax[0].imshow(fpix[200],interpolation='nearest',origin='lower',cmap='viridis',vmin=np.min(fit1),vmax=np.max(fit1));
ax[1].imshow(fit1,interpolation='nearest',origin='lower',cmap='viridis',vmin=np.min(fit1),vmax=np.max(fit1));
ax[2].imshow(fpix[200]-neighborfit,interpolation='nearest',origin='lower',cmap='viridis',vmin=np.min(fit1),vmax=np.max(fit1));
ax[0].set_title('Data');
ax[1].set_title('Fit');
ax[2].set_title('Neighbor Subtraction');

# pl.imshow(fit1-fpix[200],interpolation='nearest',origin='lower',cmap='viridis');pl.colorbar();

fig = pl.figure()
pl.imshow(residual,interpolation='nearest',origin='lower',cmap='viridis'); pl.colorbar()
print(datetime.now() - startTime)
pl.show()
