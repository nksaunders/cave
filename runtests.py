import aperturefit as af
import numpy as np
import matplotlib.pyplot as pl
import psffit as pf
from datetime import datetime
# %matplotlib inline

startTime = datetime.now()

status = 0
'''
status = 0: perform PSF fitting
status = 1: perform aperture PLD
'''

t = af.ApertureFit(205998445, 355000.0)
trn = t.Transit(per = 15, dur = 0.5, depth = 0.01)

fpix,ferr = t.GeneratePSF()

if status == 1:
    c_pix, c_det = t.Crowding(fpix)

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

    ap1, apdetrended1, apflux1 = t.AperturePLD(fpix, aperture1)
    ap2, apdetrended2, apflux2 = t.AperturePLD(fpix, aperture2)
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

if status == 0:
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

    fig, ax = pl.subplots(1,2, sharey=True)
    fig.set_size_inches(11,5)

    meanfpix = np.mean(fpix,axis=0)

    ax[0].imshow(fit1,interpolation='nearest',origin='lower',cmap='viridis');
    ax[1].imshow(fit1 - fpix[200],interpolation='nearest',origin='lower',cmap='viridis');
    ax[0].set_title('Fit');
    ax[1].set_title('Residuals');

    pl.imshow(fit1-fpix[200],interpolation='nearest',origin='lower',cmap='viridis');pl.colorbar();

    fig = pl.figure()
    pl.imshow(residual,interpolation='nearest',origin='lower',cmap='viridis'); pl.colorbar()
    print(datetime.now() - startTime)
    pl.show()

    # import pdb; pdb.set_trace()
