import aperturefit as af
import numpy as np
import matplotlib.pyplot as pl
import psffit as pf
import simulateK2
import everest as ev
from astropy.stats import median_absolute_deviation as mad
import k2plr
from k2plr.config import KPLR_ROOT
from everest.config import KEPPRF_DIR
import os
import pyfits


def FluxMotion(ID):

    sK2 = simulateK2.Target(int(ID), 454000.0)
    trn = sK2.Transit()
    aft = af.ApertureFit(trn)

    fpix, target, ferr = sK2.GeneratePSF(motion_mag = 4)
    t = np.linspace(0,90,len(fpix))

    xpos = sK2.xpos
    ypos = sK2.ypos

    motion = np.sqrt(xpos**2 + ypos**2)

    fpix_crop = np.array([fp[2:7,2:7] for fp in fpix])
    dtrn, flux = aft.FirstOrderPLD(fpix_crop)

    mad_flux = mad(flux)
    mad_motion = mad(motion)

    return mad_flux, mad_motion


def RealMotion(ID):

    client = k2plr.API()
    star = client.k2_star(ID)
    tpf = star.get_target_pixel_files(fetch = True)[0]
    ftpf = os.path.join(KPLR_ROOT, 'data', 'k2', 'target_pixel_files', '%d' % ID, tpf._filename)
    with pyfits.open(ftpf) as f:
        xpos = f[1].data['pos_corr1']
        ypos = f[1].data['pos_corr2']

        for i in range(len(xpos)):
            if abs(xpos[i]) >= 50 or abs(ypos[i]) >= 50:
                xpos[i] = 0
                ypos[i] = 0
            if np.isnan(xpos[i]):
                xpos[i] = 0
            if np.isnan(ypos[i]):
                ypos[i] = 0

    xpos = xpos[1000:2000]
    ypos = ypos[1000:2000]

    motion = np.sqrt(xpos**2 + ypos**2)

    targetstar = ev.Everest(ID)
    flux = targetstar.flux[1000:2000]

    naninds = np.where(np.isnan(flux))
    M = lambda x: np.delete(x,naninds,axis=0)

    mad_flux = mad(M(flux))
    mad_motion = mad(motion)

    return mad_flux, mad_motion


# mf, mm = RealMotion(213200501)
# mf1, mm1 = FluxMotion(212268272)
# print(str(mm) + ', ' + str(mf))
# import pdb; pdb.set_trace()

mmset = [0.107216, 0.214431, 0.321647, 0.428863, 0.536078,0.151758,0.303516,0.455274]
mfset = [205.550028234,177.540208289,291.339958971,692.731192149,1471.31395841,245.258560848,507.046253729,681.06771981]

mmreal = [0.123282,0.13561,0.35767,0.213153]
mfreal = [113.846282661,125.397836924,405.180929184,174.925886989]

fig = pl.figure()
pl.scatter(mmset,mfset, color='k')
pl.scatter(mmreal,mfreal, color='r')
pl.xlabel('Motion Vector Magnitude (pixels)')
pl.ylabel('Flux (counts)')
pl.title('Flux vs. Motion (Median Absolute Deviation)')
pl.show()
