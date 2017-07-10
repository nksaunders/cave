import aperturefit as af
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.animation as animation
import psffit as pf
import simulateK2
from datetime import datetime
from tqdm import tqdm


ID = 205998445

# simulated a star, takes an ID and flux value (corresponding to magnitude)
sK2 = simulateK2.Target(int(ID), 159000.0)
trn = sK2.Transit()
aft = af.ApertureFit(trn)

fpix, target, ferr = sK2.GeneratePSF(motion_mag = 5)

t = np.linspace(0,90,len(fpix))

def images(i):
    fpix_im = pl.imshow(fpix[i],cmap='viridis',origin='lower',interpolation='nearest')
    pl.title('5x Motion')
    pl.annotate(r'$\mathrm{Time}: %.2f \mathrm{\ days}$' % (t[i]),
                    xy = (0.72, 0.05),xycoords='axes fraction',
                    color='k', fontsize=12, bbox=dict(facecolor='w', edgecolor='k'));
    return fpix_im

fig = pl.figure()
ani = animation.FuncAnimation(fig, images, frames=750, interval = 1, repeat=True)
Writer = animation.writers['ffmpeg']
writer = Writer(fps = 15, metadata = dict(artist = 'Nicholas Saunders'), bitrate = 1800)
ani.save('motion5.mp4', writer=writer)
pl.show()
