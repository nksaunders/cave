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
sK2 = simulateK2.Target(int(ID),npts = 100)
trn = sK2.Transit()

fpix, target, ferr = sK2.GeneratePSF(12,motion_mag = 5)

t = np.linspace(0,90,len(fpix))

def images(i):
    fpix_im = pl.imshow(fpix[i],cmap='viridis',origin='lower',interpolation='nearest')
    pl.axis('off')
    return fpix_im

fig = pl.figure()
ani = animation.FuncAnimation(fig, images, frames=100, interval = 1, repeat=True)
Writer = animation.writers['ffmpeg']
writer = Writer(fps = 15, metadata = dict(artist = 'Nicholas Saunders'), bitrate = 1800)
ani.save('motiontest.mp4', writer=writer)
pl.show()
