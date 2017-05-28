import matplotlib.pyplot as pl
import matplotlib.animation as animation
import numpy as np
from everest import Transit
import k2plr
from k2plr.config import KPLR_ROOT
from intrapix import PixelFlux
from everest.config import KEPPRF_DIR
from astropy.io import fits
from scipy.special import erf
import pyfits
import os

fpix = np.zeros((1000,5,5))

t = np.linspace(0,90,len(fpix))
per = 15
dur = 0.5
depth = 0.01
trn = Transit(t, t0 = 5.0, per = per, dur = dur, depth = depth)

ID = 205998445
client = k2plr.API()
star = client.k2_star(ID)
tpf = star.get_target_pixel_files(fetch = True)[0]
ftpf = os.path.join(KPLR_ROOT, 'data', 'k2', 'target_pixel_files', '%d' % ID, tpf._filename)
with pyfits.open(ftpf) as f:
    xpos = f[1].data['pos_corr1']
    ypos = f[1].data['pos_corr2']

# throw out outliers and limit to first 1000 cadences

for i in range(len(xpos)):
    if abs(xpos[i]) >= 50 or abs(ypos[i]) >= 50:
        xpos[i] = 0
        ypos[i] = 0
    if np.isnan(xpos[i]):
        xpos[i] = 0
    if np.isnan(ypos[i]):
        ypos[i] = 0
    '''
    else:
        xpos[i] /= 5
        ypos[i] /= 5

    '''

xpos = xpos[:1000]
ypos = ypos[:1000]

# define intra-pixel sensitivity variation

intra = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        intra[i][j] = (0.995 + np.random.randint(10) / 1000)

# generate PSF model with inter-pixel sensitivity variation

cx = [1.0,0.0,0.0]
cy = [1.0,0.0,0.0]
amp = [355000.0]
x0 = 2.5
y0 = 2.5
sx = [.5]
sy = [.5]
rho = [0]

magdiff = 1.0
r = 10**(magdiff / 2.5)

def NoisePRF(x0,sigma,c):
    return (1/40) * sigma * np.e**(-(x0**2+c**2)/(2*sigma**2))*(np.sqrt(2*np.pi) \
    *(4*x0**2-4*x0*(2*c+1)+4*sigma**2+4*c**2+4*c-19)*np.e**((x0**2+c**2)/(2*sigma**2)) \
    *erf((c-x0)/(np.sqrt(2)*sigma))-8*sigma*(x0-c-1)*np.e**(x0*c/(sigma**2)))-(1/40) \
    *sigma*np.e**(-(x0**2+(c+1)**2)/(2*sigma**2))*(np.sqrt(2*np.pi)*(4*x0**2-4*x0*(2*c+1) \
    +4*sigma**2+4*c**2+4*c-19)*np.e**((x0**2+(c+1)**2)/(2*sigma**2))*erf((c-x0+1)/ \
    (np.sqrt(2)*sigma))-8*sigma*(x0-c)*np.e**(x0*(c+1)/(sigma**2)))

def perfectPRF(a,b,c):
    return np.sqrt(np.pi/2)*b*(erf(a/(np.sqrt(2)*b)) - erf((a-c-1)/(np.sqrt(2)*b)))

target = np.zeros((len(fpix),5,5))
ferr = np.zeros((len(fpix),5,5))
background_level = 800

for c in range(1000):
    for i in range(5):
        for j in range(5):

            # contribution from target
            target_val = amp[0]*NoisePRF(2.5+xpos[c],0.5,i)*NoisePRF(2.5+ypos[c],0.5,j)

            # contribution from neighbor
            val = target_val# + (1/r)*PixelFlux(cx,cy,amp,[4.-i+xpos[c]],[4.-j+ypos[c]],sx,sy,rho)

            target[c][i][j] = target_val
            fpix[c][i][j] = val

            # add photon noise
            ferr[c][i][j] = np.sqrt(fpix[c][i][j])
            randnum = np.random.randn()
            fpix[c][i][j] += ferr[c][i][j] * randnum
            target[c][i][j] += ferr[c][i][j] * randnum

            # add background noise
            noise = np.sqrt(background_level) * np.random.randn()
            fpix[c][i][j] += background_level + noise
            target[c][i][j] += background_level + noise

    # multiply by intra-pixel variation
    # fpix[c] *= intra
    target[c] *= intra

def images(i):
    fpix_im = pl.imshow(fpix[i],cmap='viridis',origin='lower',interpolation='nearest')
    return fpix_im

fig = pl.figure()
ani = animation.FuncAnimation(fig, images, frames=150, interval = 1, repeat=True)
Writer = animation.writers['ffmpeg']
writer = Writer(fps = 15, metadata = dict(artist = 'Nicholas Saunders'), bitrate = 1800)
ani.save('k2motion_near.mp4', writer=writer)
# pl.show()
