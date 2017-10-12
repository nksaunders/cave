import numpy as np
import sputter as sp
import matplotlib.pyplot as pl
import everest as ev
import csv
from numpy import genfromtxt
from everest import detrender
from everest.math import SavGol, Scatter, Downbin
from astropy.stats import median_absolute_deviation as mad
from numpy.polynomial import polynomial as P
from tqdm import tqdm
import everest

datapath = '/Users/nks1994/Documents/Research/everest/docs/c0'
dataloc = '.csv'

# CAMPAIGN 1

tags = []
mags = []
with open((datapath+str(1)+dataloc),'r') as f:
    data = csv.reader(f)
    for i,row in enumerate(data):
        if i == 0:
            continue
        else:
            tags.append(row[0])
            mags.append(row[1])

A = []

for t in tqdm(tags):
    t = int(t)
    star = everest.Everest(t)
    flux = star.apply_mask(star.flux)
    sgflux = SavGol(flux)
    A.append(np.mean(sgflux))


np.savez('A2.npz', A = A, mags = mags)

import pdb; pdb.set_trace()
