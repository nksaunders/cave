import aperturefit as af
import numpy as np
import matplotlib.pyplot as pl
import psffit as pf
import simulateK2
from datetime import datetime


class PSFrun(object):

    def __init__(self):

        # self.ID = input("Enter EPIC ID: ")
        self.ID = 205998445
        self.startTime = datetime.now()

        sK2 = simulateK2.Target(int(self.ID), 355000.0)
        trn = sK2.Transit()
        self.fpix,target, self.ferr = sK2.GeneratePSF()

        t = af.ApertureFit(self.fpix,target,self.ferr,trn)
        c_pix, c_det = t.Crowding()


        self.xpos = sK2.xpos
        self.ypos = sK2.ypos
        self.fit = pf.PSFFit(self.fpix,self.ferr)

    def FindFit(self):

        amp = [345000.0,(352000.0 / 2)]
        x0 = [2.6,3.7]
        y0 = [2.3,4.2]
        sx = [.4]
        sy = [.6]
        rho = [0.01]
        background = [1000]
        index = 200
        guess = np.concatenate([amp,x0,y0,sx,sy,rho,background])
        answer = self.fit.FindSolution(guess, index=index)
        invariant_vals = np.zeros((len(answer)))
        self.n_fpix = []


        for i in range(len(answer)):
            if i == 0:
                if i == 0:
                    invariant_vals[i] = 0
                elif i == 2:
                    invariant_vals[i] = answer[i] - self.xpos[index]
                elif i == 4:
                    invariant_vals[i] = answer[i] - self.ypos[index]
                else:
                    invariant_vals[i] = answer[i]

        for cadence in range(10):
            n_vals = np.zeros((len(invariant_vals)))
            for i in range(len(answer)):
                if i == 2:
                    n_vals[i] = invariant_vals[i] + self.xpos[cadence]
                elif i == 4:
                    n_vals[i] = invariant_vals[i] + self.ypos[cadence]
                else:
                    n_vals[i] = invariant_vals[i]

            n_fit = self.fit.PSF(n_vals)
            self.n_fpix.append(n_fit)

        self.subtracted_fpix = [self.fpix[n] - self.n_fpix[n] for n in range(10)]

        self.answerfit = self.fit.PSF(answer)
        self.neighborfit = self.fit.PSF(invariant_vals)
        self.subtraction = self.answerfit - self.neighborfit
        self.residual = self.fpix[200] - self.answerfit

    def Plot(self):

        fig, ax = pl.subplots(1,3, sharey=True)
        fig.set_size_inches(17,5)

        meanfpix = np.mean(self.fpix,axis=0)
        ax[0].imshow(self.fpix[200],interpolation='nearest',origin='lower',cmap='viridis',vmin=np.min(self.answerfit),vmax=np.max(self.answerfit));
        ax[1].imshow(self.answerfit,interpolation='nearest',origin='lower',cmap='viridis',vmin=np.min(self.answerfit),vmax=np.max(self.answerfit));
        ax[2].imshow(self.subtraction,interpolation='nearest',origin='lower',cmap='viridis',vmin=np.min(self.answerfit),vmax=np.max(self.answerfit));
        ax[0].set_title('Data');
        ax[1].set_title('Model');
        ax[2].set_title('Neighbor Subtraction');

        # pl.imshow(self.answerfit-self.fpix[200],interpolation='nearest',origin='lower',cmap='viridis');pl.colorbar();

        fig = pl.figure()
        pl.imshow(self.residual,interpolation='nearest',origin='lower',cmap='viridis'); pl.colorbar()
        pl.title("Residuals")
        print(datetime.now() - self.startTime)
        pl.show()
        import pdb; pdb.set_trace()

r = PSFrun()
r.FindFit()
r.Plot()
