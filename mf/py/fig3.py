import PlotOsiHist as poh
import os
import numpy as np

if __name__ == '__main__':
    kappa = 0
    mExtZero = 0.075
    mExtOne = 0.03
    N = 10000 # N = 1000000
    ##
    figFolder = './figs/fig3/'
    os.system('mkdir -p ' + figFolder)

    # beta vs OSI
    jii_beta_list = np.arange(1, 4.5, 0.5)
    poh.PltJiibetavsosi(jii_beta_list, mOne=mExtOne, nGenerated=N, figFolder=figFolder)

    # beta vs pop activity
    poh.PltJiibetavsRate(jii_beta_list, mZero=mExtZero, mOne=mExtOne, figFolder=figFolder)
