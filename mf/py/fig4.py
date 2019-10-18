import PlotOsiHist as poh
import os
import numpy as np

if __name__ == '__main__':
    kappa = 0
    mExtZero = 0.075
    mExtOne = 0.01
    N = 10000 # N = 1000000
    ##
    figFolder = './figs/fig4/'
    os.system('mkdir -p ' + figFolder)

    # kappa vs mE_1
    poh.BifurcationDiag(figFolder=figFolder)

    
