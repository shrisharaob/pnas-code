import PlotOsiHist as poh
import os
import numpy as np

if __name__ == '__main__':
    kappa = 0
    mExtZero = 0.075
    mExtOne = 0.03
    N = 10000 # N = 1000000
    ##
    figFolder = './figs/fig5/'
    os.system('mkdir -p ' + figFolder)

    # kappa vs OSI
    # poh.Kappa_vs_OSI(mOne=mExtOne, nGenerated=N, figFolder=figFolder)

    N = 1000000
    kappaList = [0, 9, 18]
    poh.PltHistKappaList(mOne=mExtOne, nGenerated=N, figFolder=figFolder, kappaList=kappaList)
