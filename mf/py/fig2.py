import PlotOsiHist as poh
import os, ipdb

if __name__ == '__main__':
    kappa = 0
    mExtZero = 0.075
    mExtOne = 0.03
    N = 1000000
    ##
    figFolder = './figs/fig2/'
    os.system('mkdir -p ' + figFolder)

    # OSI hist
    # poh.PltHist(mOne=mExtOne, nGenerated=N, figFolder=figFolder)



    # POP Avg Tuning Curves
    # poh.PltPopAvgTuning(mOne=mExtOne, nGenerated=N, figFolder=figFolder); raise SystemExit;

    # Tuning curves
    # n=1000
    # poh.PlotNTc(mOne=mExtOne, nGenerated=1000000, n=n, figFolder=figFolder)

    # CCC (cff)
#     N = 5000
#     cff_list = [0.25, 0.5, 0.75, 1]
# #    poh.CCC_vs_cff(compute=1, cff_list=cff_list, mExtOne=mExtOne, kappa=kappa, figFolder=figFolder)
#     poh.CCC_vs_cff(compute=0, cff_list=cff_list, mExtOne=mExtOne, kappa=kappa, figFolder=figFolder)

    # # CCC (mExtOne)
    # poh.CCCvsM1(cff_list=cff_list, mExtOne=mExtOne)

    # PO ff vs PO output Scatter
    poh.POffvsPOoutput(cff_list = [0.1], mZero=mExtZero, mExtOne=mExtOne, nPoints=5000, figFolder=figFolder, kappa=kappa, nGenerated=N)    
