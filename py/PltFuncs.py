basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import code, sys, os
import ipdb
# import pylab as plt
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
sys.path.append(basefolder)
import Keyboard as kb
from multiprocessing import Pool
from functools import partial 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
sys.path.append(basefolder + "/nda/spkStats")
sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig
from Print2Pdf import Print2Pdf
import warnings
#import GetPO
from scipy.optimize import curve_fit

cFF = 0.1

plt.ioff()

def GetBaseFolder(mExt=0.075, mExtOne=0, trNo=0, T=1000, N=10000, K=1000, kappa=0):
    rootFolder = ''
    if kappa*10 == int(kappa*10):
        baseFldr = '../binary/data/N%sK%s/m0%s/mExtOne%s/kappa%s/T%s/tr%s/'%(N, K, int(1e3 * mExt), int(1e3 * mExtOne), int(10 * kappa), int(T*1e-3), trNo)
        if cFF == 1:
            baseFldr = '../binary/data/cFF1/N%sK%s/m0%s/mExtOne%s/kappa%s/T%s/tr%s/'%(N, K, int(1e3 * mExt), int(1e3 * mExtOne), int(10 * kappa), int(T*1e-3), trNo)
        if cFF == 0.25:
            baseFldr = '../binary/data/cff250/N%sK%s/m0%s/mExtOne%s/kappa%s/T%s/tr%s/'%(N, K, int(1e3 * mExt), int(1e3 * mExtOne), int(10 * kappa), int(T*1e-3), trNo)
        if cFF == 0.5:
            baseFldr = '../binary/data/cff500/N%sK%s/m0%s/mExtOne%s/kappa%s/T%s/tr%s/'%(N, K, int(1e3 * mExt), int(1e3 * mExtOne), int(10 * kappa), int(T*1e-3), trNo)
            
    else:
        baseFldr = '../binary/data/N%sK%s/m0%s/mExtOne%s/kappa%s/T%s/tr%s/'%(N, K, int(1e3 * mExt), int(1e3 * mExtOne), int(1000 * kappa), int(T*1e-3), trNo)    
    
    return baseFldr


def LoadFr(phi, mExt, mExtOne, trNo=0, T=1000, N=10000, K=1000, IF_VERBOSE=False, kappa=0):
    baseFldr = GetBaseFolder(mExt, mExtOne, trNo, T, N, K, kappa)
    filename = 'meanrates_theta%.6f_tr%s.txt'%(phi, trNo)
    if IF_VERBOSE:
    	print baseFldr
	print filename
    return np.loadtxt(baseFldr + filename)
    

def KappaVsM1AtPhi(kappaList, phi=0, mExt=0.075, mExtOne=0, N=10000, K=1000, T=1000, trNo=0, IF_PO_SORTED=False, sortedIdx=[], minRate=0):
    m1E = np.empty((len(kappaList, )))
    mrMean = np.empty((len(kappaList, )))
    validKappa = np.empty((len(kappaList, )))
    m1E[:] = np.nan
    validKappa[:] = np.nan
    mrMean[:] = np.nan
    for kIdx, kappa in enumerate(kappaList):
        p=kappa
	try:
            IF_VERBOSE = False # True
            mr = LoadFr(phi, mExt, mExtOne, trNo, T, N, K, IF_VERBOSE=IF_VERBOSE, kappa=kappa)
            mrMean[kIdx] = np.mean(mr[:N])
	    if not IF_PO_SORTED:
		m1E[kIdx] = M1Component(mr[:N])
	    else:
		mre = mr[:N]
                mask = mr[sortedIdx] > minRate
                m1E[kIdx] = M1Component(mr[sortedIdx[mask]])		
	    validKappa[kIdx] = kappa
	    print 'o',
	except IOError:
	    print 'x', 
	    #pass
	    #print 'kappa: ', kappa, ' no files!'
    sys.stdout.flush()	    
    return validKappa, m1E, mrMean


def KappaVsM1AtTr(kappaList, nPhis = 8, mExt=0.075, mExtOne=0.075, N=10000, K=1000, T=1000, trNo=0, IF_PO_SORTED = False, sortedIdx = [], minRate = 0):
    thetas = np.linspace(0, 180, nPhis, endpoint = False)
    m1E = np.zeros((nPhis, len(kappaList)))
    m0E = np.zeros((nPhis, len(kappaList)))    
    vldKappa = np.empty((nPhis, len(kappaList)))
    vldKappa[:] = np.nan
    for i, phi in enumerate(thetas):
	print 'phi =', phi
	vldKappa[i, :], m1E[i, :], m0E[i, :] = KappaVsM1AtPhi(kappaList, phi, mExt, mExtOne, N, K, T, trNo, IF_PO_SORTED = IF_PO_SORTED, sortedIdx = sortedIdx, minRate = minRate)
    # print 'inside tr func', np.nanmean(m1E, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(m1E, 0), np.nanmean(m0E, 0) #, vldKappa
    # except RuntimeWarning:
    #     return np.nan, np.nan

def KappaVsM1(kappaList, nTrials = 10, nPhis = 8, mExt=0.075, mExtOne=0.075, N=10000, K=1000, T=1000, IF_PO_SORTED = False, sortedIdx = [], minRate = 0, pcolor = 'k', IF_COMPUTE=False):
    if IF_COMPUTE:
        m1E = np.empty((nTrials, len(kappaList)))
        m1E[:] = np.nan
        m0E = np.empty((nTrials, len(kappaList)))
        m0E[:] = np.nan
        for trNo in np.arange(nTrials): #$ trNo 0 is always the CONTROL 
            print ''
            print '--' * 27
            print 'tr#: ', trNo
            print '--' * 27
            tmp1, tmp0 = KappaVsM1AtTr(kappaList, nPhis, mExt, mExtOne, N, K, T, trNo, IF_PO_SORTED, sortedIdx, minRate = minRate)
            m1E[trNo, :] = tmp1
            m0E[trNo, :] = tmp0
            # print tmp0
        print ''

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m1EvsKappa = np.nanmean(m1E, 0)
            m1EvsKappaSTD = np.nanstd(m1E, 0)
            meanRateE = np.nanmean(m0E, 0)

        validIdx = ~np.isnan(m1EvsKappa)
        numValTrials = np.sum(~np.isnan(m1E), 0)
        # remove kappas with no trials
        kappaList = kappaList[validIdx]
        m1EvsKappa = m1EvsKappa[validIdx]
        m1EvsKappaSTD = m1EvsKappaSTD[validIdx]
        numValTrials = numValTrials[validIdx]
        m1EvsKappaSEM = np.empty((len(kappaList), ));
        m1EvsKappaSEM = m1EvsKappaSTD / numValTrials
        # m1EvsKappaSEM = m1EvsKappaSEM[validIdx]
        print 'm1E', m1E.shape
        out = {}
        out['kappa'] = kappaList
        out['m1EvsKappa'] = m1EvsKappa
        out['m1EvsKappaSEM'] = m1EvsKappaSEM
        out['meanRateE'] = meanRateE[validIdx]        
        # out['muE'] = 
        out['tr_muE'] = np.squeeze(m1E / m0E)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if out['tr_muE'].ndim > 1:
                out['mean_muE'] = np.nanmean(out['tr_muE'], 0)[validIdx]
                out['sem_muE'] = np.nanstd(out['tr_muE'], 0)[validIdx] / np.sqrt(numValTrials)
            else:
                out['mean_muE'] = np.nanmean(out['tr_muE'], 0)
                out['sem_muE'] = np.nanstd(out['tr_muE'], 0) / np.sqrt(numValTrials)
                

        out['nTrials'] = numValTrials
        

        print type(out), len(out)

        np.save('./data/kappa_vs_mE1_SIM_mZero%s_mOne%s_K%s_N%s'%(mExt, mExtOne, K, N), out)
        # ipdb.set_trace()
    else:  # plot
        if N == 10000:
            out = np.load('./data/kappa_vs_mE1_SIM_mZero%s_mOne%s_K%s_N%s.npy'%(mExt, mExtOne, K, N)).item()
            out = np.load('./data/kappa_vs_mE1_SIM_mZero%s_mOne%s_K%s.npy'%(mExt, mExtOne, K)).item()            
        else:
            out = np.load('./data/kappa_vs_mE1_SIM_mZero%s_mOne%s_K%s.npy'%(mExt, mExtOne)).item()
            # out = np.load('./PNAS/data/kappa_vs_mE1_SIM_mZero%s_mOne%s_K%s_N%s.npy'%(mExt, mExtOne, K, N)).item()            
        # mE0 = out['meanRateE']
        # ipdb.set_trace()
        kappaList = out['kappa']
        m1EvsKappa = out['m1EvsKappa']
        m1EvsKappaSEM = out['m1EvsKappaSEM']
        
        # print type(out), len(out)
        # for i in range(len(kappaList)):
        # (_, caps, _) = plt.errorbar(kappaList, m1EvsKappa, fmt = 'o-', markersize = 3, yerr = m1EvsKappaSEM, lw = 0.8, elinewidth=0.8, label = r'$N = %s, K = %s$'%(N, K))
            # (_, caps, _) = plt.errorbar(kappaList, m1EvsKappa / 0.17126, fmt = 'o', markersize = 1, yerr = m1EvsKappaSEM, lw = 0.6, elinewidth=0.2, label = r'$N = %s$'%(N, ), markeredgecolor = pcolor, color = pcolor)

        # muE = 
            # plt.plot(kappaList, m1EvsKappa / 0.17126, 'ok')
            
    return m1EvsKappa, m1EvsKappaSEM

def KappaVsM1Plot(Nlist, kappaList=np.arange(0, 8, 0.1),T=1000, K=1000, mExt=0.075):
    # N = 10000
    # mExt = 0.075
    IF_PLOT = True; #int(sys.argv[1])
    if not IF_PLOT:
        mExtOne = float(sys.argv[1])
    else:
        mExtOneList = [0] #, 0.0375, 0.075]
        # Nlist = [10000, 40000, 80000]
        # mExt = 0.075
        # theory = np.load('/homecentral/srao/Documents/code/binary/c/analysis/data/PRX/mE1_m075_K1000.npy')
        # theoryIdx = theory[0, :] <= 8
        # plt.plot(theory[0, theoryIdx], theory[1, theoryIdx] / theory[2, theoryIdx], 'k-', label = 'Theory')
        y = np.load('./data/mE1_m0{}_K{}_cff_{}.npy'.format(int(1e3*mExt), int(K), int(cFF*1e3))).item()
        plt.plot(y['kappa'], y['muE'], 'k')
        
        pcolor = ['k', 'g', 'r']
        for idx, N in enumerate(Nlist):
            for mExtOne in mExtOneList:
                if N == 10000 or N == 20000:
                    out = np.load('./data/kappa_vs_mE1_SIM_mZero%s_mOne%s_K%s_N%s.npy'%(mExt, mExtOne, K, N))
                elif N == 40000:
                    out = np.load('./data/kappa_vs_mE1_SIM_mZero%s_mOne%s_K%s_N%s.npy'%(mExt, mExtOne, K, N))
                elif N == 80000:
                    out = np.load('./data/kappa_vs_mE1_SIM_mZero%s_mOne%s_K%s_N%s.npy'%(mExt, mExtOne, K, N))
                # ipdb.set_trace()
                out = out.item()

                
                kappa = out['kappa']
                mean_muE = out['mean_muE']
                if (mean_muE.size == 1) and (len(mean_muE.shape) != mean_muE.size):
                    mean_muE = np.array([mean_muE])
                    
                sem_muE = out['sem_muE']
                nValidTrials = out['nTrials']
                # kappaIdx = kappa <= 15
                # # kappaIdx = kappa != 4.2
                # # if N==10000:
                # #     kappaIdx = kappa != 4
                # kappa = kappa[kappaIdx]
                
                # m1EvsKappa = out['m1EvsKappa'][kappaIdx]
                # m1EvsKappaSEM = out['m1EvsKappaSEM'][kappaIdx]
                # meanRateE = out['meanRateE'][kappaIdx]
                # validFinalIdx = m1EvsKappa <= meanRateE
                # kappa = kappa[validFinalIdx]
                # m1EvsKappa = m1EvsKappa[validFinalIdx]
                # m1EvsKappaSEM = m1EvsKappaSEM[validFinalIdx]
                # meanRateE = meanRateE[validFinalIdx]
                # muE =  m1EvsKappa / meanRateE


                # ipdb.set_trace()
                
                print N, pcolor[idx]

                # (_, caps, _) = plt.errorbar(kappa, muE, fmt = 'o', markersize = 1, yerr = m1EvsKappaSEM, lw = 0.4, elinewidth=0.2, label = r'N=%s'%(N), markeredgecolor = pcolor[idx], color = pcolor[idx])

                # ipdb.set_trace()
        
                # ipdb.set_trace()                
                if kappa.size > 0:
                    # (_, caps, _) = plt.errorbar(kappa, mean_muE, fmt = 'o', markersize = 2, yerr = sem_muE, lw = 0.4, elinewidth=0.2, label = r'N=%s, T=%s'%(N, T), markeredgecolor = 'w', markeredgewidth=0.1)
                    (_, caps, _) = plt.errorbar(kappa, mean_muE, fmt = 'o', markersize = 2, yerr = sem_muE, lw = 0.4, elinewidth=0.2, label = r'N=%s, T=%s, ntr=%s'%(N, T, nValidTrials[0]), markeredgecolor = 'w', markeredgewidth=0.1)
                    for cap in caps:
                        cap.set_markeredgewidth(0.2)



                print 'nTrials', nValidTrials

                # ipdb.set_trace()

                # if kappa.size > 0:
                #     plt.plot(kappa, muE, 'o', label = r'N=%s, T=%s'%(N, T), markeredgecolor='w', markersize=2)
                
        plt.xlabel(r'$\kappa$')
        plt.ylabel(r'$\mu_E$')
        plt.xlim([0, 10])
        plt.gca().set_xticks([0, 5, 10])
        plt.gca().set_yticks([0, 0.5, 1])
        plt.ylim([0, 1])
        xx = np.linspace(0, 4, 100)
        plt.plot(xx, np.zeros(xx.shape), 'k')
        plt.legend(loc = 2, frameon = False, numpoints = 1, ncol = 1, prop = {'size': 4})
        figFolder = './figs/'
        # plt.title('c =  %s, T = %s'%(cFF, T))
        plt.title('c =  %s'%(cFF))        
        if cFF == 1:
            figname = 'cff1_kappa_vs_mE1_SIM_mZero%s_mOne%s_K%s_N%s'%(mExt, mExtOne, K, N)
        else:
            figname = 'kappa_vs_mE1_SIM_mZero%s_mOne%s_K%s_N%s_T%s'%(mExt, mExtOne, K, N, T)
        paperSize = [2.25 * 1.22, 2.25]
        figFormat = 'svg'
        axPosition =  [0.28, 0.25, .65, .65]
        print figFolder, figname
        plt.ion()
        figFormat='png'
        ProcessFigure(plt.gcf(), figFolder+figname, True, IF_XTICK_INT = False, figFormat = figFormat, paperSize = paperSize, titleSize = 10, axPosition=axPosition, tickFontsize = 8, labelFontsize = 10, nDecimalsX = 1, nDecimalsY = 2)
        # plt.show()
        # plt.ion()
        #ipdb.set_trace()

def KappaVsM1PlotPoints(Nlist, ntr=10, kappaList=np.arange(0, 8, 0.1),T=1000):
    N = 10000
    K = 1000
    mExt = 0.075
    IF_PLOT = True; #int(sys.argv[1])
    if not IF_PLOT:
        mExtOne = float(sys.argv[1])
    else:
        mExtOneList = [0] #, 0.0375, 0.075]
        # Nlist = [10000, 40000, 80000]
        K = 1000
        mExt = 0.075
        # theory = np.load('/homecentral/srao/Documents/code/binary/c/analysis/data/PRX/mE1_m075_K1000.npy')
        # theoryIdx = theory[0, :] <= 8
        # plt.plot(theory[0, theoryIdx], theory[1, theoryIdx] / theory[2, theoryIdx], 'k-', label = 'Theory')
        y = np.load('./data/mE1_m0{}_K1000_cff_{}.npy'.format(int(1e3*mExt), int(cFF*1e3))).item()
        plt.plot(y['kappa'], y['muE'], 'k')
        
        pcolor = ['k', 'g', 'r']
        for idx, N in enumerate(Nlist):
            for mExtOne in mExtOneList:
                if N == 10000 or N == 20000:
                    out = np.load('./data/kappa_vs_mE1_SIM_mZero%s_mOne%s_K%s_N%s.npy'%(mExt, mExtOne, K, N))
                elif N == 40000:
                    out = np.load('./data/kappa_vs_mE1_SIM_mZero%s_mOne%s_K%s_N%s.npy'%(mExt, mExtOne, K, N))
                elif N == 80000:
                    out = np.load('./data/kappa_vs_mE1_SIM_mZero%s_mOne%s_K%s_N%s.npy'%(mExt, mExtOne, K, N))
                # ipdb.set_trace()
                out = out.item()
                # ipdb.set_trace()
                kappa = out['kappa']
                kappaIdx = kappa <= 15
                # kappaIdx = kappa != 4.2
                # if N==10000:
                #     kappaIdx = kappa != 4
                kappa = kappa[kappaIdx]
                m1EvsKappa = out['m1EvsKappa'][kappaIdx]
                m1EvsKappaSEM = out['m1EvsKappaSEM'][kappaIdx]
                meanRateE = out['meanRateE'][kappaIdx]
                validFinalIdx = m1EvsKappa <= meanRateE
                kappa = kappa[validFinalIdx]
                m1EvsKappa = m1EvsKappa[validFinalIdx]
                m1EvsKappaSEM = m1EvsKappaSEM[validFinalIdx]
                meanRateE = meanRateE[validFinalIdx]
                muE =  m1EvsKappa / meanRateE
                print N, pcolor[idx]

                # (_, caps, _) = plt.errorbar(kappa, muE, fmt = 'o', markersize = 1, yerr = m1EvsKappaSEM, lw = 0.4, elinewidth=0.2, label = r'N=%s'%(N), markeredgecolor = pcolor[idx], color = pcolor[idx])
                # (_, caps, _) = plt.errorbar(kappa, muE, fmt = 'o', markersize = 2, yerr = m1EvsKappaSEM, lw = 0.4, elinewidth=0.2, label = r'N=%s, T=%s'%(N, T), markeredgecolor = 'w', markeredgewidth=0.1)

                tr_muE = np.squeeze(out['tr_muE'])
                atk = np.ones((out['nTrials'], )) * kappa[0]
                plt.plot(atk, tr_muE, 'wo', label = r'N=%s, T=%s'%(N, T), markersize=2, markeredgecolor='k')


                # for trNo in range(ntr):
                #     muE = 

                    
                # if kappa.size > 0:
                #     plt.plot(kappa, muE, 'o', label = r'N=%s, T=%s'%(N, T), markeredgecolor='w', markersize=2)
                
                # for cap in caps:
                #     cap.set_markeredgewidth(0.2)

        plt.xlabel(r'$\kappa$')
        plt.ylabel(r'$\mu_E$')
        plt.xlim([0, 10])
        plt.gca().set_xticks([0, 5, 10])
        plt.gca().set_yticks([0, 0.5, 1])
        plt.ylim([0, 1])
        xx = np.linspace(0, 4, 100)
        plt.plot(xx, np.zeros(xx.shape), 'k')
        plt.legend(loc = 2, frameon = False, numpoints = 1, ncol = 1, prop = {'size': 6})
        figFolder = './figs/'
        # plt.title('c =  %s, T = %s'%(cFF, T))
        plt.title('c =  %s'%(cFF))        
        if cFF == 1:
            figname = 'cff1_kappa_vs_mE1_SIM_mZero%s_mOne%s_K%s_N%s'%(mExt, mExtOne, K, N)
        else:
            figname = 'trp_kappa_vs_mE1_SIM_mZero%s_mOne%s_K%s_N%s_T%s'%(mExt, mExtOne, K, N, T)
        paperSize = [2.25 * 1.22, 2.25]
        figFormat = 'svg'
        axPosition =  [0.28, 0.25, .65, .65]
        print figFolder, figname
        plt.ion()
        figFormat='png'
        ProcessFigure(plt.gcf(), figFolder+figname, True, IF_XTICK_INT = False, figFormat = figFormat, paperSize = paperSize, titleSize = 10, axPosition=axPosition, tickFontsize = 8, labelFontsize = 10, nDecimalsX = 1, nDecimalsY = 2)
        # plt.show()
        # plt.ion()
        #ipdb.set_trace()



def M1Component(x):
    out = np.nan
    if len(x) > 0:
	dPhi = np.pi / len(x)
	out = 2.0 * np.absolute(np.dot(x, np.exp(-2.0j * np.arange(len(x)) * dPhi))) / len(x)
    return out
    

def ProcessFigure(figHdl, filepath, IF_SAVE, IF_XTICK_INT = False, figFormat = 'eps', paperSize = [4, 3], titleSize = 10, axPosition = [0.25, 0.25, .65, .65], tickFontsize = 10, labelFontsize = 12, nDecimalsX = 1, nDecimalsY = 1):
    FixAxisLimits2(figHdl, IF_XTICK_INT, nDecimalsX, nDecimalsY)
    Print2Pdf(figHdl, filepath, paperSize, figFormat=figFormat, labelFontsize = labelFontsize, tickFontsize=tickFontsize, titleSize = titleSize, IF_ADJUST_POSITION = True, axPosition = axPosition)
    plt.show()

def FixAxisLimits2(fig, IF_XTICK_INT = False, nDecimalsX = 1, nDecimalsY = 1):
    ax = fig.axes[0]
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.' + '%s'%(int(nDecimalsX)) + 'f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.' + '%s'%(int(nDecimalsY)) + 'f'))
    xmiddle = 0.5 * (xmin + xmax)
    xticks = [xmin, xmiddle, xmax]
    if IF_XTICK_INT:
	if xmiddle != int(xmiddle):
	    xticks = [xmin, xmax]
	ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_xticks(xticks)
    ax.set_yticks([ymin, 0.5 *(ymin + ymax), ymax])
    plt.draw()
    
