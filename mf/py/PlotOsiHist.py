import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import scipy.io as sio
import os
import ipdb
import sys
#basefolder = "/homecentral/srao/Documents/code/mypybox"
# sys.path.append(basefolder)
# sys.path.append(basefolder + "/utils")
# #from Print2Pdf import Print2Pdf
# sys.path.append("/homecentral/srao/Documents/code/binary/prx/analysis/")
import Scripts as sr
import matplotlib as mpl
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'cm'

def Kappa_vs_OSI(mZero=0.075, mOne=0.015, kappa=0, IF_NEW_FIG=True, nGenerated=1000000, figFolder=''):
    #
    filename = '../data/kappa_vs_mZero_%s_mOne_%s_N_%s'%(int(1e3*mZero), int(1e6*mOne), nGenerated)

    out = sio.loadmat(filename)
    kappa = np.squeeze(out['kappas'])
    kappa_critical = out['kappa_c'][0][0]
    osi_E = np.squeeze(out['osi_E'])
    osi_I = np.squeeze(out['osi_I'])
    #
    vidx = kappa < 20
    plt.plot(kappa[vidx], osi_E[vidx], 'k', lw=0.5)
    plt.plot(kappa[vidx], osi_I[vidx], 'r', lw=0.5)
    #
    plt.ylim(0, 1)
    plt.xlim(0, 20)
    plt.xlabel(r'$\kappa$')        
    plt.ion()
    plt.draw()
    plt.vlines(kappa_critical, 0, 1, color = 'k', linestyles='--', lw=0.5)
    # plt.gca().set_xticks([0, 3.5, 7, kappa_critical])
    # plt.gca().set_xticklabels(['0', '3.5', '7', r'$\kappa_c$'])
    plt.gca().set_xticks([0, 10, 20, kappa_critical])
    plt.gca().set_xticklabels(['0', '10', '20', r'$\kappa_c$'])

    # ymin, ymax = plt.gca().get_ylim()
    plt.ylim(0, 1)
    plt.gca().set_yticks([0, 0.5, 1])
    plt.gca().set_yticklabels(['0', '0.5', '1'])
    ## ## ## ## ## ## ## ## ## ## ## ## ##
    if figFolder == '':
        figFolde = './figs/'
    figname = 'kappa_vs_osi'
    paperSize = [2.5, 2]
    axPosition=[.22, .22, .65, .65]
    figFormat = 'svg'
    print figFolder, figname
    plt.draw()    
    sr.ProcessFigure(plt.gcf(), figFolder+figname, True, IF_XTICK_INT = False, figFormat = figFormat, paperSize = paperSize, titleSize = 10, axPosition=axPosition, tickFontsize = 8, labelFontsize = 10, nDecimalsX = 1, nDecimalsY = 2)
    plt.show()


def BifurcationDiag(mZero=0.075, mOneList=np.array([0, 1./4, 1./2, 1]) * 0.075, figFolder=''):
    colors = ['k', 'g', 'r', 'b']
    for k, mExtOne in enumerate(mOneList):
        filename = '../data/bifurcation_curve_mZero_%s_mOne_%s.mat'%(int(1e3*mZero), int(1e6*mExtOne))
        out = sio.loadmat(filename)
        kappa = np.squeeze(out['kappa'])
        mu_E = np.squeeze(out['mu_E'])
        kappa_critical = out['kappa_c'][0][0]
        vidx = mu_E < 1
        vidx = kappa < 19
        plt.plot(kappa[vidx], mu_E[vidx], '-', color=colors[k], lw=0.85)
        # ipdb.set_trace()
        
    plt.ylim(0, 1)
    plt.xlim(0, 20)
    plt.xlabel(r'$\kappa$')        
    plt.ion()
    plt.draw()
    # plt.vlines(kappa_critical, 0, 1, color = 'k', linestyles='--')

    plt.gca().set_xticks([0, 10, 20, kappa_critical])
    plt.gca().set_xticklabels(['0', '10', '20', r'$\kappa_c$'])

    # ymin, ymax = plt.gca().get_ylim()
    plt.ylim(0, 1)
    plt.gca().set_yticks([0, 0.5, 1])
    plt.gca().set_yticklabels(['0', '0.5', '1'])
    plt.ylabel(r'$\mu_E$')

    ##
    if figFolder == '':
        figFolde = './figs/'
    figname = 'bifurcation_curve'
    paperSize = [2.5, 2]
    axPosition=[.22, .22, .75, .75]
    figFormat = 'svg'
    print figFolder, figname
    plt.draw()    
    sr.ProcessFigure(plt.gcf(), figFolder+figname, True, IF_XTICK_INT = False, figFormat = figFormat, paperSize = paperSize, titleSize = 10, axPosition=axPosition, tickFontsize = 8, labelFontsize = 10, nDecimalsX = 1, nDecimalsY = 2)


    plt.figure()
    muList = [0, 0.25, 0.5, 1]
    for k, mExtOne in enumerate(mOneList):
        plt.plot(1, 1, color=colors[k], label=r'$\mu=%s$'%(muList[k]))
    plt.legend(frameon=False, loc=10, prop={'size':20})
    plt.savefig(figFolder + './figure_legend.svg')

    
    
#    sr.ProcessFigure(plt.gcf(), figFolder+figname, True, IF_XTICK_INT = False, figFormat = figFormat, paperSize = paperSize, titleSize = 10, axPosition=axPosition, tickFontsize = 6, labelFontsize = 10, nDecimalsX = 1, nDecimalsY = 2)

    


def PltJiibetavsosi(jII_betaList, mZero=0.075, mOne=0.01, kappa=0, nGenerated=10000, figFolder=''):
    xe = []
    xi = []
    for jII_beta in jII_betaList:
        filename =  '../data/OSI/osi_mZero_%s_mOne_%s_kappa_%s_N%s_jII_beta_%s'%(int(1e3 * mZero), int(1e4 * mOne), int(kappa), nGenerated, int(jII_beta * 1e3))

        if os.path.exists(filename):
            os.system('mv ' + filename + ' ' + filename + '.mat')
        out = sio.loadmat(filename)
        osi_E = out['osi_E']
        osi_I = out['osi_I']
        xe.append(np.nanmean(osi_E))
        xi.append(np.nanmean(osi_I))
    plt.plot(jII_betaList, xe, 'ko-', lw=0.5, markersize=0.8)
    plt.plot(jII_betaList, xi, 'ro-', lw=0.5, markersize=0.8, markeredgecolor='r')
    print xe
    print xi
    if figFolder == '':
        figFolde = './figs/'
    figname = 'mean_os_vs_jII_beta'
    paperSize = [2.5, 2]
    axPosition=[.22, .22, .65, .65]
    figFormat = 'svg'
    print figFolder, figname
    plt.ion()
    # sr.FixAxisLimits(plt.gcf(), IF_XTICK_INT = False, nDecimalsX = 1, nDecimalsY = 0)
    plt.gca().set_xticks([1, 4])
    plt.gca().set_xticklabels(['1', '4'])
    # ymin, ymax = plt.gca().get_ylim()
    plt.ylim(0, 1)
    plt.gca().set_yticks([0, 0.5, 1])
    plt.gca().set_yticklabels(['0', '0.5', '1'])
    ##
    # plt.ylim(0, 0.5)
    # plt.gca().set_yticks([0, 0.25, 0.5])
    # plt.gca().set_yticklabels(['0', '0.25', '0.5'])
    ##
    plt.draw()    
    sr.ProcessFigure(plt.gcf(), figFolder+figname, True, IF_XTICK_INT = False, figFormat = figFormat, paperSize = paperSize, titleSize = 10, axPosition=axPosition, tickFontsize = 8, labelFontsize = 10, nDecimalsX = 1, nDecimalsY = 2)
    
def CheckBalConditions(JEE, JEI, JIE, JII, JE0, JI0):
    JE = -JEI/JEE
    JI = -JII/JIE
    E = JE0
    I = JI0
    if((JE < JI) or (E/I < JE/JI) or (JE < 1) or (E/I < 1) or (JE/JE < 1)):
        print "NOT IN BALANCED REGIME!!!!!! "
        raise SystemExit

def PltJiibetavsRate(jII_betaList, mZero=0.075, mOne=0.01, kappa=0, figFolder=''):
    JEE = 1
    JEI = -1.5
    JIE = 1
    JII = -1
    JE0 = 2
    JI0 = 1
    mE0 = []
    mI0 = []
    for jII_beta in jII_betaList:
        Jab = np.array([[JEE, JEI / jII_beta],
                        [JIE, JII / jII_beta]])

        Ea = np.array([JE0, JI0])
        cFF = 1 #0.2

        CheckBalConditions(JEE, JEI, JIE, JII, JE0, JI0)
        gamma = 0.0
        meanFieldRates = -1.0 * np.dot(np.linalg.inv(Jab), Ea) * mZero
        mE0.append(meanFieldRates[0])
        mI0.append(meanFieldRates[1])        
        print 'MF rates = ', meanFieldRates
    ##
    plt.figure()
    plt.plot(jII_betaList, mE0, 'k-', lw=0.5)
    plt.plot(jII_betaList, mI0, 'r-', lw=0.5)
    if figFolder == '':
        figFolde = './figs/'
    figname = 'mean_rate_vs_jII_beta'
    paperSize = [2.5, 2]
    axPosition=[.22, .22, .65, .65]
    figFormat = 'svg'
    print figFolder, figname
    plt.ion()
    # sr.FixAxisLimits(plt.gcf(), IF_XTICK_INT = False, nDecimalsX = 1, nDecimalsY = 0)
    plt.gca().set_xticks([1, 4])
    plt.gca().set_xticklabels(['1', '4'])
    # ymin, ymax = plt.gca().get_ylim()
    plt.ylim(0, 1)
    plt.gca().set_yticks([0, 0.5, 1])
    plt.gca().set_yticklabels(['0', '0.5', '1'])
    plt.draw()    
    sr.ProcessFigure(plt.gcf(), figFolder+figname, True, IF_XTICK_INT = False, figFormat = figFormat, paperSize = paperSize, titleSize = 10, axPosition=axPosition, tickFontsize = 8, labelFontsize = 10, nDecimalsX = 1, nDecimalsY = 2)
    
    

def PltHist(mZero=0.075, mOne=0.015, kappa=0, IF_NEW_FIG=True, nGenerated=1000000, figFolder=''):
    # loadmat
    #    fn = '../data/OSI/osi_kappa_%s_mExtZero_%s_mOne_%s.mat'%(kappa, mZero, mOne)
    #fn = '../data/OSI/osi_mZero%s_mOne_%s_kappa_%s.mat'%(mZero, mOne, kappa)


    filename = '../data/OSI/osi_mZero_%s_mOne_%s_kappa_%s_N%s'%(int(1e3*mZero), int(1e4*mOne), kappa, nGenerated)
    if os.path.exists(filename):
        os.system('mv ' + filename + ' ' + filename + '.mat')
    out = sio.loadmat(filename)
    osi_E = out['osi_E']
    osi_I = out['osi_I']
    print np.sum(np.isnan(osi_E))
    osi_E = osi_E[np.logical_not(np.isnan(osi_E))]
    osi_I = osi_I[np.logical_not(np.isnan(osi_I))]    
    binEdges = np.linspace(np.nanmin(osi_E), np.nanmax(osi_E), 50)

    binEdges = np.concatenate(([0], binEdges))
    # fgtmp, axtmp = plt.subplots()
    counts, bins, _ = plt.hist(osi_E, binEdges, histtype='step', normed=1, color='w')
    xe = (bins[:-1] + bins[1:]) / 2.0
    xe = bins[:-1]
    #
    binEdges = np.linspace(np.nanmin(osi_I), np.nanmax(osi_I), 50)
    binEdges = np.concatenate(([0], binEdges))
    counts_i, bins_i, _ = plt.hist(osi_I, binEdges, histtype='step', normed=1, color='w')
    xi = (bins_i[:-1] + bins_i[1:]) / 2.0
    xe = bins[:-1]
    plt.close()
        
    
    # fn = '../data/OSI/osi_mZero%s_mOne_%s_kappa_%s_N%s'%(mZero, int(1e4*mOne), kappa, nGenerated)       # print fn
    # if os.path.exists(fn):
    #     os.system('mv ' + fn + ' ' + fn + '.mat')
    # osi = sio.loadmat(fn)['saveOut']
    # ipdb.set_trace()
    #    ['osi']
    #  osi_e = np.squeeze(osi[0][0][0][0][0])
    #  osi_i = np.squeeze(osi[0][0][1][0][0])

    # osi_e = np.squeeze(osi[0][0][0][0])
    # osi_i = np.squeeze(osi[0][0][1][0])
    # 
    # plot
    if IF_NEW_FIG:
        plt.figure()
    # counts, bins, _ = plt.hist(osi_E, 101, histtype='step', normed=1, color='w')
    # plt.clf()
    # x = (bins[:-1] + bins[1:]) / 2.0
    # counts_i, bins_i, _ = plt.hist(osi_I, 101, histtype='step', normed=1, color='w')
    # plt.clf()    
    # xi = (bins_i[:-1] + bins_i[1:]) / 2.0

    plt.figure()
    plt.draw()
    ipdb.set_trace()    
    plt.plot(xe, counts, 'k', lw = 0.5)
    plt.plot(xi, counts_i, 'r', lw = 0.5)
    print np.nanmean(osi_E), ' ', np.nanmean(osi_I)

    # plt.title(r'$m_0^{(1)} = %s$'%(mOne))
    plt.xlabel('OSI')
    plt.ylabel('Density')
#    plt.xlim(0, 0.5)

    # ipdb.set_trace()nn
    # plt.title(r'$\kappa = %s$'%(kappa))

    if figFolder == '':
        figFolder = './figs/'
    figname = 'osi_distr_kappa_%s_mExtZero_%s_mOne_%s'%(int(10*kappa), int(1e3*mZero), int(1e3*mOne))

    # paperSize = [2.25 * 1.22, 2.25]
    # axPosition =  [0.28, 0.25, .65, .65]
    paperSize = [2.5 * 0.8, 2*0.8]
    # axPosition=[.22, .22, .65, .65]

    # paperSize = [2.5, 2]
    axPosition=[.25, .265, .65, .65]
    figFormat = 'svg'
    
    print figFolder, figname
    plt.ion()
    # sr.FixAxisLimits(plt.gcf(), IF_XTICK_INT = False, nDecimalsX = 1, nDecimalsY = 0)
    # plt.gca().set_xticks([0, 0.25, 0.5])
    plt.gca().set_xticks([0, 0.5, 1])    
    # plt.gca().set_xticklabels(['0', '0.25', '0.5'])
    plt.gca().set_xticklabels(['0', '0.5', '1'])    
    ymin, ymax = plt.gca().get_ylim()
    ymax = round(ymax)
    plt.ylim(0, ymax)
    plt.ylim(0, 5)
    plt.xlim(0, 1)
    plt.gca().set_yticks([0, 5])
    plt.gca().set_yticklabels(['0', '5'])        
    sr.ProcessFigure(plt.gcf(), figFolder+figname, True, IF_XTICK_INT = False, figFormat = figFormat, paperSize = paperSize, titleSize = 10, axPosition=axPosition, tickFontsize = 8, labelFontsize = 10, nDecimalsX = 1, nDecimalsY = 2)
    # ipdb.set_trace()



def PltHistKappaList(mZero=0.075, mOne=0.015, kappaList=[0, 3, 6], nGenerated=10000, figFolder=''):
    # loadmat
    plt.ioff()
    fgtmp, axtmp = plt.subplots()    
    fge, axe = plt.subplots()
    fgi, axi = plt.subplots()
    fgl, axl = plt.subplots()
    colors = ['k', 'g', 'r']
    for i, kappa in enumerate(kappaList):
#        fn = '../data/osi_kappa_%s_mExtZero_%s_mOne_%s.mat'%(int(10*kappa), int(1e3*mZero), int(1e4*mOne))
        filename = '../data/OSI/osi_mZero_%s_mOne_%s_kappa_%s_N%s'%(int(1e3*mZero), int(1e4*mOne), kappa, nGenerated)
        if os.path.exists(filename):
            os.system('mv ' + filename + ' ' + filename + '.mat')
        out = sio.loadmat(filename)
        osi_E = np.squeeze(out['osi_E'])
        osi_I = np.squeeze(out['osi_I'])
        binEdges = np.linspace(np.nanmin(osi_E), np.nanmax(osi_E), 40)
        binEdges = np.concatenate(([0], binEdges))
        counts, bins, _ = axtmp.hist(osi_E, binEdges, histtype='step', normed=1, color='k')
        xe = (bins[:-1] + bins[1:]) / 2.0
        xe = bins[:-1]
        #
        binEdges = np.linspace(np.nanmin(osi_I), np.nanmax(osi_I), 40)
        binEdges = np.concatenate(([0], binEdges))
        counts_i, bins_i, _ = axtmp.hist(osi_I, binEdges, histtype='step', normed=1, color='r')
        xi = (bins_i[:-1] + bins_i[1:]) / 2.0
        xe = bins[:-1]
        
        axe.plot(xe, counts, 'k', color=colors[i], lw=0.500)
        axi.plot(xi, counts_i, 'r', color=colors[i], lw=0.50)
        print np.nanmean(osi_E), ' ', np.nanmean(osi_I)
        plt.plot(1, 1, color=colors[i], label=r'$\kappa=%s$'%(kappa))

        

    
        

        

    plt.xlim(0, 1)
    if figFolder == '':
        figFolder = './figs/'
    fignameE = 'osi_E_distr_kappa_%s_mExtZero_%s_mOne_%s'%(int(10*kappa), int(1e3*mZero), int(1e3*mOne))
    fignameI = 'osi_I_distr_kappa_%s_mExtZero_%s_mOne_%s'%(int(10*kappa), int(1e3*mZero), int(1e3*mOne))    
    paperSize = [2.5/2, 2/2.0]
    axPosition=[.25, .22, .65, .65]
    figFormat = 'svg'
    print figFolder, fignameE
    # Excitato
    plt.figure(fgtmp.number)
    plt.xlim(0, 1)
    plt.ylim(0, 8)
    #plt.close()
    plt.figure(fge.number)
    plt.xlim(0, 1)
    plt.ylim(0, 7)
    plt.gca().set_xticks([0, 0.5, 1])
    plt.gca().set_xticklabels(['0', '0.5', '1'])
    ymin, ymax = plt.gca().get_ylim()

    # ipdb.set_trace()    
    plt.gca().set_yticks([0, int(ymax)])
    # plt.draw()

    
    sr.ProcessFigure(fge, figFolder+fignameE, True, IF_XTICK_INT = False, figFormat = figFormat, paperSize = paperSize, titleSize = 10, axPosition=axPosition, tickFontsize = 8, labelFontsize = 10, nDecimalsX = 1, nDecimalsY = 2)
    # Inhibi
    plt.figure(fgi.number)
    plt.xlim(0, 1)
    plt.ylim(0, 5)    
    plt.gca().set_xticks([0, 0.5, 1])
    plt.gca().set_xticklabels(['0', '0.5', '1'])
    ymin, ymax = plt.gca().get_ylim()
    plt.gca().set_yticks([0, int(ymax)])
    plt.draw()
    print figFolder, fignameI    
    sr.ProcessFigure(fgi, figFolder+fignameI, True, IF_XTICK_INT = False, figFormat = figFormat, paperSize = paperSize, titleSize = 10, axPosition=axPosition, tickFontsize = 8, labelFontsize = 10, nDecimalsX = 1, nDecimalsY = 2)

    plt.figure(fgl.number)
    axl.legend(frameon=False, loc=10, prop={'size':20})
    plt.savefig(figFolder + 'fig5_legend.svg')
    
    

def PltPopAvgTuning(mZero = 0.075, mOne = 0.015, kappa=0, nGenerated=1000000, figFolder=''):

    fn = '../data/TC/tc_kappa_%s_mExtZero_%s_mOne_%s_N%s'%(10*kappa, int(1e3*mZero), int(mOne*1e4), nGenerated)
#    fn = '../data/TC/tc_kappa_%s_mExtZero_%s_mOne_%s'%(int(10*kappa), int(1e3*mZero), int(1e3*mOne))
    # fn = '../data/TC/tc_kappa_%s_mExtZero_%s_mOne_%s'%(10*kappa, int(1e3*mZero), 1e3*mOne)    

    filename = '../data/TC/tc_mZero_%s_mOne_%s_kappa_%s_N%s'%(int(1e3*mZero), int(1e4 * mOne), kappa, nGenerated)
   
    if os.path.exists(filename):
        os.system('mv ' + filename + ' ' + filename + '.mat')
    tc = sio.loadmat(filename) #['tc']
    tc_e = tc['tcE']
    tc_i = tc['tcI']

    # tc_e = tc[0][0][0]
    # tc_i = tc[0][0][1]  # nNeurons x nThetas
    plt.figure()

    PlotMeanTc(tc_e, kappa, nPhis = tc_e.shape[1], pcolor='k')
    PlotMeanTc(tc_i, kappa, nPhis = tc_i.shape[1], pcolor='r')

    # ipdb.set_trace()                
#    filename = './figs/PUB_FIGS/mOne%s/'%(int(mOne * 1e3)) + 'pop_mean_tuning_kappa_%s_mExtZero_%s_mOne_%s'%(int(10*kappa), int(1e3*mZero), int(1e3*mOne))


    if figFolder == '':
        figFolder = './figs/'
    
    filename = figFolder + 'pop_mean_tuning_kappa_%s_mExtZero_%s_mOne_%s'%(int(10*kappa), int(1e3*mZero), int(1e3*mOne))
    

    # paperSize = [2.5, 2]
    paperSize = [2.5 * 0.8, 2*0.8]    
    # axPosition=[.22, .22, .65, .65]
    axPosition=[.25, .265, .65, .65]
    # paperSize = [2.25 * 1.22, 2.25]
    # figFormat = 'svg'
    # axPosition =  [0.28, 0.25, .65, .65]
    
    plt.ylim([0, 1.012])
    plt.gca().set_yticks([0, 0.5, 1])
    ax = plt.gca()
    ax.set_yticklabels(['0', '0.5', '1'])
    ax.set_xticklabels(['-90', '0', '90'])    
    sr.ProcessFigure(plt.gcf(), filename, 1, paperSize = paperSize, axPosition = axPosition, titleSize=10, nDecimalsX=1, nDecimalsY=1, figFormat='svg', labelFontsize = 10, tickFontsize = 8)
    


    
def PlotMeanTc(tc, kappa, J = 1, nPhis = 8, N = 10000, labelTxt='', pcolor='k'):
    # J is the rewired strenghtened prefactor
    # thetas = np.linspace(0, 180, nPhis, endpoint = 1)
    N, nPhis = tc.shape
    thetas = np.arange(-90, 90, 180.0/nPhis)
    prefferedOri = np.argmax(tc, 1)
    tcmax = np.max(np.abs(tc), 1)
    # ipdb.set_trace()
    tcmax.shape = N, 1
    tcmax = np.tile(tcmax, (1, nPhis))
    tc = tc / tcmax
    cvMat = np.empty((N, len(thetas)))
    for kNeuron in np.arange(N):
	cvMat[kNeuron, :] = np.roll(tc[kNeuron, :], -1 * prefferedOri[kNeuron])

    plt.ion()
    tmp = cvMat #[plotId[plotId < N], :]
    mean_tc = np.nanmean(tmp, 0)
    mean_tc = np.roll(mean_tc, int(nPhis/2))
    osi = sr.OSI(mean_tc, thetas)
    print 'osi = ', osi
    thetas = np.arange(-90.0, 91.0, 180.0 / nPhis)
    print pcolor
    plt.plot(thetas, np.concatenate((mean_tc, [mean_tc[0]])), 'o-', color = pcolor, markersize = 0.8, markeredgecolor = pcolor, lw=0.5)
    # plt.ylim(0.8, 1)
    plt.gca().set_xticks([-90, 0, 90])
    return mean_tc
    

def PlotNTc(mZero = 0.075, mOne = 0.015, kappa=0, n=10, nGenerated=1000000, figFolder=''):

    filename = '../data/TC/tc_mZero_%s_mOne_%s_kappa_%s_N%s'%(int(1e3*mZero), int(1e4 * mOne), kappa, nGenerated)
    if os.path.exists(filename):
        os.system('mv ' + filename + ' ' + filename + '.mat')
    tc = sio.loadmat(filename) #['tc']
    tc_e = tc['tcE']
    tc_i = tc['tcI']
    if figFolder == '':
        figFolder = './figs/'
    

    # fn = '../data/TC/tc_kappa_%s_mExtZero_%s_mOne_%s_N%s'%(10*kappa, int(1e3*mZero), int(mOne*1e4), nGenerated)
    # # ipdb.set_trace()
    # if os.path.exists(fn):
    #     os.system('mv ' + fn + ' ' + fn + '.mat')
    
    # tc = sio.loadmat(fn)['tc']
    # tc_e = tc[0][0][0]
    # tc_i = tc[0][0][1]  # nNeurons x nThetas
    plt.figure()
    PlotNTcAux(tc_e, neuronType='E', mZero=mZero, mOne=mOne, kappa=kappa, n=n, figFolder=figFolder)
    PlotNTcAux(tc_i, neuronType='I', mZero=mZero, mOne=mOne, kappa=kappa, n=n, figFolder=figFolder)    

    
def PlotNTcAux(tc, neuronType = 'E', mZero = 0.075, mOne = 0.015, kappa=0, n=10, figFolder=''):
    # neuronsIdx = [1798, 7753, 3233, 6218]
    nNeurons, nPhis = tc.shape    
    neuronsIdx = np.random.choice(nNeurons, n)
    plt.ion()    
    for idx in neuronsIdx:
        tmp = tc[idx, :]
        if neuronType == 'E':
            PlotTc(tmp, pcolor='k', IF_ROLL=False)
        else:
            PlotTc(tmp, pcolor='r', IF_ROLL=False)
        
        filename = figFolder + 'tc/%s/tc_%s_kappa_%s_mExtZero_%s_mOne_%s'%(neuronType, idx, int(10*kappa), int(1e3*mZero), int(1e4*mOne))

        # filename = './figs/tc/tc_%s_%s_kappa_%s_mExtZero_%s_mOne_%s'%(neuronType, idx, int(10*kappa), int(1e3*mZero), int(1e4*mOne))

        paperSize = [2.5/2, 2/2.0]
        axPosition=[.25, .20, .6, .6]
        # plt.ylim(0, 1)
        sr.FixAxisLimits(plt.gcf())

        plt.gca().set_xticks([0, 90, 180])
        # plt.gca().set_yticks([0, 0.5, 1])

        ax = plt.gca()
        ymin, ymax = plt.ylim()
        plt.ylim(0, ymax)
        yl = [0, ymax / 2.0, ymax]
        ax.set_yticks(yl)
        ylbls = []
        [ylbls.append('%.4s'%(l)) for l in yl]
        ax.set_yticklabels(ylbls)
        ax.set_xticklabels(['0', '90', '180'])

        plt.draw()
        sr.ProcessFigure(plt.gcf(), filename, 1, paperSize = paperSize, axPosition = axPosition, titleSize=6, nDecimalsX=1, nDecimalsY=1, figFormat='svg', labelFontsize = 6, tickFontsize = 6)

        plt.clf()        


def PlotTc(tc, pcolor, IF_ROLL=False, normed=False):
    nPhis = tc.shape[0]
    # thetas = np.arange(-90, 91, 180./nPhis)
    thetas = np.arange(0, 181.0, 180./nPhis)
    # colors = ['k', 'g', 'r']
    # gamma = 0;
    # for idx in neuronsIdx:    
    #     for kappaIdx, kappa in enumerate(kappaList):
    #         if kappa == 0:
    #             trNo = 0
    #         else:
    #             trNo = 1
    #         tc = GetTuningCurves(kappa, gamma, nPhis, mExt, mExtOne, rewireType, trNo, NE, K, nPop, T, kappa= 0, IF_SUCCESS = False)
            # tc = tcList[kappaIdx]
    if normed:
        tmp = tc / np.max(tc)
    else:
        tmp = tc

    if IF_ROLL:
        tmp = np.roll(tmp, -1 * np.argmax(tmp))
        tmp = np.roll(tmp, int(nPhis/2))

    plt.title('OSI = %.4s'%(sr.OSI(tc, thetas[:-1])))
    plt.plot(thetas, np.concatenate((tmp, [tmp[0]])), 'o-', color = pcolor, lw = .5, markersize = 0.8, markerfacecolor = pcolor, markeredgecolor=pcolor)

def RollTc(tc, pc='k'):

    poe = sr.POofPopulation(out, thetas, IF_IN_RANGE=0)
#    poff = load from mat
    
    
    # nNeurons, nPhis = tc.shape
    # thetas = np.arange(0.0, 180.0, 180. / nPhis)            
    # np.random.seed(1234)
    # rollidx = np.random.choice(nPhis, nNeurons)

    # thetas0 = np.arange(-90, 90, 180. / nPhis)                
    # poff = thetas0[-rollidx]    
    # out = np.zeros(tc.shape)
    # for i in range(nNeurons):
    #     out[i] = np.roll(tc[i], -1 * rollidx[i])
    # poe = sr.POofPopulation(out, thetas, IF_IN_RANGE=0)
    # poff = poff - 1e-2  + 2e-2 * np.random.uniform(size=nNeurons) * 180


    # plt.plot(poff, poe, '.', color=pc, markersize=0.25)

    # # plt.gca().set_xticks([0, 90, 180])

    # # plt.ylim(-90, 90)
    # # plt.xlim(-90, 90)
    # plt.gca().set_xticks([-90, 0, 90])
    # plt.gca().set_yticks([-90, 0, 90])
    # plt.gca().set_xticklabels(['0', '90', '180'])    
    # plt.gca().set_yticklabels(['0', '90', '180'])

    # plt.show()

    # CCC = CircularCorrCoeff(poff * np.pi / 180.0, poe * np.pi / 180)
    # print 'CCC = ', CCC

    # plt.title('CCC = %.4s'%(CCC))

    # plt.plot(thetas, tc[1], 'k')

    # plt.plot(thetas, out[1], 'g')
    # plt.vlines(poff[1], *plt.ylim())    
    # plt.vlines(poe[1], *plt.ylim(), color = 'b')
    # ipdb.set_trace()      
    return poe, poff

def PlotPOScatter(mZero=0.075, mOne=0.015, kappa=0, nGenerated=1000000):
    #
    # fn = '../data/TC/tc_kappa_%s_mExtZero_%s_mOne_%s'%(int(10*kappa), int(1e3*mZero), int(1e3*mOne))
    fn = '../data/TC/tc_kappa_%s_mExtZero_%s_mOne_%s_N%s'%(10*kappa, int(1e3*mZero), int(mOne*1e3), nGenerated)

    if os.path.exists(fn):
        os.system('mv ' + fn + ' ' + fn + '.mat')


        
    tc = sio.loadmat(fn)['tc']
    tc_e = tc[0][0][0]
    tc_i = tc[0][0][1]  # nNeurons x nThetas

    nNeurons, nThetas = tc_i.shape

    fnosi = '../data/OSI/osi_mZero%s_mOne_%s_kappa_%s_N%s'%(mZero, mOne, kappa, nGenerated)    
    if os.path.exists(fnosi):
        os.system('mv ' + fnosi + ' ' + fnosi + '.mat')
    osi = sio.loadmat(fnosi)['saveOut']    
    
    poff = np.squeeze(osi[0][0][4])  #  * 180 / np.pi
    # thetas = osi[0][0][3]

    thetas = np.linspace(0., 180., nThetas)
    
    nNeurons, nPhis = tc.shape
    #thetas = np.arange(0.0, 180.0, 180. / nPhis)
    #
    
    print 'poff is', poff[0:10]
    poe = sr.POofPopulation(tc_e, thetas, IF_IN_RANGE=1, poff=poff)



#    plt.plot(poff%180, poe%180, '.', color='k', markersize=0.25)    

    
    plt.plot(poff, poe, '.', color='k', markersize=0.25)

    # plt.plot(np.sin(2 * poff*np.pi/180), np.sin(2 *poe *np.pi / 180), '.', color='k', markersize=0.25)
    # ipdb.set_trace()

    plt.xlabel(r'$\sin(2 \theta^{PO}_{ff})$')
    plt.ylabel(r'$\sin(2 \theta^{PO}_{out})$')    
    filenameE = './figs/scatter_poff_vs_poe_%s_kappa_%s_mExtZero_%s_mOne_%s'%('E', int(10*kappa), int(1e3*mZero), int(1e3*mOne))

    paperSize = [2.5 * 0.8, 2*0.8]    
    axPosition=[.25, .265, .65, .65]
    
    sr.ProcessFigure(plt.gcf(), filenameE, 1, paperSize = paperSize, axPosition = axPosition, titleSize=6, nDecimalsX=1, nDecimalsY=1, figFormat='png', labelFontsize = 10, tickFontsize = 8)


    plt.clf()
    plt.draw()
    plt.show()
    filenameI = './figs/' + 'scatter_poff_vs_poe_%s_kappa_%s_mExtZero_%s_mOne_%s'%('I', int(10*kappa), int(1e3*mZero), int(1e3*mOne))
    
    poi = sr.POofPopulation(tc_i, thetas, IF_IN_RANGE=0)
    plt.plot(poff%180, poi%180, '.', color='r', markersize=0.25)
    plt.xlim(0, 180)
    
    # poi, poff = RollTc(tc_i[:10000], pc='r')
    sr.ProcessFigure(plt.gcf(), filenameI, 1, paperSize = paperSize, axPosition = axPosition, titleSize=6, nDecimalsX=1, nDecimalsY=1, figFormat='png', labelFontsize = 10, tickFontsize = 8)

    plt.draw()    
    
        
def CircularCorrCoeff(x, y):
    # x and y must be in radians
    n = x.size
    nX = x.size
    nY = y.size
    if nX != nY:
    	n = np.max([nX, nY])
    if(nX != n):
    	x = np.ones((n, )) * x
    if(nY != n):
    	y = np.ones((n, )) * y

    numerator = 0
    for i in range(n - 1):
	for j in range(i + 1, n):
	    numerator += np.sin(2.0 * (x[i] - x[j])) * np.sin(2.0 * (y[i] - y[j]))
    denom1 = 0
    denom2 = 0
    for i in range(n - 1):
	for j in range(i + 1, n):
	    denom1 += np.sin(2.0 * (x[i] - x[j]))**2
	    denom2 += np.sin(2.0 * (y[i] - y[j]))**2
    denom = np.sqrt(denom1 * denom2)
    return numerator / denom


def CCCvsM1(mExtOneList = [0.01, 0.0375, 0.075], cff=1):
    ccc = []
    plt.figure()
    fldr = '../data/'
    for mExtOne in mExtOneList:
        filename = 'poff_and_po_EandI_m1_%s_cff_%s.mat'%(int(1e4*mExtOne), int(cff*1e3))
        tmp = sio.loadmat(fldr + filename)
        print filename
        poFF = np.squeeze(tmp['po_of_FF'])
        poE = np.squeeze(tmp['po_of_E'])
        poI = np.squeeze(tmp['po_of_I'])
        # filename = 'poff_and_poE_m1_%s.mat'%(int(1e3*mExtOne), )
        # tmp = loadmat(fldr + filename)
        # print filename
        # poFF = np.squeeze(tmp['phiFF'])
        # poE = np.squeeze(tmp['poofpopcheck'])
        # ccc.append(CircularCorrCoeff(poFF * np.pi/180, poE * np.pi/180))
        # print gen_ccc
    plt.plot(mExtOneList, ccc, 'o-')
    plt.xlabel(r'$m_0^{(1)}$')
    plt.ylabel('Circular Corr Coeff')
    plt.savefig('./figs/gen_ccc_vs_m1.png')
    plt.show()

def CCC_vs_cff(compute=0, cff_list = [0.25, 0.5, 0.75, 1], mExtOne=0.01, figFolder='', kappa=0):
    ccc_E = []
    ccc_I = []
    out = {}
    ml_list = [1, 0.5, 0.25]
    mExtOneList = np.array(ml_list) * 0.0750
    if compute:
        fldr = '../data/'
        for mExtOne in mExtOneList:
            for cff in cff_list:
                filename = 'poff_and_po_EandI_m1_%s_cff_%s.mat'%(int(1e4*mExtOne), int(cff*1e3))
                tmp = sio.loadmat(fldr + filename)
                print filename
                poFF = np.squeeze(tmp['po_of_FF'])
                poE = np.squeeze(tmp['po_of_E'])
                poI = np.squeeze(tmp['po_of_I'])
                print 'computing ccc E'
                ccc_E.append(CircularCorrCoeff(poFF * np.pi/180, poE * np.pi/180))
                print 'computing ccc I'
                ccc_I.append(CircularCorrCoeff(poFF * np.pi/180, poI * np.pi/180))
                print ccc_E, ccc_I
            out['cff_list'] = cff_list
            out['ccc_E'] = ccc_E
            out['ccc_I'] = ccc_I
            np.save('./data/ccc_poff_and_po_EandI_m1_%s'%(int(1e4*mExtOne)), out)
    else:
        plt.ion()
        fg0, ax0 = plt.subplots()
        fg1, ax1 = plt.subplots()
        fgl, axl = plt.subplots()
        for k, mExtOne in enumerate(mExtOneList):
            out = np.load('./data/ccc_poff_and_po_EandI_m1_%s.npy'%(int(1e4*mExtOne)))[()]
            cff_list = out['cff_list']
            ccc_E = out['ccc_E'][-4:]
            ccc_I = out['ccc_I'][-4:]
            
            print 'm1 = ', mExtOne, ' ', len(ccc_E), ' ', len(ccc_I)
            ax0.plot(cff_list, ccc_E, 'o-', lw=0.5, markersize=0.8)
            ax1.plot(cff_list, ccc_I, 'o-', lw=0.5, markersize=0.8)
            axl.plot(1, 1, label=r'$\mu_0 = %s$'%(ml_list[k]))
        axl.legend(frameon=False, loc=10, prop={'size':20})

        
        if figFolder == '':
            figFolder = './figs/'
        fgl.savefig(figFolder + 'legend_ccc.svg')

        plt.figure(fg0.number)        
        plt.xlabel(r'$c$')
        # plt.ylabel('Circ Corr')
        filename = 'ccc_vscff_%s_kappa_%s_mOne_%s'%('E', int(10*kappa), int(1e3*mExtOne))
        paperSize = [2.5/2, 2/2]    
        axPosition=[.28, .33, .6, .6]
        ax = plt.gca()
        ax.set_xticks(cff_list)
        xticklabels = []
        [xticklabels.append('%s'%(c)) for c in cff_list]
        ax.set_xticklabels(xticklabels)
        ymin, ymax = plt.ylim()
        plt.ylim(0, 1)
        ax.set_yticks([0, 1])
        # ax.set_yticklabels(['0', '%.4s'%(ymax)])
        ax.set_yticklabels(['0', '1'])        
        plt.draw()
        sr.ProcessFigure(plt.gcf(), figFolder + filename, 1, paperSize = paperSize, axPosition = axPosition, titleSize=6, nDecimalsX=1, nDecimalsY=1, figFormat='svg', labelFontsize = 8, tickFontsize = 6)




        filename = 'ccc_vscff_%s_kappa_%s_mOne_%s'%('I', int(10*kappa), int(1e3*mExtOne))

        plt.figure(fg1.number)        
        plt.xlabel(r'$c$')
        # plt.ylabel('Circ Corr')
        paperSize = [2.5/2, 2/2]    
        axPosition=[.28, .33, .60, .60]
        ax = plt.gca()
        ax.set_xticks(cff_list)
        xticklabels = []
        [xticklabels.append('%s'%(c)) for c in cff_list]
        ax.set_xticklabels(xticklabels)
        # ymin, ymax = plt.ylim()
        # ax.set_yticks([0, ymax])
        # ax.set_yticklabels(['0', '%.4s'%(ymax)])
        plt.ylim(0, 1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['0', '1'])        
        
        plt.draw()
        sr.ProcessFigure(plt.gcf(), figFolder + filename, 1, paperSize = paperSize, axPosition = axPosition, titleSize=6, nDecimalsX=1, nDecimalsY=1, figFormat='svg', labelFontsize = 8, tickFontsize = 6)

        

        plt.show()
        # ipdb.set_trace()


    


def POffvsPOoutput(cff_list = [0.25], mExtOne=0.01, nPoints = 5000, figFolder='', kappa=0, mZero=0.075, nGenerated=1000000):

    fldr = '../data/'
    plt.ion()
    iidx = np.random.choice(nGenerated, nPoints)
    for cff in cff_list:
        filename = 'poff_and_po_EandI_m1_%s_cff_%s.mat'%(int(1e4*mExtOne), int(cff*1e3))
        tmp = sio.loadmat(fldr + filename)
        print filename
        poFF = np.squeeze(tmp['po_of_FF'])
        poE = np.squeeze(tmp['po_of_E'])
        poI = np.squeeze(tmp['po_of_I'])
        plt.figure(100)
        # plt.plot(poFF[:nPoints], poE[:nPoints], 'ko', lw=0.5, markersize=0.25)
        plt.plot(poFF[iidx], poE[iidx], 'ko', lw=0.5, markersize=0.25)        
        plt.figure(101)        
        plt.plot(poFF[iidx], poI[iidx], 'ro', lw=0.5, markersize=0.25, markeredgecolor='r')        
        if figFolder == '':
            figFolder = './figs/'
        plt.draw()
        plt.figure(100)
        plt.xlabel(r'$PO_{ff}$')
        plt.ylabel(r'$PO_{response}$')
        plt.ylim(0, 180)
        plt.xlim(0, 180)        
        filenameE = 'scatter_poff_vs_poe_%s_kappa_%s_mExtZero_%s_mOne_%s'%('E', int(10*kappa), int(1e3*mZero), int(1e3*mExtOne))

        paperSize = [2.5 * 0.8, 2*0.8]    
        axPosition=[.28, .28, .65, .65]


        
        ax = plt.gca()
        ax.set_xticks([0, 90, 180])
        ax.set_xticklabels(['0', '90', '180'])
        ax.set_yticks([0, 90, 180])
        ax.set_yticklabels(['0', '90', '180'])

        figFormat = 'svg'

        sr.ProcessFigure(plt.gcf(), figFolder + filenameE, 1, paperSize = paperSize, axPosition = axPosition, titleSize=6, nDecimalsX=1, nDecimalsY=1, figFormat=figFormat, labelFontsize = 10, tickFontsize = 8)
        sr.ProcessFigure(plt.gcf(), figFolder + filenameE, 1, paperSize = paperSize, axPosition = axPosition, titleSize=6, nDecimalsX=1, nDecimalsY=1, figFormat='png', labelFontsize = 10, tickFontsize = 8)        
        ## 
        plt.figure(101)
        plt.xlabel(r'$PO_{ff}$')
        plt.ylabel(r'$PO_{response}$')
        plt.xlim(0, 180)
        plt.ylim(0, 180)
        ax = plt.gca()
        ax.set_xticks([0, 90, 180])
        ax.set_xticklabels(['0', '90', '180'])
        ax.set_yticks([0, 90, 180])
        ax.set_yticklabels(['0', '90', '180'])        
        filenameI = 'scatter_poff_vs_poe_%s_kappa_%s_mExtZero_%s_mOne_%s'%('I', int(10*kappa), int(1e3*mZero), int(1e3*mExtOne))        
        sr.ProcessFigure(plt.gcf(), figFolder + filenameI, 1, paperSize = paperSize, axPosition = axPosition, titleSize=6, nDecimalsX=1, nDecimalsY=1, figFormat='png', labelFontsize = 10, tickFontsize = 8)
        sr.ProcessFigure(plt.gcf(), figFolder + filenameI, 1, paperSize = paperSize, axPosition = axPosition, titleSize=6, nDecimalsX=1, nDecimalsY=1, figFormat=figFormat, labelFontsize = 10, tickFontsize = 8)        

