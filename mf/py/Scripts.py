#basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import code, sys, os
import ipdb
import pylab as plt
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#sys.path.append(basefolder)
#import Keyboard as kb
from multiprocessing import Pool
from functools import partial 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
#sys.path.append(basefolder + "/nda/spkStats")
#sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
#from reportfig import ReportFig
from Print2Pdf import Print2Pdf
#import GetPO
from scipy.optimize import curve_fit

def GetBaseFolder(p, gamma, mExt, mExtOne, rewireType, trNo = 0, T = 1000, N = 10000, K = 1000, nPop = 2, kappa = 1):
    if rewireType == 'rand':
	tag = ''
    if rewireType == 'exp':
	tag = '1'
    rootFolder = ''
    ########################## RETORE TO DEFAULT #########################
    # baseFldr = rootFolder + '/homecentral/srao/Documents/code/binary/prx/c/'
    ###############################
    if mExtOne == 0:
        if N == 10000 or N == 80000 or N==20000 or N == 40000:
            baseFldr = rootFolder + '/homecentral/srao/Documents/code/binary/c/'
            if nPop == 1:
                baseFldr = baseFldr + 'onepop/data/N%sK%s/m0%s/mExtOne%s/p%sgamma%s/T%s/tr%s/'%(N, K, int(1e3 * mExt), int(1e3 * mExtOne), int(p * 10), int(gamma * 10), int(T*1e-3), trNo)
            if nPop == 2:
                if gamma >= .1 or gamma == 0:
                    baseFldr = baseFldr + 'twopop/PRX_data/data/rewire/N%sK%s/m0%s/mExtOne%s/kappa%s/p0gamma%s/T%s/tr%s/'%(N, K, int(1e3 * mExt), int(1e3 * mExtOne), int(p * 10), int(gamma * 10), int(T*1e-3), trNo)
                else:
                    baseFldr = baseFldr + 'twopop/PRX_data/data/rewire/N%sK%s/m0%s/mExtOne%s/kappa%s/p0gamma/T%s/tr%s/'%(N, K, int(1e3 * mExt), int(1e3 * mExtOne), int(p * 10), int(T*1e-3), trNo)
        # elif N==80000:
        #     baseFldr = rootFolder + '/homecentral/srao/Documents/code/binary/prx/c/'
        #     if nPop == 1:
        #         baseFldr = baseFldr + 'onepop/data/N%sK%s/m0%s/mExtOne%s/p%sgamma%s/T%s/'%(N, K, int(1e3 * mExt), int(1e3 * mExtOne), int(p * 10), int(gamma * 10), int(T*1e-3))
        #     if nPop == 2:
        #         if gamma >= .1 or gamma == 0:
        #             baseFldr = baseFldr + 'twopop/data/N%sK%s/m0%s/mExtOne%s/p%sgamma%s/T%s/'%(N, K, int(1e3 * mExt), int(1e3 * mExtOne), int(p * 10), int(gamma * 10), int(T*1e-3))
        #         else:
        #             baseFldr = baseFldr + 'twopop/data/N%sK%s/m0%s/mExtOne%s/p%sgamma/T%s/'%(N, K, int(1e3 * mExt), int(1e3 * mExtOne), int(p * 10), int(T*1e-3))

    else:
        baseFldr = rootFolder + '/homecentral/srao/Documents/code/binary/prx/c/'
        if nPop == 1:
            baseFldr = baseFldr + 'onepop/data/N%sK%s/m0%s/mExtOne%s/p%sgamma%s/T%s/tr%s/'%(N, K, int(1e3 * mExt), int(1e3 * mExtOne), int(p * 10), int(gamma * 10), int(T*1e-3), trNo )
        if nPop == 2:
            if gamma >= .1 or gamma == 0:
                baseFldr = baseFldr + 'twopop/data/N%sK%s/m0%s/mExtOne%s/p%sgamma%s/T%s/tr%s/'%(N, K, int(1e3 * mExt), int(1e3 * mExtOne), int(p * 10), int(gamma * 10), int(T*1e-3), trNo )
            else:
                baseFldr = baseFldr + 'twopop/data/N%sK%s/m0%s/mExtOne%s/p%sgamma/T%s/tr%s/'%(N, K, int(1e3 * mExt), int(1e3 * mExtOne), int(p * 10), int(T*1e-3), trNo )
                    
	# if gamma >= .1 or gamma == 0:
        #     baseFldr = baseFldr + 'twopop/data/JII/N%sK%s/m0%s/mExtOne%s/p%sgamma%s/T%s/tr%s/'%(N, K, int(1e3 * mExt), int(1e3 * mExtOne), int(p * 10), int(gamma * 10), int(T*1e-3), trNo )
        # else:
        #     baseFldr = baseFldr + 'twopop/data/JII/N%sK%s/m0%s/mExtOne%s/p%sgamma/T%s/tr%s/'%(N, K, int(1e3 * mExt), int(1e3 * mExtOne), int(p * 10), int(T*1e-3), trNo )

	# if gamma >= .1 or gamma == 0:
        #     baseFldr = baseFldr + 'twopop/data/tmp/'
        # else:
        #     baseFldr = baseFldr + 'twopop/data/tmp/'

            
    # print baseFldr
    return baseFldr

# def GetBaseFolder(p, gamma, mExt, mExtOne, rewireType, trNo = 0, T = 1000, N = 10000, K = 1000, nPop = 2, kappa = 1):
#     # print 'p=', p, ' , kappa = ', kappa
#     if rewireType == 'rand':
# 	tag = ''
#     if rewireType == 'exp':
# 	tag = '1'
#     rootFolder = ''
#     baseFldr = rootFolder + '/homecentral/srao/Documents/code/binary/prx/c/'
#     if nPop == 1:
#     	baseFldr = baseFldr + 'onepop/data/N%sK%s/m0%s/mExtOne%s/p%sgamma%s/T%s/'%(N, K, int(1e3 * mExt), int(1e3 * mExtOne), int(p * 10), int(gamma * 10), int(T*1e-3), )
#     if nPop == 2:
# 	if gamma >= .1 or gamma == 0:
# 	    baseFldr = baseFldr + 'twopop/data/N%sK%s/m0%s/mExtOne%s/p%sgamma%s/T%s/'%(N, K, int(1e3 * mExt), int(1e3 * mExtOne), int(p * 10), int(gamma * 10), int(T*1e-3), )
# 	else:
# 	    baseFldr = baseFldr + 'twopop/data/N%sK%s/m0%s/mExtOne%s/p%sgamma/T%s/'%(N, K, int(1e3 * mExt), int(1e3 * mExtOne), int(p * 10), int(T*1e-3), )
#     # print baseFldr	    
#     return baseFldr
    

def LoadFr(p, gamma, phi, mExt, mExtOne, rewireType, trNo = 0, T = 1000, N = 10000, K = 1000, nPop = 2, IF_VERBOSE = False, kappa = 0):

    kappa=p
    baseFldr = GetBaseFolder(p, gamma, mExt, mExtOne, rewireType, trNo, T, N, K, nPop, kappa)
    # ipdb.set_trace()
    filename = 'meanrates_theta%.6f_tr%s.txt'%(phi, trNo)
    if IF_VERBOSE:
    	print baseFldr
	print filename
    return np.loadtxt(baseFldr + filename)


def GetTuningCurves(p, gamma, nPhis, mExt, mExtOne, rewireType, trNo = 0, N = 10000, K = 1000, nPop = 2, T = 1000, kappa = 0, IF_SUCCESS = False):
    NE = N
    NI = N
    tc = np.zeros((NE + NI, nPhis))
    tc[:] = np.nan
    phis = np.linspace(0, 180, nPhis, endpoint = False)
    IF_FILE_LOADED = False
    for i, iPhi in enumerate(phis):
	print i, iPhi
	try:
	    if i == 0:
		print 'loading from fldr: ',
    #                       p, gamma, phi, mExt, mExtOne, rewireType, trNo = 0, T = 1000, N = 10000, K = 1000, nPop = 2, IF_VERBOSE = False, kappa = 0):                
		fr = LoadFr(p, gamma, iPhi, mExt, mExtOne, rewireType, trNo, T, NE, K, nPop, IF_VERBOSE = True, kappa = kappa)
	    else:
		fr = LoadFr(p, gamma, iPhi, mExt, mExtOne, rewireType, trNo, T, NE, K, nPop, IF_VERBOSE = False, kappa = kappa)
	    if(len(fr) == 1):
		if(np.isnan(fr)):
		    print 'file not found!'
            print fr.shape
            assert fr.size == NE+NI
	    tc[:, i] = fr
            IF_FILE_LOADED = True
	except IOError:
	    print 'file not found!'
	except AssertionError:
            print 'N=', fr.size, '!!!!!!!!!!!!!'
	    print 'file not found!'
            
	    # raise SystemExit
    if IF_SUCCESS:
        return tc, IF_FILE_LOADED
    else:
        return tc


def ProcessFigure(figHdl, filepath, IF_SAVE, IF_XTICK_INT = False, figFormat = 'eps', paperSize = [4, 3], titleSize = 10, axPosition = [0.25, 0.25, .65, .65], tickFontsize = 10, labelFontsize = 12, nDecimalsX = 1, nDecimalsY = 1):
    FixAxisLimits(figHdl)
    FixAxisLimits(plt.gcf(), IF_XTICK_INT, nDecimalsX, nDecimalsY)
    Print2Pdf(plt.gcf(), filepath, paperSize, figFormat=figFormat, labelFontsize = labelFontsize, tickFontsize=tickFontsize, titleSize = titleSize, IF_ADJUST_POSITION = True, axPosition = axPosition)
    plt.show()

def FixAxisLimits(fig, IF_XTICK_INT = False, nDecimalsX = 1, nDecimalsY = 1):
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

def GetPhase(firingRate, atTheta, IF_IN_RANGE = False):
    out = np.nan
   # ipdb.set_trace()
    thetas = atTheta * np.pi / 180
    x = np.dot(np.cos(2 * thetas), firingRate)
    y = np.dot(np.sin(2 * thetas), firingRate)
    out = np.arctan2(y, x) * 180 / np.pi
    # print 'at = ', out, 'a1 ', np.fmod(out + 180, 180.), ' a2 ', (out + 360) / 2

    # zk = np.dot(firingRate, np.exp(2j * atTheta * np.pi / 180))
    # out = np.angle(zk) * 180.0 / np.pi
    # if IF_IN_RANGE:
    #     if(out < 0):
    #         out = np.fmod(out + 180, 180.0)
    if x < 0 and y > 0:
        out = out / 2.0
    elif x > 0 and y < 0:
        out = (out + 360) / 2.0
    elif x < 0 and y < 0:
        out = (out + 360) / 2.0
    else:
        out = out / 2.0;
        

    # if IF_IN_RANGE:
    #     if(out < 0):
    #         out += 360
    #         out = out / 2.0
            
    return out #* 0.5

def POofPopulation(tc, theta = np.arange(0.0, 180.0, 22.5), IF_IN_RANGE = False, poff=''):
    # return value in degrees
    nNeurons, nAngles = tc.shape
    theta = np.linspace(0, 180, nAngles, endpoint = False)
    po = np.zeros((nNeurons, ))
    for kNeuron in np.arange(nNeurons):
        # if kNeuron==10:
        #     ipdb.set_trace()
        po[kNeuron] = GetPhase(tc[kNeuron, :], theta, IF_IN_RANGE)
        # if kNeuron < 51:
        #     plt.plot(theta, tc[kNeuron, :])
        #     print 'Po = ', po[kNeuron]
        #     _, ymax = plt.ylim()
        #     plt.vlines(po[kNeuron], 0, ymax)
        #     plt.title('po ff %.5s po output %.5s'%(poff[kNeuron], po[kNeuron]))
        #     plt.draw()
        #     plt.savefig('tc_%s.png'%(kNeuron))
        #     # if kNeuron==10:
        #     #     ipdb.set_trace()
        #     plt.clf()            
        #     # plt.waitforbuttonpress()
        #     # plt.clf()
    return po 

def GetPOofPop(p, gamma, mExt, mExtOne, rewireType, nPhis = 8, trNo = 0, N = 10000, K = 1000, nPop = 2, T = 1000, IF_IN_RANGE = True, kappa = 1):
    nNeurons = N
    thetas= np.linspace(0, np.pi, nNeurons, endpoint = False)
    tc = GetTuningCurves(p, gamma, nPhis, mExt, mExtOne, rewireType, trNo, N, K, nPop, T, kappa)
    prefferedOri = POofPopulation(tc[:N], IF_IN_RANGE = True) * np.pi / 180.0
    return prefferedOri

def GetPOofPopAllNeurons(p, gamma, mExt, mExtOne, rewireType, nPhis = 8, trNo = 0, N = 10000, K = 1000, nPop = 2, T = 1000, IF_IN_RANGE = True, kappa = 1):
    nNeurons = N
    thetas= np.linspace(0, np.pi, nNeurons, endpoint = False)
    tc = GetTuningCurves(p, gamma, nPhis, mExt, mExtOne, rewireType, trNo, N, K, nPop, T, kappa)
    print tc.shape
    prefferedOri = POofPopulation(tc, IF_IN_RANGE = True) * np.pi / 180.0

    return prefferedOri
    

def OSI(firingRate, atTheta):
    out = np.nan
    zk = np.dot(firingRate, np.exp(2j * atTheta * np.pi / 180))
    # denum = np.absolute(np.dot(firingRate, np.exp(2j * np.zeros(len(atTheta)))))
    if(firingRate.mean() > 0.0):
        out = np.absolute(zk) / np.sum(firingRate)
    return out

def OSIOfPop(firingRates, atThetas):
    # thetas in degrees
    nNeurons, nThetas = firingRates.shape
    atThetas = np.linspace(0, 180, nThetas, endpoint = False)
    out = np.zeros((nNeurons, ))
    for i in range(nNeurons):
        out[i] = OSI(firingRates[i , :], atThetas)
    return out

def LoadM1vsT(p = 0, gamma = 0, phi = 0, trNo = 0, mExt = 0.075, mExtOne = 0.075, K = 1000, NE = 10000, T = 1000, nPop = 2, rewireType = 'rand', IF_VERBOSE = True, kappa = 0):
    N = NE
    baseFldr = GetBaseFolder(p, gamma, mExt, mExtOne, rewireType, trNo, T, N, K, nPop, kappa)
    if IF_VERBOSE:
    	print baseFldr
    filename = 'MI1_inst_theta%.6f_tr%s.txt'%(phi, trNo)
    return np.loadtxt(baseFldr + filename, delimiter = ';')

def PlotM1vsT(p = 0, gamma = 0, phi = 0, trNo = 0, mExt = 0.075, mExtOne = 0.075, K = 1000, NE = 10000, T = 1000, nPop = 2, rewireType = 'cntrl', IF_VERBOSE = True, kappa = 0):
    out = LoadM1vsT(p, gamma, phi, trNo, mExt, mExtOne, K, NE, T, nPop, rewireType, IF_VERBOSE, kappa)
    _, nColumns = out.shape
    if nColumns == 3:
        m1 = out[:, 0]
        m1Phase = out[:, 1]
        phi_ext = out[:, 2]
        tAxis = np.linspace(0, 1, m1.size)
        plt.plot(tAxis, m1)
        plt.xlabel('Time (a.u)')
        plt.ylabel(r'$m_E^{(1)}$')
        plt.ylim(0, .25)
        plt.vlines(.3, *plt.ylim(), color = 'k')
        plt.vlines(.7, *plt.ylim(), color = 'k')        
        filename = './figs/twopop/' + 'stimulus_tracking_m1_p%sgmma%s'%(int(p*10), int(gamma *100))
	paperSize = [2.5, 2]
        axPosition = [0.22, 0.2, .7, .7]
        print 'printing figure'
        ProcessFigure(plt.gcf(), filename, 1, paperSize = paperSize, axPosition=axPosition, titleSize=10, nDecimalsX=1, nDecimalsY=1, figFormat='pdf', labelFontsize = 8, tickFontsize = 6)
        plt.figure()
        plt.plot(tAxis, phi_ext*180/np.pi, '--', label = r'stimulus')    
        plt.plot(tAxis, m1Phase*180/np.pi, alpha = 0.5, label = r'$\angle m_E(\phi)$')
        plt.ylim(0, 180)
        plt.xlabel('Time (a.u)')
        plt.ylabel(r'Phase (deg)')
        plt.legend(loc = 2, frameon = False, numpoints = 1, ncol = 1, prop = {'size': 8})            
        filename = './figs/twopop/' + 'stimulus_tracking_phase_p%sgmma%s'%(int(p*10), int(gamma *100))
	paperSize = [2.5, 2]
        axPosition = [0.22, 0.2, .7, .7]
        print 'printing figure'
        ProcessFigure(plt.gcf(), filename, 1, paperSize = paperSize, axPosition=axPosition, titleSize=10, nDecimalsX=1, nDecimalsY=1, figFormat='pdf', labelFontsize = 8, tickFontsize = 6)        
        ipdb.set_trace()
    else:
        print '-' * 25
        print 'no simulus change'
        print '-' * 25        

def M1Component(x):
    out = np.nan
    if len(x) > 0:
	dPhi = np.pi / len(x)
	out = 2.0 * np.absolute(np.dot(x, np.exp(-2.0j * np.arange(len(x)) * dPhi))) / len(x)
    return out

def PopM1Component(tc):
    nNeurons, nPhis = tc.shape
    outM1 = np.zeros((nNeurons, ))
    for i in range(nNeurons):
        outM1[i] = M1Component(tc[i, :])
    return outM1    
    
def KappaVsM1AtPhi(kappaList, p=0, gamma=0, phi=0, mExt=0.075, mExtOne=0.075, rewireType='rand', N=10000, K=1000, nPop=2, T=1000, trNo=0, IF_PO_SORTED = False, sortedIdx = [], minRate = 0):
    m1E = np.empty((len(kappaList, )))
    mrMean = np.empty((len(kappaList, )))
    validKappa = np.empty((len(kappaList, )))
    m1E[:] = np.nan
    validKappa[:] = np.nan
    mrMean[:] = np.nan
    for kIdx, kappa in enumerate(kappaList):
        p=kappa
	try:
            IF_VERBOSE = False #True
            mr = LoadFr(p, gamma, phi, mExt, mExtOne, rewireType, trNo, T, N, K, nPop, IF_VERBOSE = IF_VERBOSE, kappa = kappa)
            mrMean[kIdx] = np.mean(mr[:N])
	    if not IF_PO_SORTED:
		m1E[kIdx] = M1Component(mr[:N])
	    else:
		mre = mr[:N]
		#m1E[kIdx] = M1Component(mre[sortedIdx[kIdx]])
                mask = mr[sortedIdx] > minRate
                m1E[kIdx] = M1Component(mr[sortedIdx[mask]])		
	    validKappa[kIdx] = kappa
	    # ipdb.set_trace()
	    print 'o',
	except IOError:
	    print 'x', 
	    #pass
	    #print 'kappa: ', kappa, ' no files!'
    sys.stdout.flush()	    
    return validKappa, m1E, mrMean


def KappaVsM1AtTr(kappaList, p=0, gamma=0, nPhis = 8, mExt=0.075, mExtOne=0.075, rewireType='rand', N=10000, K=1000, nPop=2, T=1000, trNo=0, IF_PO_SORTED = False, sortedIdx = [], minRate = 0):
    thetas = np.linspace(0, 180, nPhis, endpoint = False)
    m1E = np.zeros((nPhis, len(kappaList)))
    m0E = np.zeros((nPhis, len(kappaList)))    
    vldKappa = np.empty((nPhis, len(kappaList)))
    vldKappa[:] = np.nan
    for i, phi in enumerate(thetas):
	print 'phi =', phi
	vldKappa[i, :], m1E[i, :], m0E[i, :] = KappaVsM1AtPhi(kappaList, p, gamma, phi, mExt, mExtOne, rewireType, N, K, nPop, T, trNo, IF_PO_SORTED = IF_PO_SORTED, sortedIdx = sortedIdx, minRate = minRate)
    print 'inside tr func', np.nanmean(m1E, 0)
    return np.nanmean(m1E, 0), np.nanmean(m0E, 0) #, vldKappa

def KappaVsM1(kappaList, nTrials = 10, p=0, gamma=0, nPhis = 8, mExt=0.075, mExtOne=0.075, rewireType='rand', N=10000, K=1000, nPop=2, T=1000, IF_PO_SORTED = False, sortedIdx = [], minRate = 0, pcolor = 'k', IF_COMPUTE=False):
    if IF_COMPUTE:
        m1E = np.empty((nTrials, len(kappaList)))
        m1E[:] = np.nan
        m0E = np.empty((nTrials, len(kappaList)))
        m0E[:] = np.nan
        for trNo in range(nTrials): #$ trNo 0 is always the CONTROL 
            print ''
            print '--' * 27
            print 'tr#: ', trNo
            print '--' * 27
            tmp1, tmp0 = KappaVsM1AtTr(kappaList, p, gamma, nPhis, mExt, mExtOne, rewireType, N, K, nPop, T, trNo, IF_PO_SORTED, sortedIdx, minRate = minRate)
            m1E[trNo, :] = tmp1
            m0E[trNo, :] = tmp0
            print tmp0

        m1EvsKappa = np.nanmean(m1E, 0)
        m1EvsKappaSTD = np.nanstd(m1E, 0)    
        numValTrials = np.sum(~np.isnan(m1E), 0)
        validIdx = ~np.isnan(m1EvsKappa)
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
        out['meanRateE'] = np.nanmean(m0E, 0)[validIdx]
        print type(out), len(out)

        
        np.save('./PNAS/data/kappa_vs_mE1_SIM_mZero%s_mOne%s_K%s_N%s'%(mExt, mExtOne, K, N), out)
    else:  # plot
        if N == 10000:
            out = np.load('./PNAS/data/kappa_vs_mE1_SIM_mZero%s_mOne%s_K%s_N%s.npy'%(mExt, mExtOne, K, N)).item()
            out = np.load('./PNAS/data/kappa_vs_mE1_SIM_mZero%s_mOne%s_K%s.npy'%(mExt, mExtOne, K)).item()            
        else:
            out = np.load('./PNAS/data/kappa_vs_mE1_SIM_mZero%s_mOne%s_K%s.npy'%(mExt, mExtOne)).item()
            # out = np.load('./PNAS/data/kappa_vs_mE1_SIM_mZero%s_mOne%s_K%s_N%s.npy'%(mExt, mExtOne, K, N)).item()            
        # mE0 = out['meanRateE']
        ipdb.set_trace()
        kappaList = out['kappa']
        m1EvsKappa = out['m1EvsKappa']
        m1EvsKappaSEM = out['m1EvsKappaSEM']
        # print type(out), len(out)
        for i in range(len(kappaList)):
        # (_, caps, _) = plt.errorbar(kappaList, m1EvsKappa, fmt = 'o-', markersize = 3, yerr = m1EvsKappaSEM, lw = 0.8, elinewidth=0.8, label = r'$N = %s, K = %s$'%(N, K))
            # (_, caps, _) = plt.errorbar(kappaList, m1EvsKappa / 0.17126, fmt = 'o', markersize = 1, yerr = m1EvsKappaSEM, lw = 0.6, elinewidth=0.2, label = r'$N = %s$'%(N, ), markeredgecolor = pcolor, color = pcolor)
            plt.plot(kappaList, m1EvsKappa / 0.17126, 'ok')
            
    return m1EvsKappa, m1EvsKappaSEM


def ProcessFigure(figHdl, filepath, IF_SAVE, IF_XTICK_INT = False, figFormat = 'eps', paperSize = [4, 3], titleSize = 10, axPosition = [0.25, 0.25, .65, .65], tickFontsize = 10, labelFontsize = 12, nDecimalsX = 1, nDecimalsY = 1):
    # FixAxisLimits(figHdl)
    # FixAxisLimits(plt.gcf(), IF_XTICK_INT, nDecimalsX, nDecimalsY)
    Print2Pdf(plt.gcf(), filepath, paperSize, figFormat=figFormat, labelFontsize = labelFontsize, tickFontsize=tickFontsize, titleSize = titleSize, IF_ADJUST_POSITION = True, axPosition = axPosition)
    plt.show()

def FixAxisLimits(fig, IF_XTICK_INT = False, nDecimalsX = 1, nDecimalsY = 1):
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


# def GenerateFig5():
    
    
def KappaMOneVsM1(kappaList, mOneList, kappaCritical, mExt = 0.075, nTrials = 10, p=0, gamma=0, nPhis = 8, rewireType='rand', N=10000, K=1000, nPop=2, T=1000, IF_PO_SORTED = False, sortedIdx = [], minRate = 0):
    colors = ['k', 'g', 'c', 'b']
    gray = [0.8, 0.8, 0.8]
    for l, mExtOne in enumerate(mOneList):
        out = KappaVsM1(kappaList, nTrials = nTrials, p=0, gamma=0, nPhis = nPhis, mExt=mExt, mExtOne=mExtOne, rewireType=rewireType, N=N, K=K, nPop=nPop, T=T, IF_PO_SORTED=IF_PO_SORTED, sortedIdx=sortedIdx, minRate = minRate, pcolor = 'k') #[0.45, 0.45, 0.45]) 
    # PRETTY FIG

    plt.xlim(0, kappaList[-1])
    plt.xlabel(r'$\kappa$', fontsize = 20)
    plt.ylabel(r'$m_E^{(1)}$', fontsize = 20)

    #numerical solution
    out = np.load('./data/mE1_m0%s_K%s_NEW.npy'%(int(mExt * 1e3), K))
    plt.plot(out[0, :], out[1, :], 'k--')
    
    FixAxisLimits(plt.gcf(), IF_XTICK_INT = False, nDecimalsX = 1, nDecimalsY = 2)    
    ax = plt.gca()
    xtickLocs = ax.get_xticks()
    labels = np.array([item.get_text() for item in ax.get_xticklabels()])
    newTickLocs = np.empty((len(labels) + 1, ))
    newTickLocs[kappaCritical > xtickLocs] = xtickLocs[[kappaCritical > xtickLocs]]
    newTickLocs[np.sum(xtickLocs < kappaCritical)] = kappaCritical
    newTickLocs[-1] = xtickLocs[-1]
    newLabels = np.array(['' for i in range(len(labels) + 1)])
    newLabels[kappaCritical > xtickLocs] = labels[[kappaCritical > xtickLocs]]
    newLabels = newLabels.tolist()
    newLabels[np.sum(xtickLocs < kappaCritical)] = r'$\kappa_c$'
    newLabels[-1] = labels[-1]
    ax.set_xticks(newTickLocs)
    ax.set_xticklabels(newLabels)
    figFolder = './figs/' 

    paperSize = [2.25 * 1.22, 2.25]
    figFormat = 'png'
    axPosition =  [0.28, 0.25, .65, .65]
    print figFolder, figname

    plt.ion()
    ProcessFigure(plt.gcf(), figFolder+figname, True, IF_XTICK_INT = False, figFormat = 'svg', paperSize = paperSize, titleSize = 10, axPosition=axPosition, tickFontsize = 8, labelFontsize = 12, nDecimalsX = 1, nDecimalsY = 2)
    plt.show()


def MeanTrialOSIvsKappa(kappa, mExtOne, nTrials = 10, p = 0, gamma = 0, nPhis = 8, mExt = 0.075,  trNo = 1, rewireType = 'rand', N = 10000, K = 1000, nPop = 2, T = 1000, trialsStartidx = 0):
    # colors = [plt.cm.Dark2(i) for i in np.linspace(0, 1, clrCntr + len(kappaList), endpoint = False)]
    meanOSI = []
    meanOSI_I = []
    phis = np.linspace(0, 180, nPhis, endpoint = False)
    validTrList = []
    validTrList = []
    meanOSI = []
    meanOSI_I = []
    counter = 0
    validKappa = []
    p = kappa
    print 'ntrials = ', nTrials
    for trNo in np.arange(trialsStartidx, trialsStartidx + nTrials, 1):
	try:
	    print trNo
	    tc = GetTuningCurves(p, gamma, nPhis, mExt, mExtOne, rewireType, trNo, N, K, nPop, T, kappa)
	    osi = OSIOfPop(tc[:N, :], phis)
	    osiI = OSIOfPop(tc[N:, :], phis)
	    meanOSI.append(np.nanmean(osi))
	    meanOSI_I.append(np.nanmean(osiI))
	    if ~np.isnan(meanOSI[counter]):
		validTrList.append(trNo)
	    validKappa.append(kappa)
	except IOError:
	    print "p = ", p, " gamma = ", gamma, " trial# ", trNo, " file not found"
    meanOSI = np.array(meanOSI)
    meanOSI_I = np.array(meanOSI_I)
    validTrialIdx = ~np.isnan(meanOSI)
    nValidTrials = np.sum(validTrialIdx)

    if nValidTrials > 1:
        return np.nanmean(meanOSI), np.nanstd(meanOSI) / nValidTrials, np.nanmean(meanOSI_I), np.nanstd(meanOSI_I) / nValidTrials
    elif nValidTrials == 1:
        return np.nanmean(meanOSI), np.nan, np.nanmean(meanOSI_I), np.nan
    else:
        return np.nan, np.nan, np.nan, np.nan

def PlotMeanOSIvsKappa(kappaList, mExtOne, nTrials = 10, p = 0, gamma = 0, nPhis = 8, mExt = 0.075,  trNo = 1, rewireType = 'rand', N = 10000, K = 1000, nPop = 2, T = 1000, IF_NEW_FIG = True, clrCntr = 0, filename = '', IF_LEGEND = True, legendTxt = '', color = '', trialsStartidx = 0):
    out = np.empty((len(kappaList), 4))
    kappaList = np.array(kappaList)
    for kk, kappa in enumerate(kappaList):
        if kappa == 8:
            trialsStartidx = 3000
        out[kk, :] = MeanTrialOSIvsKappa(kappa, mExtOne, nTrials, p, gamma, nPhis, mExt,  trNo, rewireType, N, K, nPop, T, trialsStartidx)
    validKappaIdx = ~np.isnan(out[:, 0])
    nValidKappas = np.sum(validKappaIdx)
    print 'kappa = 0'
    print out[0, 0], out[0, 2]
    # PLOT
    (_, caps, _) = plt.errorbar(kappaList[validKappaIdx], out[validKappaIdx, 0], fmt = 'o-', markersize = 1, yerr = out[validKappaIdx, 1], lw = 0.6, elinewidth=0.2, label = 'E', markeredgecolor = 'k', color = 'k')
    for cap in caps:
        cap.set_markeredgewidth(0.2)
    (_, caps, _) = plt.errorbar(kappaList[validKappaIdx], out[validKappaIdx, 2], fmt = 'o-', markersize = 1, yerr = out[validKappaIdx, 3], lw = 0.6, elinewidth=0.2, label = 'I', markeredgecolor = 'r', color = 'r')
    for cap in caps:
        cap.set_markeredgewidth(0.2)
    # PRETTY PRINT
    
    plt.xlabel(r'$\kappa$')
    plt.ylabel(r'$\langle OSI \rangle$')
    plt.gca().set_position([0.25, 0.25, .65, .65])

    ipdb.set_trace()
    np.savez('./data/kappa_vs_OSI_mean_trials_mone%s_K%s'%(int(mExtOne * 1e3), K), kappaList = kappaList[validKappaIdx], poposi=out[validKappaIdx])
    filepath = "./figs/kappa_vs_OSI_mean_trials_mone%s"%(int(mExtOne * 1e3))
    print filepath
    plt.legend(loc = 2, frameon = False, numpoints = 1, prop = {'size': 8})
    print 'saving as: ', filepath
    paperSize = [2.25 * 1.22, 2.25]
    axPosition =  [0.28, 0.25, .65, .65]
    plt.ion()
    plt.ylim(0, 1)
    FixAxisLimits(plt.gcf(), IF_XTICK_INT = False, nDecimalsX = 1, nDecimalsY = 2)
    plt.xlim(-.5, 6.5)
    ProcessFigure(plt.gcf(), filepath, True, IF_XTICK_INT = False, figFormat = 'svg', paperSize = paperSize, titleSize = 10, axPosition=axPosition, tickFontsize = 8, labelFontsize = 12, nDecimalsX = 1, nDecimalsY = 2)
    plt.show()
    
    
    
def CompareMeanOSIvsKappa(kappaList, mExtOne, p = 0, gamma = 0, nPhis = 8, mExt = 0.075,  trNo = 1, rewireType = 'rand', N = 10000, K = 1000, nPop = 2, T = 1000, IF_NEW_FIG = True, clrCntr = 0, filename = '', IF_LEGEND = True, legendTxt = '', color = '', neuronType = 'E'):
    # colors = [plt.cm.Dark2(i) for i in np.linspace(0, 1, 1 + clrCntr + len(pList) * len(gList) * len(mExtOneList) * len(trList), endpoint = False)]
    # colors = [plt.cm.Dark2(i) for i in np.linspace(0, 1, clrCntr + len(kappaList) * len(gList) * len(mExtOneList) * len(KList), endpoint = False)]
    colors = [plt.cm.Dark2(i) for i in np.linspace(0, 1, clrCntr + len(kappaList), endpoint = False)]    
    meanOSI = []
    meanOSI_I = []
    phis = np.linspace(0, 180, nPhis, endpoint = False)
    validTrList = []
    legendTxtTag = legendTxt
    if IF_NEW_FIG:
	plt.figure()
    validTrList = []
    meanOSI = []
    meanOSI_I = []
    counter = 0
    validKappa = []
    for kk, kappa in enumerate(kappaList):
        p = kappa
	# legendTxt = ', K=%s'%(K) + legendTxtTag
	legendTxt = ', %s'%(neuronType) + legendTxtTag
	try:
	    print trNo
	    tc = GetTuningCurves(p, gamma, nPhis, mExt, mExtOne, rewireType, trNo, N, K, nPop, T, kappa)
	    osi = OSIOfPop(tc[:N, :], phis)
	    osiI = OSIOfPop(tc[N:, :], phis)
	    meanOSI.append(np.nanmean(osi))
	    meanOSI_I.append(np.nanmean(osiI))
	    if ~np.isnan(meanOSI[counter]):
		validTrList.append(trNo)
	    clrCntr += 1
	    counter += 1
	    validKappa.append(kappa)
	except IOError:
	    print "p = ", p, " gamma = ", gamma, " trial# ", trNo, " file not found"
    meanOSI = np.array(meanOSI)
    meanOSI_I = np.array(meanOSI_I)
    kappaList = np.asarray(kappaList, dtype = float)
    validIdx = ~np.isnan(meanOSI)
    validKappa = kappaList[validIdx]
    pcolor = color
    if color == '':
	pcolor = colors[kk]
    if neuronType == 'E':
	plt.plot(validKappa, meanOSI[validIdx], 'o-', color = pcolor, label = r'$m_0^{(1)}=%s$'%(mExtOne) + legendTxt, markeredgecolor = pcolor)
    else:
	plt.plot(validKappa, meanOSI_I[validIdx], 'o-', color = pcolor, label = r'$m_0^{(1)}=%s$'%(mExtOne) + legendTxt, markeredgecolor = pcolor)
    print '--'*26
    osiLast = meanOSI[-1]
    osilastCnt = -1
    osiLastI = meanOSI_I[-1]
    ipdb.set_trace()
    while np.isnan(osiLast):
        osilastCnt -= 1
        print osilastCnt
        osiLast = meanOSI[osilastCnt]	
        osiLastI = meanOSI_I[osilastCnt]
    print 'pc change in mean OSI = ', 100 * (osiLast - meanOSI[0]) / meanOSI[0]
    print '--'*26
    plt.gca().set_position([0.15, 0.15, .65, .65])
    plt.ylim(0, .5)
    xmax = max(validKappa) + 1
    xmin = min(validKappa) - 1
    plt.xlim(xmin, xmax)
    plt.xlim(0, 12)    
    plt.ylim(0, 1)
    plt.xlabel(r'$\kappa$')
    plt.ylabel(r'$\langle OSI \rangle$')
    plt.gca().set_position([0.25, 0.25, .65, .65])
    filename = filename + "p%sg%sk%s_K%s_"%(p, gamma, kappa, K) + neuronType
    filepath = "./figs/rewiring/compareOSI_mean_"+ rewireType + '_' +  filename
    print filepath
    paperSize = [4, 3]
    plt.legend(loc = 2, frameon = False, numpoints = 1, prop = {'size': 8})
    print 'saving as: ', filepath
    ProcessFigure(plt.gcf(), filename, IF_SAVE = 1, IF_XTICK_INT = False, figFormat = 'pdf')
    plt.show()
    print meanOSI

#def Plot


def PltOSIHist(p, gamma, nPhis, mExt, mExtOne, trNo = 0, N = 10000, K = 1000, nPop = 2, T = 1000, IF_NEW_FIG = True, color = 'k', IF_PLOT = True, neuronType = 'E'):
    NE = N
    NI = N
    tc = np.zeros((NE + NI, nPhis))
    phis = np.linspace(0, 180, nPhis, endpoint = False)
    if IF_NEW_FIG:
        plt.figure()
    try:
        rewireType = 'rand'
        tc = GetTuningCurves(p, gamma, nPhis, mExt, mExtOne, rewireType, trNo, N, K, nPop, T, kappa= 0, IF_SUCCESS = False)
        if neuronType == 'E':
            osi = OSIOfPop(tc[:NE, :], phis)
        else:
            osi = OSIOfPop(tc[NE:, :], phis)
        print "K = ", K, ", osi simulation: ", np.nanmean(osi)
        if IF_PLOT:
            plt.xlabel('OSI', fontsize = 12)    
            plt.ylabel('Density', fontsize = 12)
            plt.hist(osi[~np.isnan(osi)], 27, normed = 1, histtype = 'step', label = r'$p = %s. \gamma = %s$'%(p, gamma, ), color = color, lw = 1)    
            plt.xlim(0, 1)
            plt.gca().set_xticks([0, 0.5, 1])
            _, ymax = plt.ylim()
            plt.gca().set_yticks([0, np.ceil(ymax)])    
            plt.title(r'$m_0^{(0)} = %s, \,m_0^{(1)} = %s $'%(mExt, mExtOne))
            _, ymax = plt.ylim()
            print "mean OSI = ", np.nanmean(osi)
        return osi, OSIOfPop(tc[NE:, :], phis)
    except IOError:
        return np.nan
        pass


# global clrCntr
# clrCntr = 0
def CompareOSIHist(pList,  trNo = 0, mExt = 0.075, mExtOneList = [0.02],  nPhis = 8, N = 10000, KList = [1000], nPop = 2, T = 1000, IF_NEW_FIG = True, clrCntr = 0, filename = '', IF_LEGEND = True, legendTxt = '', neuronType = 'E', gList = [0]):
    if IF_NEW_FIG:
	plt.figure()
    # colors = [plt.cm.Dark2(i) for i in np.linspace(0, 1, 1 + clrCntr + len(pList) * len(gList) * len(mExtOneList), endpoint = False)]
    colors = ['k', 'g', 'r']
    for mExtOne in mExtOneList:
	for p in pList:
            if p == 0:
                trNo = 0
            else:
                trNo = 1
	    for gamma in gList:
		for K in KList:
		    try:
			PltOSIHist(p, gamma, nPhis, mExt, mExtOne, trNo = trNo, IF_NEW_FIG = False, color = colors[clrCntr], T=T, K=K, neuronType = neuronType)
			clrCntr += 1
		    except IOError:
			print "p = ", p, " gamma = ", gamma, " trial# ", trNo, " file not found"
    # plt.gca().legend(bbox_to_anchor = (1.1, 1.5))
    if IF_LEGEND:
	plt.legend(loc = 0, frameon = False, numpoints = 1, prop = {'size': 12})
    plt.gca().set_position([0.15, 0.15, .65, .65])
    # plt.title('%s'%(neuronType))
    if nPop == 2:
       filename = './figs/PUB_FIGS/' + 'OSIhist_vs_kappa_%s_K%s'%(neuronType, K)
       paperSize = [2.5, 2]
       axPosition=[.26, .24, .65, .65]
       ProcessFigure(plt.gcf(), filename, 1, paperSize = paperSize, axPosition = axPosition, titleSize=10, nDecimalsX=1, nDecimalsY=1, figFormat='svg', labelFontsize = 8, tickFontsize = 6)


def CompareMeanTc(pList,  trNo = 0, mExt = 0.075, mExtOneList = [0.02],  nPhis = 8, NE = 10000, KList = [1000], nPop = 2, T = 1000, IF_NEW_FIG = True, clrCntr = 0, filename = '', IF_LEGEND = True, legendTxt = '', neuronType = 'E', gList = [0]):
    if IF_NEW_FIG:
	plt.figure()
    # colors = [plt.cm.Dark2(i) for i in np.linspace(0, 1, 1 + clrCntr + len(pList) * len(gList) * len(mExtOneList), endpoint = False)]
    colors = ['k', 'g', 'r']
    clrCntr = -1
    for mExtOne in mExtOneList:
	for p in pList:
            clrCntr = clrCntr + 1
            if p == 0:
                trNo = 0
            else:
                trNo = 1
	    for gamma in gList:
		for K in KList:
		    try:
                        rewireType = 'rand'
                        tc = GetTuningCurves(p, gamma, nPhis, mExt, mExtOne, rewireType, trNo, NE, K, nPop, T, kappa= 0, IF_SUCCESS = False)
                        if neuronType == 'E':
                            tc = tc[:NE]
                        else:
                            tc = tc[NE:]
                        print 'color=', colors[clrCntr], clrCntr
			PlotMeanTc(tc, p, 1, nPhis, NE, '', pcolor = colors[clrCntr])

		    except IOError:
			print "p = ", p, " gamma = ", gamma, " trial# ", trNo, " file not found"
    # plt.gca().legend(bbox_to_anchor = (1.1, 1.5))
    if IF_LEGEND:
	plt.legend(loc = 0, frameon = False, numpoints = 1, prop = {'size': 12})
    plt.gca().set_position([0.15, 0.15, .65, .65])
    if neuronType == 'I':
        plt.xlabel('PO')
        plt.ylabel('Response')
    # plt.title('%s'%(neuronType))
    if nPop == 2:
       filename = './figs/PUB_FIGS/' + 'meanTc_vs_kappa_%s_K%s'%(neuronType, K)
       paperSize = [2.5, 2]
       axPosition=[.26, .24, .65, .65]
       ProcessFigure(plt.gcf(), filename, 1, paperSize = paperSize, axPosition = axPosition, titleSize=10, nDecimalsX=1, nDecimalsY=1, figFormat='svg', labelFontsize = 8, tickFontsize = 6)
       



def PlotMeanTc(tc, kappa, J = 1, nPhis = 8, NE = 10000, labelTxt='', pcolor='k'):
    # J is the rewired strenghtened prefactor
    # thetas = np.linspace(0, 180, nPhis, endpoint = 1)
    thetas = np.arange(-90, 90, 180.0/8)
    prefferedOri = np.argmax(tc, 1)
    tcmax = np.max(np.abs(tc), 1)
    # ipdb.set_trace()
    tcmax.shape = NE, 1
    tcmax = np.tile(tcmax, (1, nPhis))
    tc = tc / tcmax
    cvMat = np.empty((NE, len(thetas)))
    for kNeuron in np.arange(NE):
	cvMat[kNeuron, :] = np.roll(tc[kNeuron, :], -1 * prefferedOri[kNeuron])
    plt.ion()
    tmpE = cvMat #[plotId[plotId < NE], :]
    meanE = np.nanmean(tmpE, 0)
    meanE = np.roll(meanE, 4)
    osi = OSI(meanE, np.arange(0, 180, 22.5))
    print 'osi = ', osi
    thetas = np.arange(-90, 91, 22.5)
    print pcolor
    plt.plot(thetas, np.concatenate((meanE, [meanE[0]])), 'o-', label=labelTxt + ' osi: %.5s'%(osi), color = pcolor, markersize = 2, markeredgecolor = pcolor)
    # plt.ylim(0.8, 1)
    plt.gca().set_xticks([-90, 0, 90])

    filename = './figs/twopop/rewire/' + 'pop_mean_tuning_kappa%s'%(kappa)
    paperSize = [2.5, 2]
    axPosition=[.22, .22, .65, .65]
    plt.ylim([0, 1])
    plt.gca().set_yticks([0, 0.5, 1])
    ProcessFigure(plt.gcf(), filename, 1, paperSize = paperSize, axPosition = axPosition, titleSize=10, nDecimalsX=1, nDecimalsY=1, figFormat='svg', labelFontsize = 10, tickFontsize = 8)

       


def PlotTuningCurvesKappaNonZero(kappaList, NE = 10000, neuronType = 'E',  nPhis = 8, mExt = 0.075, mExtOne = 0.02, rewireType = 'rand', trNo = 0, K = 1000, nPop = 2, T = 1000, roll = False):
    thetas = np.arange(0., 181., 180./nPhis)
    # thetas = np.arange(-90, 91, 180./nPhis)    
    neuronsIdx = [1798, 7753, 3233, 6218]
    if neuronType == 'E':
        neuronsIdx = np.random.choice(NE, 100)
    else:
        neuronsIdx = np.random.choice(NE, 100) + NE
    print neuronsIdx
    filetag = neuronType
    colors = ['k', 'g', 'r']
    gamma = 0;
    tclist = []
    for kappaIdx, kappa in enumerate(kappaList):
        if kappa == 0:
            trNo = 0
        else:
            trNo = 1
        tclist.append(GetTuningCurves(kappa, gamma, nPhis, mExt, mExtOne, rewireType, trNo, NE, K, nPop, T, kappa= 0, IF_SUCCESS = False))
    
    for idx in neuronsIdx:    
        for kappaIdx, kappa in enumerate(kappaList):
            if kappa == 0:
                trNo = 0
            else:
                trNo = 1
            # tc = GetTuningCurves(kappa, gamma, nPhis, mExt, mExtOne, rewireType, trNo, NE, K, nPop, T, kappa= 0, IF_SUCCESS = False)
            tc = tclist[kappaIdx]
            # tc = tcList[kappaIdx]
            tmp = tc[idx, :]
            if roll:
                tmp = np.roll(tmp, -1 * np.argmax(tmp))
                tmp = np.roll(tmp, 4)        
            plt.plot(thetas, np.concatenate((tmp, [tmp[0]])), 'o-', color = colors[kappaIdx], lw = .5, markersize = 0.95, markerfacecolor = colors[kappaIdx], markeredgecolor=colors[kappaIdx])
        filename = './PNAS/figs/sim/mExtOne%s/tc_%s/tuning_curves_%s_idx_%s'%(int(1e3*mExtOne), neuronType, filetag, idx)
        paperSize = [2.5/2, 2/2.0]
        axPosition=[.25, .25, .65, .65]
        FixAxisLimits(plt.gcf())
        ymin, ymax = plt.ylim()
        plt.ylim(0, ymax)
        plt.gca().set_yticks([0, ymax * 0.5, ymax])
        plt.gca().set_yticklabels(['0', '%.4s'%(ymax * 0.5), '%.4s'%(ymax)])
        plt.gca().set_xticks([0, 90, 180])
        plt.gca().set_xticklabels(['0', '90', '180'])        
        plt.draw()
        ProcessFigure(plt.gcf(), filename, 1, paperSize = paperSize, axPosition = axPosition, titleSize=10, nDecimalsX=1, nDecimalsY=1, figFormat='svg', labelFontsize = 6, tickFontsize = 6)
        plt.clf()
    












