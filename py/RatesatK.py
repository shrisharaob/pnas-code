import sys
from scipy.optimize import fsolve
import numpy as np
from scipy.integrate import quad
from scipy.special import erfc, erfcinv
import pylab as plt
import sys
basefolder = "/homecentral/srao/Documents/code/mypybox"
sys.path.append(basefolder)
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf
import ipdb

def alphaE(m0, meanFieldRates):
    return JEE**2 * meanFieldRates[0] + JEI**2 * meanFieldRates[1] + (1 / cFF) * JE0**2 * m0

def alphaI(m0, meanFieldRates):
    return JIE**2 * meanFieldRates[0] + JII**2 * meanFieldRates[1] + (1 / cFF) * JI0**2 * m0 
    

def MyErfc(z):
    # normal CDF
    return 0.5 * erfc(z / np.sqrt(2.0))

def PrintError(outStruct, varName):
    print "-----------------------------------"
    print "Variable        :",  varName
    print "Solution        :", outStruct[0]
    print "error           :", outStruct[1]['fvec']
    print "Solution Status :", outStruct[-1]
    print "-----------------------------------"    
    
def CheckBalConditions(JEE, JEI, JIE, JII, JE0, JI0):
    JE = -JEI/JEE
    JI = -JII/JIE
    E = JE0
    I = JI0
    if((JE < JI) or (E/I < JE/JI) or (E/I < 1) or (JE < 1)):
        print "NOT IN BALANCED REGIME!!!!!! "
        raise SystemExit
        
def HPrime(z):
    return -1.0 * np.exp(-0.50 * z**2) / np.sqrt(2.0 * np.pi)


def Pcritical(uE0, alpha, JEE):
    b = (1 - uE0) / np.sqrt(alpha)
    C0 = np.exp(-b**2 / 2.0) / np.sqrt(2.0 * np.pi)
    num = np.sqrt(alpha)
    denom = C0 * JEE
    denum2 = C0 * (JEE + JEE**2 / (2.0 * alpha * np.sqrt(K)))
    print "new fin k corrected p_c = ", num / denum2
    return num / denom

def QFuncInv(z):
    return np.sqrt(2.0) * erfcinv(z * 2.0)
    
def PcriticalFiniteK(uE0, alphaE0, JEE, JEI, hPrime, K, A):
    b = (1 - uE0) / np.sqrt(alphaE0)
    num = -1.0 * np.sqrt(alphaE0)
    denomMatlab = hPrime * (JEE + JEE**2 / (2.0 * alphaE0 * np.sqrt(float(K))))
    denom = hPrime * (JEE + JEE**2 * A / (2.0  * np.sqrt(float(K) * alphaE0)))
    print "New corrected p_c = ", num / denom
    print 'denom matlab = ' , denomMatlab
    return num / denom

def UE0GivenUE1(u0Guess, *args):
    # solve for u_E(0) given U_E(1)
    u0 = u0Guess
    jee, alpha, u1, meanFieldRate = args
    funcME0 = lambda phi: (1.0 / np.pi) * MyErfc((1.0 - u0 - u1 * np.cos(2.0 * phi)) / np.sqrt(alpha))
    return meanFieldRate - quad(funcME0, 0, np.pi)[0]

def Eq0(u0Guess, *args):
    # solve for u_E(0)
    u0 = u0Guess
    jee, alpha, me0 = args
    return me0 - MyErfc((1.0 - u0) / np.sqrt(alpha))

def EqFiniteK(u0Guess, *args):
    # solve for u_E(0), u_I(0), u_E(0) at p
    uE0, uI0, mE1 = u0Guess
    K, p, m_ext = args
    detJab = np.linalg.det(Jab)
    sqrtK = np.sqrt(K)

    uTildeE0 = uE0 / sqrtK - JE0 * m_ext  # * cFF
    uTildeI0 = uI0 / sqrtK - JI0 * m_ext  # * cFF

    mE0 = ( JII * uTildeE0 - JEI * uTildeI0) / detJab
    mI0 = (-JIE * uTildeE0 + JEE * uTildeI0) / detJab
    uE1 = p * JEE * mE1
    
    aE0 = alphaE(m0, [mE0, mI0])
    aI0 = alphaI(m0, [mE0, mI0])    

    aE1 = p * JEE**2 * mE1 / sqrtK
    aI1 = 0

    #p * JIE**2 * mE1 / sqrtK
    # if(aE1 > aE0):
    #     aE1 = aE0 * 0.1

    AEofPhi = lambda phi: aE0 + aE1 * np.cos(2.0 * phi)
    AIofPhi = lambda phi: aI0 + aI1 * np.cos(2.0 * phi)    

    funcME0 = lambda phi: (1.0 / np.pi) * MyErfc((1.0 - uE0 - uE1 * np.cos(2.0 * phi)) / np.sqrt(AEofPhi(phi)))
    funcMI0 = lambda phi: (1.0 / np.pi) * MyErfc((1.0 - uI0) / np.sqrt(AIofPhi(phi)))
    funcME1 = lambda phi: (2.0 * np.cos(2.0 * phi) / np.pi) * MyErfc((1.0 - uE0 - uE1 * np.cos(2.0 * phi)) / np.sqrt(AEofPhi(phi)))
    # print uE0, uI0, AEofPhi(0)
    a = mE0 - quad(funcME0, 0, np.pi, limit = 200, epsabs = 1e-10)[0]
    b = mI0 - np.abs(quad(funcMI0, 0, np.pi, limit = 200, epsabs = 1e-10)[0])
    c = mE1 - np.abs(quad(funcME1, 0, np.pi, limit = 200, epsabs = 1e-10)[0])
    # print mE0, mE1, mI0
    return (a, b, c)



def SolveForPCritical(m0List):  # for large K
    nPoints = 100
    # uE1List = np.linspace(1e-6, 1.0, nPoints)
    uE1List = np.linspace(-10, 10.0, nPoints)    
    for l, lM0 in enumerate(m0List):
        meanFieldRates = -1.0 * np.dot(np.linalg.inv(Jab), Ea) * lM0
        alpha_E = alphaE(lM0, meanFieldRates)
        #============================================ 
        uE0Solution = fsolve(Eq0, [0.1], args = (JEE, alphaE(lM0, meanFieldRates), meanFieldRates[0]), full_output = 1)    
        PrintError(uE0Solution, 'u_E(0)')
        uE0 = uE0Solution[0]
        pCAnalytic = Pcritical(uE0, alpha_E, JEE)
        # PcriticalCorrected(uE0, alpha_E, JEE);
        # raise SystemExit
        #------------------------------------------
        uE0List = []
        mE1List = []
        pList = []
        for i, iUE1 in enumerate(uE1List):
            uE0Sol = fsolve(UE0GivenUE1, [0.1], args = (JEE, alpha_E, iUE1, meanFieldRates[0]), full_output = True, xtol = 1e-6)
            if(uE0Sol[-2] == 1): # .i.e IF SOLUTION CONVERGED
                FuncME0 = lambda phi: (1.0 / np.pi) * MyErfc((1.0 - uE0Sol[0][0] - iUE1 * np.cos(2.0 * phi)) / np.sqrt(alpha_E))
                # PrintError(uE0Sol, 'U_E(0)[U_E(1)]')            
                print uE0Sol[0][0], iUE1, quad(FuncME0, 0, np.pi)[0], uE0Sol[-1], uE0Sol[-2]
                uE0List.append((uE0Sol[0][0], iUE1))
                FuncME1 = lambda phi: (2.0 / np.pi) * np.cos(2.0 * phi) * MyErfc((1.0 - uE0Sol[0][0] - iUE1 * np.cos(2.0 * phi)) / np.sqrt(alpha_E))
                me1 = quad(FuncME1, 0, np.pi)[0]
                if(me1 <= meanFieldRates[0] and iUE1 > 0):
                    pList.append(iUE1 / (JEE * me1))
                    mE1List.append(me1)
        if(lM0 == m0List[-1]):
            plt.plot(pList, np.array(mE1List, dtype='float') / lM0, 'bo-', markersize = 4, label = 'numeric')
            plt.vlines(pCAnalytic, 0, meanFieldRates[0], 'k', label = 'analytic')
            plt.legend(loc = 0, frameon = False, numpoints = 1)
            plt.xlabel('p')
            plt.ylabel(r'$m_E^{(1)}$')
            # plt.xlim(3.5, 5.5)
            # plt.ylim(0, 0.3)
            plt.text(pList[-1], mE1List[-1], '%s'%(lM0))            
#            plt.show()
        else:
            plt.plot(pList, np.array(mE1List, dtype = 'float') / lM0, 'bo-', markersize = 4)
            plt.vlines(pCAnalytic, 0, meanFieldRates[0], 'k')
        if(l == 0):
            plt.text(pList[-1], mE1List[-1], r'$m_0 = %s$'%(lM0))
        else:
            plt.text(pList[-1], mE1List[-1], '%s'%(lM0))
    print "--" * 25
    print "m0 = ", m0List[0], "p_critical = ", pCAnalytic[0]
    print "--" * 24    
    np.save('./data/ulist_p_vs_mI1_m0%s'%(int(lM0 * 1e3)), [pList, mE1List, np.ones((len(pList), )) * meanFieldRates[0], np.ones((len(pList), )) * meanFieldRates[1]])  

    

if __name__ == "__main__":
    mIIncreaseFactor = 1.0
    otherFactor = 1.0
    JEE = 1.0 * otherFactor
    JIE = 1.0 * otherFactor #0.7 #1.0 
    JEI = -1.5 / mIIncreaseFactor
    JII = -1.1450 / mIIncreaseFactor
    
    JE0 = 2.0 * otherFactor * 9 * 0.1
    JI0 = 1.350 * otherFactor * 9 * 0.1
    Jab = np.array([[JEE, JEI],
                    [JIE, JII]])
    Ea = np.array([JE0, JI0])
    gamma = 0.0
    m0 = 0.075
    
    p = float(sys.argv[1])    
    K = int(sys.argv[2])
    cFF = float(sys.argv[3])
    
    CheckBalConditions(JEE, JEI, JIE, JII, JE0, JI0)


    print '-' * 50
    print '-' * 15, ' ' * 2, 'large K', ' ' * 2,  '-' * 15
    meanFieldRates = -1.0 * np.dot(np.linalg.inv(Jab), Ea) * m0
    print "mf rates == ", meanFieldRates
    print "alphaE mf == ", alphaE(m0, meanFieldRates) 
    aE0 = alphaE(m0, meanFieldRates)
    aI0 = alphaI(m0, meanFieldRates)
    sqrtK = np.sqrt(K)
    uE0Sol = fsolve(Eq0, [1e-2], args = (JEE, aE0, meanFieldRates[0]), full_output = True, xtol = 1e-6)
    uI0Sol = fsolve(Eq0, [1e-2], args = (JII, aI0, meanFieldRates[1]), full_output = True, xtol = 1e-6)
    u0Guess = [uE0Sol[0][0], uI0Sol[0][0], 0.2613249636852059 ]
#    u0Guess = [uE0Sol[0][0], uI0Sol[0][0], 1e-6]    
    print "initial guess = ", u0Guess
    print "  "

    print '-' * 50
    print '-' * 15, ' ' * 2, 'fin K', ' ' * 2,  '-' * 15
    print '-' * 50
    print ''
    uE0Sol = fsolve(EqFiniteK, u0Guess, args = (K, p, m0), full_output = True, xtol = 1e-12, epsfcn = 1e-12)
    PrintError(uE0Sol, 'u_E(0),  u_I(0), m_E(1)]')
    uE0, uI0, mE1 = uE0Sol[0]
    detJab = np.linalg.det(Jab)
    sqrtK = np.sqrt(K)
    uTildeE0 = uE0 / sqrtK - JE0 * m0
    uTildeI0 = uI0 / sqrtK - JI0 * m0
    mE0 = ( JII * uTildeE0 - JEI * uTildeI0) / detJab
    mI0 = (-JIE * uTildeE0 + JEE * uTildeI0) / detJab
    aE0 = alphaE(m0, [mE0, mI0])
    aI0 = alphaI(m0, [mE0, mI0])    
    

    hPrimeIn =  HPrime((1.0 - uE0)/ np.sqrt(aE0))
    pCriticalAtK = PcriticalFiniteK(uE0, aE0, JEE, JEI, hPrimeIn, K, (1.0 - uE0)/ np.sqrt(aE0))
    print "m0 = ", m0
    print 'alphaE0 = ', aE0, 'hprime = ', hPrimeIn
    # raise SystemExit
    print ''
    print 'x'*15, ' rates at K solutions', ' ', 'x'*14
    print "rates at K = ", K, ' c = ', cFF
    print ''
    print "          me0            mi0            me1"
    print mE0, mI0, mE1
    print ''
    print " k_c for K = ", pCriticalAtK
    print ''
    print '...', Pcritical(uE0, aE0, JEE) #, Pcritical(uE0, aE0 - m0**2, JEE)

    ###########################################################################################
    compute = 0

    if compute:
        print "numerics ..."
        pStart = pCriticalAtK + 3
        pList = np.linspace(pStart, pCriticalAtK - 1, 5000)
        nPoints = len(pList)
        print 'nPoints = ', nPoints
        mE1SolList = []
        mE0SolList = []
        mI0SolList = []        
        pValidList = []
        uE0SolList = []
        uI0SolList = []
        muE = []
        for ip, p in enumerate(pList):
            if ip == 0:
                uE0Sol = fsolve(EqFiniteK, u0Guess, args = (K, p, m0), full_output = True, xtol = 1e-10, epsfcn = 1e-10)
            else:
                if not (len(uE0SolList) == 0 or len(uI0SolList) == 0 or len(mE1SolList) == 0):
                   u0Guess = [uE0SolList[-1], uI0SolList[-1], np.abs(mE1SolList[-1]) * 1e-1]
                uE0Sol = fsolve(EqFiniteK, uE0Sol[0], args = (K, p, m0), full_output = True, xtol = 1e-10, epsfcn = 1e-10)
            # if solverType == 'fixed-point-iter':
            #     uE0Sol = IterativeSolution(nIterations, uE0Sol[0], (K, p, m0), IF_VERBOSE = True)
            #     uE0, uI0, mE1 = uE0Sol
            #     IF_CONVERGED = ~np.isnan(uE0) and ~np.isnan(uI0) and ~np.isnan(mE1)
            # else:
            uE0, uI0, mE1 = uE0Sol[0]
            IF_CONVERGED = uE0Sol[-2] == 1
            uTildeE0 = uE0 / sqrtK - JE0 * m0
            uTildeI0 = uI0 / sqrtK - JI0 * m0
            mE0 = ( JII * uTildeE0 - JEI * uTildeI0) / detJab
            mI0 = (-JIE * uTildeE0 + JEE * uTildeI0) / detJab

            # aE0 = JEE**2 * mE0 + JEI**2 * mI0
            # aI0 = JIE**2 * mE0 + JII**2 * mI0    

            aE0 = alphaE(m0, [mE0, mI0])
            aI0 = alphaI(m0, [mE0, mI0])    

            if(IF_CONVERGED): # .i.e IF SOLUTION CONVERGED
                uE0SolList.append(uE0)
                uI0SolList.append(uI0)            
                mE0SolList.append(mE0)
                mI0SolList.append(mI0)
                mE1SolList.append(mE1)            
                pValidList.append(p)
                muE.append(mE1 / mE0)
        print "done"   

        muE  =  np.array(muE, dtype = 'float')
        vidx = muE <= 1
        pValidList = np.array(pValidList, dtype = 'float')
        mE0SolList = np.array(mE0SolList, dtype = 'float')
        mE1SolList= np.array(mE1SolList, dtype = 'float')

        out = {}
        out['muE'] = muE[vidx]
        out['kappa'] = pValidList[vidx]
        out['mE0'] = mE0SolList[vidx]
        out['mE1'] = mE1SolList[vidx]


        # np.save('./data/mE1_m0%s_K%s'%(int(m0 * 1e3), K), [pValidList, np.array(mE1SolList, dtype = 'float'), np.array(mE0SolList, dtype = 'float')])

        np.save('./data/mE1_m0%s_K%s_cff_%s'%(int(m0 * 1e3), K, int(cFF*1e3)), out)

        # plt.plot(pValidList, np.array(muE, dtype = 'float'), 'c.', label = 'corrected', lw = 1.5)
        plt.plot(out['kappa'], out['muE'], 'k')
    else:
        
        out = np.load('./data/mE1_m0%s_K%s_cff_%s.npy'%(int(m0 * 1e3), K, int(cFF*1e3)))[()]  # loading dict datatype 
        plt.plot(out['kappa'], out['muE'], 'k')
        plt.show()


    # plt.figure()
    # SolveForPCritical([0.075])
    # plt.ion()            
    # plt.show()
    # ipdb.set_trace()
