import numpy as np
import PltFuncs as p
import ipdb
import pylab as plt



def KappaVsM1(kappaList, NList = [80000], nTrials=10, T=1000, K=1000):
    # saves x and y in pnas/data
    for N in NList:
        p.KappaVsM1(kappaList, mExtOne=0, IF_COMPUTE=1, T=T, nTrials=nTrials, nPhis=1, N=N, K=K)


if __name__ == "__main__":
    # T=4000
    # kappaList = np.arange(0, 10, 0.1)
    kappaList = np.array([8.2, 8.6, 8.8, 9.0])
    # kappaList = np.array([8.8])    
    # kappaList = np.array([4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5])
    # kappaList = np.array([7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2])
    nTrials = 10
    NList = [20000] #, 80000] #, 20000, 40000]
    K = 2000


    for T in [2000, 4000]:
        KappaVsM1(kappaList, NList, nTrials, T=T, K=K)
        p.KappaVsM1Plot(NList, kappaList, T=T, K=K)
        #p.KappaVsM1PlotPoints(NList, kappaList, T=T)        

    # plt.ion()
    # plt.show()
    # ipdb.set_trace()
