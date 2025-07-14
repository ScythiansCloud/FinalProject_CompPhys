import settings.settings_task3 as settings
import numpy as np
import matplotlib.pyplot as plt

def potential(r):
    if r < settings.sig+ settings.delta:
        return 8*settings.kBT* ((0.6/r)**12-(0.6/r)**6)
    else:
        return -settings.A*settings.sig /(24*(r-settings.sig)) + settings.Zprimesqrd* settings.lambda_B/r*np.exp(-settings.kappa_D*r)
    

def plotpot():

    Csi = [10,100,333,666,1000]
    bins = 100

    v10 = np.zeros(bins)
    v100 = np.zeros(bins)
    v333 = np.zeros(bins)
    v666 = np.zeros(bins)
    v1000 = np.zeros(bins)

    vs = [v10,v100, v333, v666, v1000]

    

    for Cs, vi in zip(Csi, vs):
        settings.init(Cs)
        print(settings.Cs, settings.kappa_D)

        r = np.linspace(0.01, settings.L/3, bins)

        for ri, i in zip(r, range(bins)):
            vi[i] = potential(ri)
    
    plt.figure





