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
    bins = 10000

    v10 = np.zeros(bins)
    v100 = np.zeros(bins)
    v333 = np.zeros(bins)
    v666 = np.zeros(bins)
    v1000 = np.zeros(bins)

    vs = [v10,v100, v333, v666, v1000]

    colors = ['red', 'green', 'purple', 'gold','blue']

    for Cs, vi in zip(Csi, vs):
        
        settings.init(Cs)
        print(settings.Cs, settings.kappa_D)

        r = np.linspace(0.01, settings.L/3, bins)

        for ri, i in zip(r, range(bins)):
            vi[i] = potential(ri)
    
    r = np.linspace(0.01, settings.L/3, bins)
    
    plt.figure()
    for i in range(len(vs)):
        plt.plot(r, vs[i], label = f'Cs = {Csi[i]}', color = colors[i], zorder= i, lw=1 )
    
    plt.ylabel(r'$V_{VDLO}$ [k$_B$T]')
    plt.xlabel(r'r [$\sigma$]')

    plt.ylim([-2.5, 8])
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.savefig('largepot.png', bbox_inches='tight', dpi=300)

    plt.figure()
    for i in range(len(vs)):
        plt.plot(r, vs[i], label = f'Cs = {Csi[i]}', color = colors[i], zorder= i )
    
    # plt.ylabel(r'$V_{VDLO}$ [k$_B$T]')
    # plt.xlabel(r'r [$\sigma$]')

    plt.ylim([-0.3, 2])
    plt.xlim([0.95,2])
    #plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.savefig('zoomedpot.png', bbox_inches='tight', dpi=300)

    print(r[1500])
    peakmaxima= []
    for i in range(len(vs)):
        peakmaxima.append(np.max(vs[i][1500:]))
    
    plt.figure()
    plt.plot(Csi, peakmaxima, 'x', color='black', lw=1, label='Barrier-height', ls='-')
    plt.xlabel(r'$C_s$ [$\sigma^{-3}$]')
    plt.ylabel(r'Potential [k$_B$T]')
    plt.legend()
    plt.savefig('peaks.png', bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    print('hi')
    plotpot()





