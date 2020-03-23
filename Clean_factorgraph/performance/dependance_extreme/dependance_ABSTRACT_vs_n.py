import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc


if __name__ == '__main__':


    d_20_80 = np.load('dep_25_25.npy') # [50, 0, 0, 50]
    d_32_18 = np.load('dep_30_30.npy') # [60, 0, 0, 60]
    d_40_10 = np.load('dep_35_35.npy') # [70, 0, 0, 70]

    rc('text', usetex=True)
    rc('font', size=16)

    plt.plot(np.arange(0, 2*len(d_20_80), 2), d_20_80, marker='1', markeredgecolor='black', markerfacecolor='black', markersize='12', color='#00a1ffff', linewidth='3', label='N = 50')
    plt.plot(np.arange(0, 2 * len(d_32_18), 2), d_32_18, marker='3', markeredgecolor='black', markerfacecolor='black', markersize='12', color='#41d936ff', linewidth='3', label='N = 60')
    plt.plot(np.arange(0, 2 * len(d_40_10), 2), d_40_10, marker='2', markeredgecolor='black', markerfacecolor='black', markersize='12', color='#ff7f00ff', linewidth='3', label = 'N = 70')
    plt.legend(loc=0)
    plt.xlabel('Distance $L_1$')
    plt.ylabel('Taux de succ\`es')
    #plt.grid()
    plt.xlim([0,50])
    plt.ylim([0, 1.05])
    plt.show()