import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

if __name__ == '__main__':


    ind_25 = np.load('ind_25.npy')
    ind_50 = np.load('ind_50.npy')
    ind_75 = np.load('ind_75.npy')
    ind_100 = np.load('ind_100.npy')

    rc('text', usetex=True)
    rc('font', size=16)

    plt.plot(np.arange(0, 2 * len(ind_25), 2), ind_25, marker='1', markeredgecolor='black', markerfacecolor='black', markersize='12', color='#00a1ffff', linewidth='3',
             label='N = 100')
    plt.plot(np.arange(0, 2 * len(ind_50), 2), ind_50, marker='3', markeredgecolor='black', markerfacecolor='black', markersize='12', color='#41d936ff', linewidth='3',
             label='N = 200')
    plt.plot(np.arange(0, 2 * len(ind_75), 2), ind_75, marker='2', markeredgecolor='black', markerfacecolor='black', markersize='12', color='#ff7f00ff', linewidth='3',
             label='N = 300')

    #plt.plot(np.arange(0, 2 * len(ind_25), 2), ind_25, marker='x', color='#00a1ffff', linewidth='2',
    #         label='N = 100')
    #plt.plot(np.arange(0, 2 * len(ind_50), 2), ind_50, marker='x', color='#41d936ff', linewidth='2',
    #     label='N = 200')
    #plt.plot(np.arange(0, 2 * len(ind_75), 2), ind_75, marker='x', color='#ff7f00ff', linewidth='2',
    #     label='N = 300')
    plt.legend(loc=0)
    plt.xlabel('Distance $L_1$')
    plt.ylabel('Taux de succ\`es')
    # plt.grid()
    plt.xlim([0, 100])
    plt.ylim([0, 1.05])
    plt.show()
    exit()

    plt.plot(np.arange(0, 2*len(ind_25), 2), ind_25, marker='x', label='N = 100')
    plt.plot(np.arange(0, 2 * len(ind_50), 2), ind_50, marker='x', label='N = 200')
    plt.plot(np.arange(0, 2 * len(ind_75), 2), ind_75, marker='x', label = 'N = 300')
    plt.plot(np.arange(0, 2 * len(ind_100), 2), ind_100, marker='x', label = 'N = 400')
    plt.plot()
    plt.legend(loc=0)
    plt.xlabel('L1 norm')
    plt.ylabel('Success rate')

    plt.show()