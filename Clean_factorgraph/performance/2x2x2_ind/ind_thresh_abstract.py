import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

if __name__ == '__main__':


    ind_25 = np.load('ind_13_2.npy')

    #print(np.mean(ind_25, axis=0))
    #print(np.std(ind_25, axis=0))
    #plt.fill_between(np.arange(0, 42, 2), np.mean(ind_25, axis=0)+ np.std(ind_25, axis=0), np.mean(ind_25, axis=0) - np.std(ind_25, axis=0), alpha=0.2)

    #plt.show()

    ind_50 = np.load('ind_19_2.npy')
    ind_75 = np.load('ind_25_2.npy')
    #ind_100 = np.load('ind_100.npy')
    rc('text', usetex=True)
    rc('font', size=16)

    plt.plot(np.arange(0, 2*len(np.mean(ind_25, axis=0)), 2), np.mean(ind_25, axis=0), marker='1',  markeredgecolor='black', markerfacecolor='black', markersize='12', color='#00a1ffff', linewidth='2', label='N = 104')
    plt.fill_between(np.arange(0, 2*len(np.mean(ind_25, axis=0)), 2), np.mean(ind_25, axis=0) + np.std(ind_25, axis=0),
                     np.mean(ind_25, axis=0) - np.std(ind_25, axis=0), color='#00a1ffff', alpha=0.2)


    plt.plot(np.arange(0, 2 * len(np.mean(ind_50, axis=0)), 2), np.mean(ind_50, axis=0),   marker='3', markeredgecolor='black', markerfacecolor='black', markersize='12', color='#41d936ff', linewidth='2', label='N = 152')

    plt.fill_between(np.arange(0, 2 * len(np.mean(ind_50, axis=0)), 2), np.mean(ind_50, axis=0) + np.std(ind_50, axis=0),
                     np.mean(ind_50, axis=0) - np.std(ind_50, axis=0), color='#41d936ff', alpha=0.2)

    plt.plot(np.arange(0, 2 * len(np.mean(ind_75, axis=0)), 2), np.mean(ind_75, axis=0), marker='2', markeredgecolor='black', markerfacecolor='black', markersize='12', color='#ff7f00ff', linewidth='2', label = 'N = 200')

    plt.fill_between(np.arange(0, 2 * len(np.mean(ind_75, axis=0)), 2), np.mean(ind_75, axis=0) + np.std(ind_75, axis=0),
                     np.mean(ind_75, axis=0) - np.std(ind_75, axis=0), color='#ff7f00ff', alpha=0.2)


    #plt.plot(np.arange(0, 2 * len(ind_100), 2), ind_100, marker='x', label = 'N = 400')
    plt.legend(loc=0)
    plt.xlabel('Distance $L_1$')
    plt.ylabel('Taux de succ\`es')
    # plt.grid()
    plt.xlim([0, 100])
    plt.ylim([0, 1.05])
    plt.show()