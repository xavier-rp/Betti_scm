import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

if __name__ == '__main__':


    ind_25 = np.load('dep_15.npy')
    ind_50 = np.load('dep_20.npy')
    ind_75 = np.load('dep_25.npy')
    #ind_100 = np.load('ind_100.npy')

    rc('text', usetex=True)
    rc('font', size=16)


    #plt.plot(np.arange(0, 2*len(np.mean(ind_25, axis=0)[0: 16]), 2), np.mean(ind_25, axis=0)[0:16], marker='1', markersize='15', color='#00a1ffff', linewidth='3', label='N = 60')
    #plt.fill_between(np.arange(0, 2*len(np.mean(ind_25, axis=0)[0: 16]), 2), np.mean(ind_25, axis=0)[0: 16] + np.std(ind_25, axis=0)[0: 16],
    #                 np.mean(ind_25, axis=0)[0: 16] - np.std(ind_25, axis=0)[0: 16], alpha=0.2)

    plt.plot(np.arange(0, 2 * len(np.mean(ind_25, axis=0)[0:]), 2), np.mean(ind_25, axis=0)[0:], marker='1', color='#00a1ffff', linewidth='2', markeredgecolor='black', markerfacecolor='black', markersize='12', label='N = 60')
    plt.fill_between(np.arange(0, 2 * len(np.mean(ind_25, axis=0)[0:]), 2),
                     np.mean(ind_25, axis=0)[0:] + np.std(ind_25, axis=0)[0:],
                     np.mean(ind_25, axis=0)[0:] - np.std(ind_25, axis=0)[0:], color='#00a1ffff', alpha=0.2)


    #plt.plot(np.arange(0, 2 * len(np.mean(ind_25, axis=0)[0: 21]), 2), np.mean(ind_50, axis=0)[0: 21],  marker='3', markersize='15', color='#41d936ff', linewidth='3',
    #          label='N = 80')

    #plt.fill_between(np.arange(0, 2 * len(np.mean(ind_25, axis=0)), 2), np.mean(ind_50, axis=0) + np.std(ind_50, axis=0),
    #                 np.mean(ind_50, axis=0) - np.std(ind_50, axis=0), alpha=0.2)

    plt.plot(np.arange(0, 2 * len(np.mean(ind_25, axis=0)[0:]), 2), np.mean(ind_50, axis=0)[0:], marker='3', color='#41d936ff', markeredgecolor='black', markerfacecolor='black', markersize='12', linewidth='2',
             label='N = 80')

    plt.fill_between(np.arange(0, 2 * len(np.mean(ind_25, axis=0)), 2), np.mean(ind_50, axis=0) + np.std(ind_50, axis=0),
                 np.mean(ind_50, axis=0) - np.std(ind_50, axis=0), color='#41d936ff', alpha=0.2)

    #plt.plot(np.arange(0, 2 * len(np.mean(ind_75, axis=0)[0: 26]), 2), np.mean(ind_75, axis=0)[0: 26], marker='2', markersize='15', color='#ff7f00ff', linewidth='3', label = 'N = 100')

    #plt.fill_between(np.arange(0, 2 * len(np.mean(ind_75, axis=0)), 2), np.mean(ind_75, axis=0) + np.std(ind_75, axis=0),
    #                 np.mean(ind_75, axis=0) - np.std(ind_75, axis=0), alpha=0.2)

    plt.plot(np.arange(0, 2 * len(np.mean(ind_75, axis=0)[0:]), 2), np.mean(ind_75, axis=0)[0:], marker='2', color='#ff7f00ff', linewidth='2', markeredgecolor='black', markerfacecolor='black', markersize='12', label='N = 100')

    plt.fill_between(np.arange(0, 2 * len(np.mean(ind_75, axis=0)[0:]), 2), np.mean(ind_75, axis=0)[0:] + np.std(ind_75, axis=0)[0:],
                     np.mean(ind_75, axis=0)[0:] - np.std(ind_75, axis=0)[0:], color='#ff7f00ff', alpha=0.2)


    #plt.plot(np.arange(0, 2 * len(ind_100), 2), ind_100, marker='x', label = 'N = 400')
    plt.legend(loc=0)
    plt.xlabel('Distance $L_1$')
    plt.ylabel('Taux de succ\`es')
    # plt.grid()
    plt.xlim([0, 50])
    plt.ylim([0, 1.05])
    plt.show()
