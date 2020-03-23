import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

if __name__ == '__main__':


    ind_25 = np.load('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/performance/exact_vs_asympt/dep_10_10.npy')


    ind_50 = np.load('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/performance/exact_vs_asympt/dep_10_10_exact.npy')


    rc('text', usetex=True)
    rc('font', size=16)

    plt.plot(np.arange(0, 2 * len(ind_50), 2), ind_50, marker='2', markeredgecolor='black', markerfacecolor='black', markersize='12', color='#00a1ffff', linewidth='3', label='Exacte')

    plt.plot(np.arange(0, 2*len(ind_25), 2), ind_25, marker='1', markeredgecolor='black', markerfacecolor='black', markersize='12', color='#ff7f00ff', linewidth='3', label='Asymptotique')

    plt.legend(loc=0)
    plt.xlabel('Distance $L_1$')
    plt.ylabel('Taux de succ\`es')
    plt.xlim([0, 12])
    plt.ylim([0, 1.05])

    plt.show()