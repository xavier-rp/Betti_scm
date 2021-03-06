import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
if __name__ == '__main__':
    """
    Chi = 10, pval = 0.0015654022580048148à

    N = 100
    [[63.69416035 15.        ]
 [10.         11.30583965]]

    N = 200

    [[135.06281265  30.        ]
 [ 20.          14.93718735]]

    N = 300

    [[206.49477294  45.        ]
 [ 30.          18.50522706]]

    N = 400

    [[278.09251464  60.        ]
 [ 40.          21.90748536]]

    """


    original_table = np.array([[63.69416035, 15], [10, 11.30583965]]) #dep_100
    original_table = np.array([[135.06281265, 30], [ 20, 14.93718735]]) # dep_200
    original_table = np.array([[206.49477294, 45], [30, 18.50522706]]) # dep_300
    original_table = np.array([[278.09251464, 60], [40, 21.90748536]]) # dep_400

    #DON'T MIND THE NAME OF THE VARIABLES, I WAS JUST LAZY AND REUSED A FILE
    ind_25 = np.load('dep_100.npy')
    ind_50 = np.load('dep_200.npy')
    ind_75 = np.load('dep_300.npy')
    ind_100 = np.load('dep_400.npy')

    rc('text', usetex=True)
    rc('font', size=16)

    plt.plot(np.arange(0, 2*len(ind_25), 2), ind_25, marker='1', markeredgecolor='black', markerfacecolor='black', markersize='12', color='#00a1ffff', linewidth='3', label='N = 100')
    plt.plot(np.arange(0, 2 * len(ind_50), 2), ind_50, marker='3', markeredgecolor='black', markerfacecolor='black', markersize='12', color='#41d936ff', linewidth='3',label='N = 200')
    plt.plot(np.arange(0, 2 * len(ind_75), 2), ind_75, marker='2', markeredgecolor='black', markerfacecolor='black', markersize='12', color='#ff7f00ff', linewidth='3',label = 'N = 300')
    plt.plot(np.arange(0, 2 * len(ind_100), 2), ind_100, marker='x', markeredgecolor='black', markerfacecolor='black', markersize='10', color='#a1a1a1', linewidth='3', label = 'N = 400')
    plt.legend(loc=0)
    plt.xlabel('Distance $L_1$')
    plt.ylabel('Taux de succ\`es')
    #plt.grid()
    plt.xlim([0, 20])
    plt.ylim([0.5, 1.05])
    plt.show()