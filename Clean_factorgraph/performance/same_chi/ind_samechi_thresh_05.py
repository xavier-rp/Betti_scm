import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
if __name__ == '__main__':
    """
    x = 0.5, p = 0.47950012218695615
    N = 100

    [[ 7.51280963 20.        ]
 [15.         57.48719037]]

    N = 200

    [[ 13.42317634  40.        ]
 [ 30.         116.57682366]]

    N = 300

    [[ 19.12598277  60.        ]
 [ 45.         175.87401723]]

    N = 400

[[ 24.71979651  80.        ]
 [ 60.         235.28020349]]

    """


    #Independance vs N
    original_table = np.array([[7.51280963, 20], [15, 57.48719037]]) # ind_samechi_05_100
    original_table = np.array([[ 13.42317634, 40], [30, 116.57682366]]) # ind_samechi_05_200
    original_table = np.array([[19.12598277, 60], [45, 175.87401723]]) # ind_samechi_05_300
    original_table = np.array([[24.71979651, 80], [60, 235.28020349]]) # ind_samechi_05_400

    #DON'T MIND THE NAME OF THE VARIABLES, I WAS JUST LAZY AND REUSED A FILE
    ind_25 = np.load('ind_samechi_05_100.npy')
    ind_50 = np.load('ind_samechi_05_200.npy')
    ind_75 = np.load('ind_samechi_05_300.npy')
    ind_100 = np.load('ind_samechi_05_400.npy')

    rc('text', usetex=True)
    rc('font', size=16)

    plt.plot(np.arange(0, 2 * len(ind_25), 2), ind_25, marker='1', markeredgecolor='black', markerfacecolor='black',
             markersize='12', color='#00a1ffff', linewidth='3', label='N = 100')
    plt.plot(np.arange(0, 2 * len(ind_50), 2), ind_50, marker='3', markeredgecolor='black', markerfacecolor='black',
             markersize='12', color='#41d936ff', linewidth='3', label='N = 200')
    plt.plot(np.arange(0, 2 * len(ind_75), 2), ind_75, marker='2', markeredgecolor='black', markerfacecolor='black',
             markersize='12', color='#ff7f00ff', linewidth='3', label='N = 300')
    plt.plot(np.arange(0, 2 * len(ind_100), 2), ind_100, marker='x', markeredgecolor='black', markerfacecolor='black',
             markersize='10', color='#a1a1a1', linewidth='3', label='N = 400')

    plt.legend(loc=0)
    plt.xlabel('Distance $L_1$')
    plt.ylabel('Taux de succ\`es')
    # plt.grid()
    plt.xlim([0, 35])
    plt.ylim([0.5, 1.05])
    plt.show()