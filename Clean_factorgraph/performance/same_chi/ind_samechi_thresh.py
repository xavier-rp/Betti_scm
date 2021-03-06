import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
if __name__ == '__main__':
    """
    X = 2.500000000000001, p = 0.1138462980066579
    N = 100

    [[42.62613217 15.        ]
 [25.         17.37386783]]

    N = 200

    [[93.4870287 30.       ]
 [50.        26.5129713]]

    N = 300

    [[144.48792507  45.        ]
 [ 75.          35.51207493]]

    N = 400

    [[195.71877794  60.        ]
 [100.          44.28122206]]

    """


    #Independance vs N
    original_table = np.array([[42.62613217, 15], [25, 17.37386783]]) #'ind_samechi_100'
    original_table = np.array([[93.4870287, 30], [50, 26.5129713]]) # 'ind_samechi_200'
    original_table = np.array([[144.48792507, 45], [75, 35.51207493]]) # 'ind_samechi_300'
    original_table = np.array([[195.71877794,  60], [100, 44.28122206]]) # 'ind_samechi_400'

    #DON'T MIND THE NAME OF THE VARIABLES, I WAS JUST LAZY AND REUSED A FILE
    ind_25 = np.load('ind_samechi_100.npy')
    ind_50 = np.load('ind_samechi_200.npy')
    ind_75 = np.load('ind_samechi_300.npy')
    ind_100 = np.load('ind_samechi_400.npy')

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
    plt.xlim([0, 35])
    plt.ylim([0.5, 1.05])
    plt.show()