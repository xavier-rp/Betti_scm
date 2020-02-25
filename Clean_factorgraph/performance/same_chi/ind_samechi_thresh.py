import matplotlib.pyplot as plt
import numpy as np

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

    plt.plot(np.arange(0, 2*len(ind_25), 2), ind_25, marker='x', label='N = 100')
    plt.plot(np.arange(0, 2 * len(ind_50), 2), ind_50, marker='x', label='N = 200')
    plt.plot(np.arange(0, 2 * len(ind_75), 2), ind_75, marker='x', label = 'N = 300')
    plt.plot(np.arange(0, 2 * len(ind_100), 2), ind_100, marker='x', label = 'N = 400')
    plt.legend(loc=0)
    plt.xlabel('L1 norm')
    plt.ylabel('Success rate')

    plt.show()