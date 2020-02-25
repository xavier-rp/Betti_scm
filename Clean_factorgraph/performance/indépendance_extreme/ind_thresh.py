import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':


    ind_25 = np.load('ind_25.npy')
    ind_50 = np.load('ind_50.npy')
    ind_75 = np.load('ind_75.npy')
    ind_100 = np.load('ind_100.npy')

    plt.plot(np.arange(0, 2*len(ind_25), 2), ind_25, marker='x', label='N = 100')
    plt.plot(np.arange(0, 2 * len(ind_50), 2), ind_50, marker='x', label='N = 200')
    plt.plot(np.arange(0, 2 * len(ind_75), 2), ind_75, marker='x', label = 'N = 300')
    plt.plot(np.arange(0, 2 * len(ind_100), 2), ind_100, marker='x', label = 'N = 400')
    plt.plot()
    plt.legend(loc=0)
    plt.xlabel('L1 norm')
    plt.ylabel('Success rate')

    plt.show()