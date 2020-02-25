import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':


    d_20_80 = np.load('dep_50_50.npy') # [50, 0, 0, 50]
    d_32_18 = np.load('dep_60_60.npy') # [60, 0, 0, 60]
    d_40_10 = np.load('dep_70_70.npy') # [70, 0, 0, 70]



    plt.plot(np.arange(0, 2*len(d_20_80), 2), d_20_80, marker='x', label='N = 100')
    plt.plot(np.arange(0, 2 * len(d_32_18), 2), d_32_18, marker='x', label='N = 120')
    plt.plot(np.arange(0, 2 * len(d_40_10), 2), d_40_10, marker='x', label = 'N = 140')
    plt.legend(loc=0)
    plt.xlabel('L1 norm')
    plt.ylabel('Success rate')

    plt.show()