import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':


    d_20_80 = np.load('d_20_80.npy') # [20, 0, 0, 80]
    d_32_18 = np.load('dep_30_70.npy') # [30, 0, 0, 70]
    d_40_10 = np.load('dep_40_60.npy') # [40, 0, 0, 60]
    d_50_50 = np.load('d_50_50.npy') # [50, 0, 0, 50]


    plt.plot(np.arange(0, 2*len(d_20_80), 2), d_20_80, marker='x', label='20-80')
    plt.plot(np.arange(0, 2 * len(d_32_18), 2), d_32_18, marker='x', label='30-70')
    plt.plot(np.arange(0, 2 * len(d_40_10), 2), d_40_10, marker='x', label = '40-60')
    plt.plot(np.arange(0, 2 * len(d_50_50), 2), d_50_50, marker='x', label = '50-50')
    plt.legend(loc=0)
    plt.xlabel('L1 norm')
    plt.ylabel('Success rate')

    plt.show()