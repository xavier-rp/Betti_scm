import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':


    d_20_80 = np.load('d_20_80.npy') # [20, 0, 0, 80]
    d_32_18 = np.load('d_32_18.npy') # [32, 18, 18, 32]
    d_40_10 = np.load('d_40_10.npy') # [40, 10, 10, 40]
    d_50_50 = np.load('d_50_50.npy') # [50, 0, 0, 50]


    plt.plot(np.arange(0, 2*len(d_20_80), 2), d_20_80, marker='x', label='20-80')
    plt.plot(np.arange(0, 2 * len(d_32_18), 2), d_32_18, marker='x', label='32-18')
    plt.plot(np.arange(0, 2 * len(d_40_10), 2), d_40_10, marker='x', label = '40-10')
    plt.plot(np.arange(0, 2 * len(d_50_50), 2), d_50_50, marker='x', label = '50-50')
    plt.legend(loc=0)
    plt.xlabel('L1 norm')
    plt.ylabel('Success rate')

    plt.show()