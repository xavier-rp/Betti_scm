import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    ind_25 = np.load('ind_12_25_25_38.npy')


    plt.plot(np.arange(0, 2 * len(ind_25), 2), ind_25, marker='x', label='N = 100')
    plt.plot([24, 24], [0, 1])

    plt.plot([50, 50], [0, 1])

    plt.plot([76, 76], [0, 1])

    plt.legend(loc=0)
    plt.xlabel('L1 norm')
    plt.ylabel('Success rate')

    plt.show()