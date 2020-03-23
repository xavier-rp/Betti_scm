import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':


    ind_25 = np.load('dep_37_53_36_74.npy')

    #print(np.mean(ind_25, axis=0))
    #print(np.std(ind_25, axis=0))
    #plt.fill_between(np.arange(0, 42, 2), np.mean(ind_25, axis=0)+ np.std(ind_25, axis=0), np.mean(ind_25, axis=0) - np.std(ind_25, axis=0), alpha=0.2)

    #plt.show()

    ind_50 = np.load('dep_50.npy')
    #ind_75 = np.load('dep_75.npy')
    #ind_100 = np.load('ind_100.npy')

    plt.plot(np.arange(0, 2*len(np.mean(ind_25, axis=0)), 2), np.mean(ind_25, axis=0), marker='x', label='N = 100')
    plt.fill_between(np.arange(0, 2*len(np.mean(ind_25, axis=0)), 2), np.mean(ind_25, axis=0) + np.std(ind_25, axis=0),
                     np.mean(ind_25, axis=0) - np.std(ind_25, axis=0), alpha=0.2)


    plt.plot(np.arange(0, 2 * len(np.mean(ind_25, axis=0)), 2), np.mean(ind_50, axis=0), marker='x', label='N = 200')

    plt.fill_between(np.arange(0, 2 * len(np.mean(ind_25, axis=0)), 2), np.mean(ind_50, axis=0) + np.std(ind_50, axis=0),
                     np.mean(ind_50, axis=0) - np.std(ind_50, axis=0), alpha=0.2)

    #plt.plot(np.arange(0, 2 * len(np.mean(ind_25, axis=0)), 2), np.mean(ind_75, axis=0), marker='x', label = 'N = 300')

    #plt.fill_between(np.arange(0, 2 * len(np.mean(ind_25, axis=0)), 2), np.mean(ind_75, axis=0) + np.std(ind_75, axis=0),
     #                np.mean(ind_75, axis=0) - np.std(ind_75, axis=0), alpha=0.2)


    #plt.plot(np.arange(0, 2 * len(ind_100), 2), ind_100, marker='x', label = 'N = 400')

    plt.legend(loc=0)
    plt.xlabel('L1 norm')
    plt.ylabel('Success rate')

    plt.show()