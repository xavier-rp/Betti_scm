import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    # original_table = np.array([[[20, 24], [25, 32]], [[30, 21], [22, 26]]])
    # chi , p (0.6440169211375366, 0.4222599440018261)
    ind_25 = np.load('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/performance/2x2x2_samechi/ind_20_24_25_32_30_21_22_26.npy')


    # original_table = np.array([[[36, 13], [24, 27]], [[25, 25], [31, 19]]])
    # chi, p (7.7105039060721285, 0.0054900424420377455)
    ind_50 = np.load('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/performance/2x2x2_samechi/ind_36_13_24_27_25_25_31_19.npy')

    # original_table = np.array([[[59, 32], [45, 57]], [[54, 64], [49, 40]]])*2
    # (17.948617417275557, 2.2694921049632322e-05)
    ind_800 = np.load('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/performance/2x2x2_samechi/118_64_90_114_108_128_98_80.npy')


    # original_table = np.array([[[33, 22], [13, 30]], [[24, 23], [26, 29]]])
    # chi, p (3.4781786991131804, 0.06218314417183539)
    ind_75 = np.load('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/performance/2x2x2_samechi/ind_33_22_13_30_24_23_26_29.npy')
    #ind_100 = np.load('ind_100.npy')

    plt.plot(np.arange(0, 2*len(np.mean(ind_25, axis=0)), 2), np.mean(ind_25, axis=0), marker='x', label='20_24_25_32_30_21_22_26')
    plt.fill_between(np.arange(0, 2*len(np.mean(ind_25, axis=0)), 2), np.mean(ind_25, axis=0) + np.std(ind_25, axis=0),
                     np.mean(ind_25, axis=0) - np.std(ind_25, axis=0), alpha=0.2)


    plt.plot(np.arange(0, 2 * len(np.mean(ind_25, axis=0)), 2), np.mean(ind_50, axis=0), marker='x', label='36_13_24_27_25_25_31_19')

    plt.fill_between(np.arange(0, 2 * len(np.mean(ind_25, axis=0)), 2), np.mean(ind_50, axis=0) + np.std(ind_50, axis=0),
                     np.mean(ind_50, axis=0) - np.std(ind_50, axis=0), alpha=0.2)

    plt.plot(np.arange(0, 2 * len(np.mean(ind_25, axis=0)), 2), np.mean(ind_75, axis=0), marker='x', label = '33_22_13_30_24_23_26_29')

    plt.fill_between(np.arange(0, 2 * len(np.mean(ind_25, axis=0)), 2), np.mean(ind_75, axis=0) + np.std(ind_75, axis=0),
                     np.mean(ind_75, axis=0) - np.std(ind_75, axis=0), alpha=0.2)

    #plt.plot(np.arange(0, 2 * len(np.mean(ind_25, axis=0)), 2), np.mean(ind_800, axis=0), marker='x',
    #         label='118_64_90_114_108_128_98_80')

    #plt.fill_between(np.arange(0, 2 * len(np.mean(ind_25, axis=0)), 2),
    #                 np.mean(ind_800, axis=0) + np.std(ind_800, axis=0),
    #                 np.mean(ind_800, axis=0) - np.std(ind_800, axis=0), alpha=0.2)

    #plt.plot(np.arange(0, 2 * len(ind_100), 2), ind_100, marker='x', label = 'N = 400')

    plt.legend(loc=0)
    plt.xlabel('L1 norm')
    plt.ylabel('Success rate')

    plt.show()