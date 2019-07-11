import numpy as np
from scipy.stats import chi2
import scipy as sp
from loglin_model import *
import time
import itertools
import json
import os
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm


######### tests préliminaires

def mle_multinomial_from_table(cont_table):
    n = np.sum(cont_table)
    p_list = []
    for element in cont_table.flatten():
        p_list.append(element/n)

    return p_list

def multinomial_cont_table(nb_trials, nb_categories):
    probabilities = [1 / float(nb_categories)] * nb_categories
    return np.random.multinomial(nb_trials, probabilities, 1).reshape(2, 2)

def multinomial_problist_cont_table(nb_trials, prob_list):
    return np.random.multinomial(nb_trials, prob_list, 1).reshape(2, 2)

def multinomial_problist_cont_cube(nb_trials, prob_list):
    return np.random.multinomial(nb_trials, prob_list, 1).reshape(2, 2, 2)

def multinomial_cont_cube(nb_trials, nb_categories):
    probabilities = [1 / float(nb_categories)] * nb_categories
    return np.random.multinomial(nb_trials, probabilities, 1).reshape(2, 2, 2)


def sample_mult_cont_table(nb_samples, nb_trials, nb_categories):
    samples = []
    for i in range(nb_samples):
        samples.append(multinomial_cont_table(nb_trials, nb_categories))
    return samples

def sample_mult_cont_table_prob(nb_samples, nb_trials, problist):
    samples = []
    for i in range(nb_samples):
        samples.append(multinomial_problist_cont_table(nb_trials, problist))
    return samples

def sample_mult_cont_cube(nb_samples, nb_trials, nb_categories):
    samples = []
    for i in range(nb_samples):
        samples.append(multinomial_cont_cube(nb_trials, nb_categories))
    return samples


def chisq_stats(sample_list, observed):
    chisqlist = []
    for sample in sample_list:
        chisqlist.append(chisq_formula(observed, sample))
    return chisqlist

def chisq_formula(cont_tab, expected):
    #Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
    #via MLE or iterative proportional fitting.
    if float(0) in expected:
        test_stat = 0
    else:
        test_stat = np.sum((cont_tab - expected) ** 2 / expected)

    return test_stat

def sampled_chisq_test(cont_table, expected_table, sampled_array):
    if float(0) in expected_table:
        test_stat = 0
        pval = 1
    else:
        test_stat = np.sum((cont_table - expected_table) ** 2 / expected_table)
        cdf = np.sum((sampled_array < test_stat) * 1) / len(sampled_array)
        pval = 1 - cdf
    return test_stat, pval

def chisq_test(cont_tab, expected):
    #Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
    #via MLE or iterative proportional fitting.

    df = 3
    test_stat = np.sum((cont_tab-expected)**2/expected)
    p_val = chi2.sf(test_stat, df)

    return test_stat, p_val

if __name__ == '__main__':
    #print(get_cont_table(1, 2419, matrix1))   [[4  0] [3 31]] p = 0.00015799999999999148

    #print(get_cont_table(2, 19, matrix1))     [[3  1] [0 34]] p = 0.00033199999999999896

    #print(get_cont_table(2, 129, matrix1))    [[3  1] [1 33]] p = 0.0005659999999999554

    #print(get_cont_table(3, 2059, matrix1))   [[5  6] [0 27]] p = 0.0002969999999999917  WITH 4 * 10 000 000 : 0.000268000000000046



    #table = np.array([[5,  6], [0, 27]])
    #expected_original = mle_2x2_ind(table)
    #problist = mle_multinomial_from_table(expected_original)
    #chisqlist_prob = []
    #start = time.clock()
    #print(problist)
    #with open('convergencepvalue_5_6_0_27_10.csv', 'w', newline='') as csvfile:
    #    writer = csv.writer(csvfile)

    #    for it in tqdm(range(10000000)):
    #        sample = multinomial_problist_cont_table(38, problist)
    #        expected = mle_2x2_ind(sample)
    #        writer.writerow([chisq_formula(sample, expected)])

    #exit()

    import os

    directory = 'Convergencepvalue'
    chisqlist = []
    table = np.array([[5, 6], [0, 27]])
    expected_original = mle_2x2_ind(table)
    start = time.clock()
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            print(filename)
            with open(os.path.join(directory, filename)) as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    chisqlist.append(float(row[0]))
            print('Time for one file : ', time.clock() - start)

    tranche = 100000000
    pvalist = []
    previous_idx = 0
    for idx in np.arange(tranche, 100000000+tranche, tranche):
        pvalist.append(sampled_chisq_test(table, expected_original, chisqlist[previous_idx:idx])[1])
        previous_idx = idx
        #np.save('exact_chi1_scaled_1000', chisqlist_prob)
    print(pvalist)
    print('MEAN : ', np.mean(pvalist))
    print('STD : ', np.std(pvalist))
    print('min : ', np.min(pvalist))
    print('Max : ', np.max(pvalist))

    exit()

    ######## Différence Chi2 réelle, Chi2 multinomial 1/4, Chi2 multinomiale MLE
    tranche = 1000000
    samples = sample_mult_cont_table(1000000, 38, 4)
    pvalist = []
    for i in range(10):
        sam = np.array([[2,  0], [ 0, 36]])*1
        expected1 = mle_2x2_ind(sam)
        problist = mle_multinomial_from_table(expected1)
        chisqlist_prob = []
        start = time.clock()
        for it in range(1000000):
            sample = multinomial_problist_cont_table(38*1, problist)
            expected = mle_2x2_ind(sample)
            chisqlist_prob.append(chisq_formula(sample, expected))
        print(time.clock() - start)
        pvalist.append(sampled_chisq_test(sam, expected1, chisqlist_prob))
        #np.save('exact_chi1_scaled_1000', chisqlist_prob)

    print(pvalist)

    #TODO : df = 3 semble être plus pr de la distribution asymptotique

    ##### Différence entre les pvalues
    df = 1
    #chisqlist_chi1 = np.load('exact_chi1_scaled_1000.npy')
    chisqlist_chi1 = np.load('exact_chi1_table_1_1_6_30.npy')


    ##### différences dans l'allure des courbes
    fig, ax = plt.subplots(1, 1)
    bins2 = np.arange(0, max(chisqlist_chi1) + 1, 0.1)


    #plt.hist(chisqlist_chi1, bins=bins2, weights=np.repeat(1/len(chisqlist_chi1), len(chisqlist_chi1)), alpha=0.5, label='EXACT')
    plt.hist(chisqlist_chi1, bins=bins2, density=True, alpha=0.5,)
    x = np.arange(0, 100, 0.01)
    # ax.plot(x, chi2.pdf(x, df), 'r-', label='chi2 pdf')
    ax.plot(x, chi2.pdf(x, df), label='Asympt degree 3')
    #ax.plot(x, chi2.pdf(x, 1), label='degree 1')
    # bins2 = np.arange(0, max(chisqlist) + 5, 1)
    # print(max(chisqlist))
    # plt.hist(chisqlist, bins=bins2, alpha=0.5)
    plt.legend()
    plt.xlim([0, 20])
    plt.ylim([0, 1])
    plt.show()




    exit()