import numpy as np
from scipy.stats import chi2
import scipy as sp
from loglin_model import *
import time


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
    import matplotlib.pyplot as plt

    chisqlist = []

    for it in range(1000000):
        print(it)
        sample = multinomial_problist_cont_table(38, [1/4, 1/4, 1/4, 1/4])
        expected = mle_2x2_ind(sample)
        chisqlist.append(chisq_formula(sample, expected))

    np.save('chisqlist_chi1_38', np.array(chisqlist))

    exit()


    chisqlist = np.load('chisqlist_chi1.npy')

    ##### différences dans l'allure des courbes
    fig, ax = plt.subplots(1, 1)
    # bins = np.arange(0, max(chisqlist_unquart) + 5, 0.1)
    ## print(max(chisqlist))
    # plt.hist(chisqlist_unquart, bins=bins, alpha=0.5, density=True, label='un quart')

    bins6 = np.arange(0, max(chisqlist) + 5, 0.01)
    # print(max(chisqlist))
    plt.hist(chisqlist, bins=bins6, alpha=0.5, density=True, label='exp chi1')

    x = np.arange(0, 100, 0.01)

    ax.plot(x, chi2.pdf(x, 1), label='degree 1')
    # bins2 = np.arange(0, max(chisqlist) + 5, 1)
    # print(max(chisqlist))
    # plt.hist(chisqlist, bins=bins2, alpha=0.5)
    plt.legend()
    plt.xlim([0, 20])
    plt.ylim([0, 1])
    plt.show()

    #plist = []
    #for i in range(1000):
    #    contable = multinomial_cont_table(38, 4)
    #    plist.append(mle_multinomial_from_table(contable)[0])
    #print(np.sum(plist)/1000)

    #contable = multinomial_cont_table(38, 4)
    #print(contable)
    #plist = mle_multinomial_from_table(contable)
    #print(plist)
    #exit()

    #fig, ax = plt.subplots(1, 1)
    #x = np.linspace(chi2.ppf(0.01, 1),
    #                chi2.ppf(0.99, 1), 100)
    #ax.plot(x, chi2.pdf(x, 1),
    #        'r-', lw=5, alpha=0.6, label='chi2 pdf')
    #plt.show()

    ######### Temps linéaire :
    #timelist = []
    #itlist = []

    #for it in [1, 10, 100, 1000, 10000, 100000, 1000000, 5000000]:
    #    print(it)
    #    start = time.time()
    #    sample_mult_cont_table(int(it), 38, 4)
    #    timelist.append(time.time()-start)
    #    itlist.append(int(it))
    #plt.plot(itlist, timelist)
    #plt.show()
    #exit()


    ######## Différence Chi2 réelle, Chi2 multinomial 1/4, Chi2 multinomiale MLE
    #samples = sample_mult_cont_table(1000000, 38, 4)

    #print(samples[0])
    #sam = np.array([[20000,  0], [ 0, 360000]])
    #expected = mle_2x2_ind(sam)
    #print(expected)
    #problist = mle_multinomial_from_table(expected)
    #print(problist)
    #print(chisq_formula(sam, expected))
    #for i in range(100):
    #    sample = multinomial_problist_cont_table(3800, problist)
        #if chisq_formula(sample, expected) < 1:
        #    print(sample, expected)
    #    print(chisq_formula(sample, expected))

    #exit()
    #samplesprob = sample_mult_cont_table_prob(1000000, 38000000, problist)

    #chisqlist = []
    #chisqlist_prob = []

    #for it in range(10000000):
    #    print(it)
    #    sample = multinomial_problist_cont_table(380000, problist)
    #    chisqlist_prob.append(chisq_formula(sample, expected))
    #for sample in samples:
    #    expected = mle_2x2_ind(sample)
    #    if expected is not None:
    #        chisqlist.append(chisq_formula(sample, expected))

    #for sample in samplesprob:
    #        chisqlist_prob.append(chisq_formula(sample, expected))

    #np.save('chisqlist_unquart', chisqlist)
    #np.save('exact_chisqlist380000_10millions', chisqlist_prob)
    #exit()

    df = 3
    #TODO : df = 3 semble être plus pr de la distribution asymptotique

    #fig, ax = plt.subplots(1, 1)
    #chisqlist_1000 = np.load('chisqlist_prob1000.npy')
    #bins = np.arange(0, max(chisqlist_1000) + 5, 0.1)
    ## print(max(chisqlist))
    #plt.hist(chisqlist_1000, bins=bins, alpha=0.5, density=True, label='un quart')
    #plt.show()


    ##### Différence entre les pvalues
    # np.array([[21, 9], [3, 5]])  # prob
    # np.array([[512, 234], [103, 151]]) #lorsque 1000
    # np.array([[5120, 2340], [1030, 1510]])  # lorsque 10000
    # np.array([[1, 5], [5, 27]]) # probanother
    # np.array([[ 2,  0], [ 0, 36]]) problowpval avec 0 et 753 pval de e-10, phi = 1
    exact_chisqlist10mil = np.load('exact_chisqlist380000_10millions.npy')
    exact_chisqlist = np.load('exact_chisqlist.npy')
    exact_chisqlist380000 = np.load('exact_chisqlist380000.npy')
    exact_chisqlist38000000 = np.load('exact_chisqlist38000000.npy')
    #chisqlist_unquart = np.load('chisqlist_unquart.npy')
    #chisqlist_prob = np.load('chisqlist_prob.npy')
    #chisqlist_probanother = np.load('chisqlist_probanother.npy')
    chisqlist_problowpval = np.load('chisqlist_problowpval.npy')
    cont = np.array([[ 2,  0], [ 0, 36]]) # problowval
    expec = mle_2x2_ind(cont)
    #print(expec)
    pvalue_exact = sampled_chisq_test(cont, expec, exact_chisqlist)
    #pvalunquart = sampled_chisq_test(cont, expec, chisqlist_unquart)
    #pvalueprob = sampled_chisq_test(cont, expec, chisqlist_prob)
    #pval = chisq_test(cont, expec)
    #print(cont, expec, chisq_test(cont, expec))
    #print(pval, pvalunquart, pvalueprob)
    print('EXACT : ', pvalue_exact)
    print('WRONG DIST : ', sampled_chisq_test(cont, expec, chisqlist_problowpval))
    print('ASYMPT : ', chisq_test(cont, expec))
    #exit()

    ##### différences dans l'allure des courbes
    fig, ax = plt.subplots(1, 1)
    #bins = np.arange(0, max(chisqlist_unquart) + 5, 0.1)
    ## print(max(chisqlist))
    #plt.hist(chisqlist_unquart, bins=bins, alpha=0.5, density=True, label='un quart')

    bins2 = np.arange(0, max(exact_chisqlist) + 5, 0.1)
    # print(max(chisqlist))
    plt.hist(exact_chisqlist, bins=bins2, alpha=0.5, density=True, label='EXACT')

    #bins3 = np.arange(0, max(chisqlist_prob) + 5, 0.1)
    ## print(max(chisqlist))
    #plt.hist(chisqlist_probanother, bins=bins3, alpha=0.5, density=True, label='probanother')

    #bins4 = np.arange(0, max(chisqlist_problowpval) + 5, 0.5)
    # print(max(chisqlist))
    #plt.hist(chisqlist_problowpval, bins=bins4, alpha=0.5, density=True, label='WRONG')

    #bins4 = np.arange(0, max(exact_chisqlist380000) + 5, 0.1)
    # print(max(chisqlist))
    #plt.hist(exact_chisqlist380000, bins=bins4, alpha=0.5, density=True, label='380000')

    #bins5 = np.arange(0, max(exact_chisqlist38000000) + 5, 0.01)
    # print(max(chisqlist))
    #plt.hist(exact_chisqlist38000000, bins=bins5, alpha=0.5, density=True, label='380000')

    bins6 = np.arange(0, max(exact_chisqlist10mil) + 5, 0.01)
    # print(max(chisqlist))
    plt.hist(exact_chisqlist10mil, bins=bins6, alpha=0.5, density=True, label='10mil')

    x = np.arange(0, 100, 0.01)
    # ax.plot(x, chi2.pdf(x, df), 'r-', label='chi2 pdf')
    ax.plot(x, chi2.pdf(x, df), label='Asympt degree 3')
    ax.plot(x, chi2.pdf(x, 1), label='degree 1')
    # bins2 = np.arange(0, max(chisqlist) + 5, 1)
    # print(max(chisqlist))
    # plt.hist(chisqlist, bins=bins2, alpha=0.5)
    plt.legend()
    plt.xlim([0, 20])
    plt.ylim([0, 1])
    plt.show()




    exit()


    #chisqlist = []

    #for sample in samples:
    #    expected = iterative_proportional_fitting_AB_AC_BC_no_zeros(sample, delta=0.000001)
    #    if expected is not None :
    #        chisqlist.append(chisq_formula(sample, expected))

    #print('Sampling time : ', time.time() - start)
    #start = time.time()
    #chisqlist = np.array(chisqlist)
    #np.save('chisqlist2x2x2100000', chisqlist)
    #print('To array time : ', time.time() - start)
    #print(max(chisqlist))

    #exit()
    chisqlist = np.load('chisqlist.npy')
    #stat = 1
    #start = time.time()
    #difflist = []
    #cdflist = []
    #print(len(chisqlist))
    #for stat in np.arange(0.1, 100, 0.1):
    #    cdflist.append(np.sum((chisqlist < stat)*1)/len(chisqlist))
    #    #difflist.append(np.abs(chi2.cdf(stat, 1) - np.sum((chisqlist < stat)*1)/1000000))
    #    #difflist.append(chi2.cdf(stat, 1) - np.sum((chisqlist < stat) * 1) / 1000000)
    #    #print('Sampled cumulative prob for chi = ' + str(stat) + 'is :', np.sum((chisqlist < stat)*1)/1000000, 'CHI2 cumulative prob : ', chi2.cdf(stat, 1))
    #print('Probability time : ', time.time() - start)
    ##plt.plot(np.arange(0.1, 100, 0.1), difflist)
    #plt.plot(np.arange(0.1, 100, 0.1), cdflist)
    #plt.plot(np.arange(0.1, 100, 0.1), chi2.cdf(np.arange(0.1, 100, 0.1), 1))
    #print(cdflist[-1])
    #plt.show()

    print(chisq_formula(np.array([[38,0], [0,0]]),np.array([[38,0], [0,0]])))
    #chisqlist = chisq_stats(samples, np.array([[38,0], [0,0]]))
    #for chi in chisqlist:
    #    print(chi)
    exit()
    #print('HERE')
    #for chi in chisqlist:
    #    if type(chi) != np.float64:
    #        print(chi)

    df = 1
    fig, ax = plt.subplots(1, 1)


    bins = np.arange(0, max(chisqlist)+5, 0.1)
    #print(max(chisqlist))
    plt.hist(chisqlist, bins=bins, alpha=0.5, density=True)

    x = np.arange(min(chisqlist), max(chisqlist), 0.01)
    #ax.plot(x, chi2.pdf(x, df), 'r-', label='chi2 pdf')
    ax.plot(x, chi2.pdf(x, df))
    ax.plot(x, chi2.pdf(x, 1))
    #bins2 = np.arange(0, max(chisqlist) + 5, 1)
    #print(max(chisqlist))
    #plt.hist(chisqlist, bins=bins2, alpha=0.5)
    plt.show()








