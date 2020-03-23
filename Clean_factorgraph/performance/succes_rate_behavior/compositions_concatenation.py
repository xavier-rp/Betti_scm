# encoding=utf8
import numpy as np
from itertools import permutations
from itertools import combinations
from copy import deepcopy
import scipy as sp
from scipy.stats import chi2
from loglin_model import mle_2x2_ind, iterative_proportional_fitting_AB_AC_BC_no_zeros
import matplotlib.pyplot as plt
import time
from matplotlib import rc

def revlex_partitions(n):
    """
    Generate all integer partitions of n
    in reverse lexicographic order
    """
    if n == 0:
        yield []
        return
    for p in revlex_partitions(n - 1):
        if len(p) == 1 or (len(p) > 1 and p[-1] < p[-2]):
            p[-1] += 1
            yield p
            p[-1] -= 1
        p.append(1)
        yield p
        p.pop()

def chisq_test(cont_tab, expected):
    #Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
    #via MLE or iterative proportional fitting.
    if np.any(expected == 0):
        return 0, 1
    df = 1
    test_stat = np.sum((cont_tab-expected)**2/expected)
    p_val = chi2.sf(test_stat, df)

    return test_stat, p_val

def l_one(sampled_table, model_table):

    return np.sum(np.abs(sampled_table - model_table))

def accel_asc(n):
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]

def partitionfunc(n,k,l=1):
    '''n is the integer to partition, k is the length of partitions, l is the min partition element size'''
    if k < 1:
        raise StopIteration
    if k == 1:
        if n >= l:
            yield (n,)
        raise StopIteration
    for i in range(l,n+1):
        for result in partitionfunc(n-i,k-1,i):
            yield (i,)+result

if __name__ == '__main__':
    start = time.clock()
    filename = 'dep_35_35'

    original_table = np.array([[45, 30], [10, 15]])
    original_table = np.array([[295, 317], [268, 502]])
    #original_table = np.array([[30, 20], [20, 30]])
    original_table = np.array([[25, 25], [25, 25]])

    l1_list = []
    ratio_list = [1]
    total_list = []
    succes_list = []

    #Iterate over a range of number to partition. The L1 norm is equal to 2*number_to_partition
    for number_to_partition in range(1, 51, 1):

        table_dictio = {}
        l2 = []
        l3 = []
        for p in partitionfunc(number_to_partition, 2):  # revlex_partitions(number_to_partition):
            l2.append(np.array(deepcopy(p)))

        for p in partitionfunc(number_to_partition, 3):  # revlex_partitions(number_to_partition):
            l3.append(np.array(deepcopy(p)))

        simplest_part = np.concatenate((np.array([number_to_partition]), np.array([number_to_partition,0,0])*-1))
        perm_set = set()
        for perm in permutations(simplest_part):

            if perm not in perm_set:
                perm_set.add(perm)
                # We add the permutation to the original_table. If the resulting table has negative
                # values, it doesn't represent a contingency table, so we discard it
                sampled_table = np.array(perm).reshape((2, 2)) + original_table
                if np.any(sampled_table < 0):
                    continue
                else:
                    # If the table is valid, we find its expected table under the hypothesis of independance
                    # and evaluate its pvalue, and we add this table in a dictionary (to avoid repetitions
                    # because there are many permutations that are identical)
                    expected = mle_2x2_ind(sampled_table)
                    p = chisq_test(sampled_table, expected)[1]
                    table_dictio[perm] = p

        for partition in l2:
            totalpart = np.concatenate((np.array([number_to_partition, 0]), partition*-1))
            perm_set = set()
            for perm in permutations(totalpart):

                if perm not in perm_set:
                    perm_set.add(perm)
                    #We add the permutation to the original_table. If the resulting table has negative
                    #values, it doesn't represent a contingency table, so we discard it
                    sampled_table = np.array(perm).reshape((2,2)) + original_table
                    if np.any(sampled_table < 0):
                        continue
                    else:
                    #If the table is valid, we find its expected table under the hypothesis of independance
                    #and evaluate its pvalue, and we add this table in a dictionary (to avoid repetitions
                    # because there are many permutations that are identical)
                        expected = mle_2x2_ind(sampled_table)
                        p = chisq_test(sampled_table, expected)[1]
                        table_dictio[perm] = p
            totalpart = totalpart * -1
            perm_set = set()
            for perm in permutations(totalpart):

                if perm not in perm_set:
                    perm_set.add(perm)
                    # We add the permutation to the original_table. If the resulting table has negative
                    # values, it doesn't represent a contingency table, so we discard it
                    sampled_table = np.array(perm).reshape((2, 2)) + original_table
                    if np.any(sampled_table < 0):
                        continue
                    else:
                        # If the table is valid, we find its expected table under the hypothesis of independance
                        # and evaluate its pvalue, and we add this table in a dictionary (to avoid repetitions
                        # because there are many permutations that are identical)
                        expected = mle_2x2_ind(sampled_table)
                        p = chisq_test(sampled_table, expected)[1]
                        table_dictio[perm] = p

        for partition in l3:
            totalpart = np.concatenate((np.array([number_to_partition]), partition * -1))
            perm_set = set()
            for perm in permutations(totalpart):

                if perm not in perm_set:
                    perm_set.add(perm)
                    # We add the permutation to the original_table. If the resulting table has negative
                    # values, it doesn't represent a contingency table, so we discard it
                    sampled_table = np.array(perm).reshape((2, 2)) + original_table
                    if np.any(sampled_table < 0):
                        continue
                    else:
                        # If the table is valid, we find its expected table under the hypothesis of independance
                        # and evaluate its pvalue, and we add this table in a dictionary (to avoid repetitions
                        # because there are many permutations that are identical)
                        expected = mle_2x2_ind(sampled_table)
                        p = chisq_test(sampled_table, expected)[1]
                        table_dictio[perm] = p
            totalpart = totalpart * -1
            perm_set = set()
            for perm in permutations(totalpart):

                if perm not in perm_set:
                    perm_set.add(perm)
                    # We add the permutation to the original_table. If the resulting table has negative
                    # values, it doesn't represent a contingency table, so we discard it
                    sampled_table = np.array(perm).reshape((2, 2)) + original_table
                    if np.any(sampled_table < 0):
                        continue
                    else:
                        # If the table is valid, we find its expected table under the hypothesis of independance
                        # and evaluate its pvalue, and we add this table in a dictionary (to avoid repetitions
                        # because there are many permutations that are identical)
                        expected = mle_2x2_ind(sampled_table)
                        p = chisq_test(sampled_table, expected)[1]
                        table_dictio[perm] = p

        for partition in l2:
            for partidos in l2:
                totalpart = np.concatenate((partition, partidos * -1))
                perm_set = set()
                for perm in permutations(totalpart):

                    if perm not in perm_set:
                        perm_set.add(perm)
                        # We add the permutation to the original_table. If the resulting table has negative
                        # values, it doesn't represent a contingency table, so we discard it
                        sampled_table = np.array(perm).reshape((2, 2)) + original_table
                        if np.any(sampled_table < 0):
                            continue
                        else:
                            # If the table is valid, we find its expected table under the hypothesis of independance
                            # and evaluate its pvalue, and we add this table in a dictionary (to avoid repetitions
                            # because there are many permutations that are identical)
                            expected = mle_2x2_ind(sampled_table)
                            p = chisq_test(sampled_table, expected)[1]
                            table_dictio[perm] = p

        keylist = list(table_dictio.keys())
        total = len(keylist)
        count = 0

        for key in keylist:
            if table_dictio[key] > 0.01:
                count += 1
            #print(key, table_dictio[key])
        print(number_to_partition, count/total)
        print('TOTAL : ', total, ' COUNT : ', count)
        total_list.append(total)
        succes_list.append(count)
        l1_list.append(number_to_partition*2)
        ratio_list.append(count/total)

    np.save(filename, ratio_list)
    ratio_list = np.load(filename + '.npy')

    l1_list = np.arange(0, 2*len(ratio_list), 2)
    print(l1_list)
    print('TIME : ', start - time.clock())
    rc('text', usetex=True)
    rc('font', size=16)
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Distance $L_1$')
    ax1.set_ylabel('Nombre de tables')
    #ax1.plot(t, data1, color=color)
    #ax1.tick_params(axis='y', labelcolor=color)

    #plt.plot(l1_list, ratio_list, marker='x')
    ax1.plot([50, 50], [0, np.max(total_list)], '--k')

    ax1.plot(l1_list[1:], total_list, marker='1', markeredgecolor='black', markerfacecolor='black', markersize='12', color='#00a1ffff', linewidth='3', label ='Total')
    ax1.plot(l1_list[1:], succes_list,  marker='2', markeredgecolor='black', markerfacecolor='black', markersize='12', color='#ff7f00ff', linewidth='3', label = 'Succ\`es')
    ax1.plot(l1_list, ratio_list, marker='3', markeredgecolor='black', markerfacecolor='black', markersize='12',
             color='#41d936ff', linewidth='3', label='Taux de succ\`es')
    #plt.legend(loc=0)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Taux de succ\`es')  # we already handled the x-label with ax1
    ax2.plot(l1_list, ratio_list, marker='3', markeredgecolor='black', markerfacecolor='black', markersize='12', color='#41d936ff', linewidth='3', label='Taux de succ\`es')

    #plt.legend(loc=0)
    plt.show()



