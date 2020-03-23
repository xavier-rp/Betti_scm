import numpy as np
from itertools import permutations
from itertools import combinations
from copy import deepcopy
import scipy as sp
from scipy.stats import chi2
from loglin_model import mle_2x2_ind, iterative_proportional_fitting_AB_AC_BC_no_zeros
import matplotlib.pyplot as plt
import time

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
        print('HERE')
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
    filename = 'CHANGE'

    table_1 = np.array([[25, 25], [25, 25]])
    table_2 = np.array([[50, 0], [0, 50]])

    l1_list = []
    ratio_list = [1]

    #Iterate over a range of number to partition. The L1 norm is equal to 2*number_to_partition
    for number_to_partition in range(24, 51, 1):
        print(number_to_partition)
        table_dictio_1 = {}
        table_dictio_2 = {}
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
                sampled_table_1 = np.array(perm).reshape((2, 2)) + table_1
                sampled_table_2 = np.array(perm).reshape((2, 2)) + table_2
                if np.any(sampled_table_1 < 0):
                    if np.any(sampled_table_2 < 0):
                        continue
                    else:
                        expected_2 = mle_2x2_ind(sampled_table_2)
                        p = chisq_test(sampled_table_2, expected_2)[1]
                        table_dictio_2[tuple(sampled_table_2.flatten())] = p

                else:
                    if np.any(sampled_table_2 < 0):
                        # If the table is valid, we find its expected table under the hypothesis of independance
                        # and evaluate its pvalue, and we add this table in a dictionary (to avoid repetitions
                        # because there are many permutations that are identical)
                        expected_1 = mle_2x2_ind(sampled_table_1)
                        p = chisq_test(sampled_table_1, expected_1)[1]
                        table_dictio_1[tuple(sampled_table_1.flatten())] = p
                    else:
                        expected_2 = mle_2x2_ind(sampled_table_2)
                        p = chisq_test(sampled_table_2, expected_2)[1]
                        table_dictio_2[tuple(sampled_table_2.flatten())] = p
                        expected_1 = mle_2x2_ind(sampled_table_1)
                        p = chisq_test(sampled_table_1, expected_1)[1]
                        table_dictio_1[tuple(sampled_table_1.flatten())] = p

        for partition in l2:
            totalpart = np.concatenate((np.array([number_to_partition, 0]), partition*-1))
            perm_set = set()
            for perm in permutations(totalpart):

                if perm not in perm_set:
                    perm_set.add(perm)
                    # We add the permutation to the original_table. If the resulting table has negative
                    # values, it doesn't represent a contingency table, so we discard it
                    sampled_table_1 = np.array(perm).reshape((2, 2)) + table_1
                    sampled_table_2 = np.array(perm).reshape((2, 2)) + table_2
                    if np.any(sampled_table_1 < 0):
                        if np.any(sampled_table_2 < 0):
                            continue
                        else:
                            expected_2 = mle_2x2_ind(sampled_table_2)
                            p = chisq_test(sampled_table_2, expected_2)[1]
                            table_dictio_2[tuple(sampled_table_2.flatten())] = p

                    else:
                        if np.any(sampled_table_2 < 0):
                            # If the table is valid, we find its expected table under the hypothesis of independance
                            # and evaluate its pvalue, and we add this table in a dictionary (to avoid repetitions
                            # because there are many permutations that are identical)
                            expected_1 = mle_2x2_ind(sampled_table_1)
                            p = chisq_test(sampled_table_1, expected_1)[1]
                            table_dictio_1[tuple(sampled_table_1.flatten())] = p
                        else:
                            expected_2 = mle_2x2_ind(sampled_table_2)
                            p = chisq_test(sampled_table_2, expected_2)[1]
                            table_dictio_2[tuple(sampled_table_2.flatten())] = p
                            expected_1 = mle_2x2_ind(sampled_table_1)
                            p = chisq_test(sampled_table_1, expected_1)[1]
                            table_dictio_1[tuple(sampled_table_1.flatten())] = p
            totalpart = totalpart * -1
            perm_set = set()
            for perm in permutations(totalpart):

                if perm not in perm_set:
                    perm_set.add(perm)
                    # We add the permutation to the original_table. If the resulting table has negative
                    # values, it doesn't represent a contingency table, so we discard it
                    sampled_table_1 = np.array(perm).reshape((2, 2)) + table_1
                    sampled_table_2 = np.array(perm).reshape((2, 2)) + table_2
                    if np.any(sampled_table_1 < 0):
                        if np.any(sampled_table_2 < 0):
                            continue
                        else:
                            expected_2 = mle_2x2_ind(sampled_table_2)
                            p = chisq_test(sampled_table_2, expected_2)[1]
                            table_dictio_2[tuple(sampled_table_2.flatten())] = p

                    else:
                        if np.any(sampled_table_2 < 0):
                            # If the table is valid, we find its expected table under the hypothesis of independance
                            # and evaluate its pvalue, and we add this table in a dictionary (to avoid repetitions
                            # because there are many permutations that are identical)
                            expected_1 = mle_2x2_ind(sampled_table_1)
                            p = chisq_test(sampled_table_1, expected_1)[1]
                            table_dictio_1[tuple(sampled_table_1.flatten())] = p
                        else:
                            expected_2 = mle_2x2_ind(sampled_table_2)
                            p = chisq_test(sampled_table_2, expected_2)[1]
                            table_dictio_2[tuple(sampled_table_2.flatten())] = p
                            expected_1 = mle_2x2_ind(sampled_table_1)
                            p = chisq_test(sampled_table_1, expected_1)[1]
                            table_dictio_1[tuple(sampled_table_1.flatten())] = p

        for partition in l3:
            totalpart = np.concatenate((np.array([number_to_partition]), partition * -1))
            perm_set = set()
            for perm in permutations(totalpart):

                if perm not in perm_set:
                    perm_set.add(perm)
                    # We add the permutation to the original_table. If the resulting table has negative
                    # values, it doesn't represent a contingency table, so we discard it
                    sampled_table_1 = np.array(perm).reshape((2, 2)) + table_1
                    sampled_table_2 = np.array(perm).reshape((2, 2)) + table_2
                    if np.any(sampled_table_1 < 0):
                        if np.any(sampled_table_2 < 0):
                            continue
                        else:
                            expected_2 = mle_2x2_ind(sampled_table_2)
                            p = chisq_test(sampled_table_2, expected_2)[1]
                            table_dictio_2[tuple(sampled_table_2.flatten())] = p

                    else:
                        if np.any(sampled_table_2 < 0):
                            # If the table is valid, we find its expected table under the hypothesis of independance
                            # and evaluate its pvalue, and we add this table in a dictionary (to avoid repetitions
                            # because there are many permutations that are identical)
                            expected_1 = mle_2x2_ind(sampled_table_1)
                            p = chisq_test(sampled_table_1, expected_1)[1]
                            table_dictio_1[tuple(sampled_table_1.flatten())] = p
                        else:
                            expected_2 = mle_2x2_ind(sampled_table_2)
                            p = chisq_test(sampled_table_2, expected_2)[1]
                            table_dictio_2[tuple(sampled_table_2.flatten())] = p
                            expected_1 = mle_2x2_ind(sampled_table_1)
                            p = chisq_test(sampled_table_1, expected_1)[1]
                            table_dictio_1[tuple(sampled_table_1.flatten())] = p
            totalpart = totalpart * -1
            perm_set = set()
            for perm in permutations(totalpart):

                if perm not in perm_set:
                    perm_set.add(perm)
                    # We add the permutation to the original_table. If the resulting table has negative
                    # values, it doesn't represent a contingency table, so we discard it
                    sampled_table_1 = np.array(perm).reshape((2, 2)) + table_1
                    sampled_table_2 = np.array(perm).reshape((2, 2)) + table_2
                    if np.any(sampled_table_1 < 0):
                        if np.any(sampled_table_2 < 0):
                            continue
                        else:
                            expected_2 = mle_2x2_ind(sampled_table_2)
                            p = chisq_test(sampled_table_2, expected_2)[1]
                            table_dictio_2[tuple(sampled_table_2.flatten())] = p

                    else:
                        if np.any(sampled_table_2 < 0):
                            # If the table is valid, we find its expected table under the hypothesis of independance
                            # and evaluate its pvalue, and we add this table in a dictionary (to avoid repetitions
                            # because there are many permutations that are identical)
                            expected_1 = mle_2x2_ind(sampled_table_1)
                            p = chisq_test(sampled_table_1, expected_1)[1]
                            table_dictio_1[tuple(sampled_table_1.flatten())] = p
                        else:
                            expected_2 = mle_2x2_ind(sampled_table_2)
                            p = chisq_test(sampled_table_2, expected_2)[1]
                            table_dictio_2[tuple(sampled_table_2.flatten())] = p
                            expected_1 = mle_2x2_ind(sampled_table_1)
                            p = chisq_test(sampled_table_1, expected_1)[1]
                            table_dictio_1[tuple(sampled_table_1.flatten())] = p

        for partition in l2:
            for partidos in l2:
                totalpart = np.concatenate((partition, partidos * -1))
                perm_set = set()
                for perm in permutations(totalpart):

                    if perm not in perm_set:
                        perm_set.add(perm)
                        # We add the permutation to the original_table. If the resulting table has negative
                        # values, it doesn't represent a contingency table, so we discard it
                        sampled_table_1 = np.array(perm).reshape((2, 2)) + table_1
                        sampled_table_2 = np.array(perm).reshape((2, 2)) + table_2
                        if np.any(sampled_table_1 < 0):
                            if np.any(sampled_table_2 < 0):
                                continue
                            else:
                                expected_2 = mle_2x2_ind(sampled_table_2)
                                p = chisq_test(sampled_table_2, expected_2)[1]
                                table_dictio_2[tuple(sampled_table_2.flatten())] = p

                        else:
                            if np.any(sampled_table_2 < 0):
                                # If the table is valid, we find its expected table under the hypothesis of independance
                                # and evaluate its pvalue, and we add this table in a dictionary (to avoid repetitions
                                # because there are many permutations that are identical)
                                expected_1 = mle_2x2_ind(sampled_table_1)
                                p = chisq_test(sampled_table_1, expected_1)[1]
                                table_dictio_1[tuple(sampled_table_1.flatten())] = p
                            else:
                                expected_2 = mle_2x2_ind(sampled_table_2)
                                p = chisq_test(sampled_table_2, expected_2)[1]
                                table_dictio_2[tuple(sampled_table_2.flatten())] = p
                                expected_1 = mle_2x2_ind(sampled_table_1)
                                p = chisq_test(sampled_table_1, expected_1)[1]
                                table_dictio_1[tuple(sampled_table_1.flatten())] = p

        keylist_1 = list(table_dictio_1.keys())
        keylist_2 = list(table_dictio_2.keys())

        for key in keylist_1:
            try:
                print(table_dictio_2[key], table_dictio_1[key], key)
                print(number_to_partition*2)
            except:
                continue




