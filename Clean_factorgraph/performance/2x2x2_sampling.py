import numpy as np
from itertools import permutations
from itertools import combinations
from copy import deepcopy
import scipy as sp
from scipy.stats import chi2
from loglin_model import *
import matplotlib.pyplot as plt
import time
import random

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


def test_models(cont_cube, alpha):
    models = ['ind', 'AB_C', 'AC_B', 'BC_A', 'AB_AC', 'AB_BC', 'AC_BC', 'AB_AC_BC', 'ABC']
    p_value_list = []

    expected_ind = mle_2x2x2_ind(cont_cube)

    expected_AB_C = mle_2x2x2_AB_C(cont_cube)

    expected_AC_B = mle_2x2x2_AC_B(cont_cube)

    expected_BC_A = mle_2x2x2_BC_A(cont_cube)

    expected_AB_AC = mle_2x2x2_AB_AC(cont_cube)

    expected_AB_BC = mle_2x2x2_AB_BC(cont_cube)

    expected_AC_BC = mle_2x2x2_AC_BC(cont_cube)

    expected_AB_AC_BC = iterative_proportional_fitting_AB_AC_BC_no_zeros(cont_cube)


    p_value_list.append(chisq_test(cont_cube, expected_ind)[-1])
    p_value_list.append(chisq_test(cont_cube, expected_AB_C)[-1])
    p_value_list.append(chisq_test(cont_cube, expected_AC_B)[-1])
    p_value_list.append(chisq_test(cont_cube, expected_BC_A)[-1])
    p_value_list.append(chisq_test(cont_cube, expected_AB_AC)[-1])
    p_value_list.append(chisq_test(cont_cube, expected_AB_BC)[-1])
    p_value_list.append(chisq_test(cont_cube, expected_AC_BC)[-1])
    p_value_list.append(chisq_test(cont_cube, expected_AB_AC_BC)[-1])

    chisqlist = []

    chisqlist.append(chisq_test(cont_cube, expected_ind)[0])
    chisqlist.append(chisq_test(cont_cube, expected_AB_C)[0])
    chisqlist.append(chisq_test(cont_cube, expected_AC_B)[0])
    chisqlist.append(chisq_test(cont_cube, expected_BC_A)[0])
    chisqlist.append(chisq_test(cont_cube, expected_AB_AC)[0])
    chisqlist.append(chisq_test(cont_cube, expected_AB_BC)[0])
    chisqlist.append(chisq_test(cont_cube, expected_AC_BC)[0])
    chisqlist.append(chisq_test(cont_cube, expected_AB_AC_BC)[0])

    return p_value_list, chisqlist, models

if __name__ == '__main__':
    """
        original_table = np.array([[[25, 25], [25, 25]], [[25, 25], [25, 25]]])/2
    1 1.0
TOTAL :  56  COUNT :  56
Time take =  0.058504
2 1.0
TOTAL :  812  COUNT :  812
Time take =  0.863019
3 1.0
TOTAL :  5768  COUNT :  5768
Time take =  6.155633
4 1.0
TOTAL :  26474  COUNT :  26474
Time take =  29.466783999999997
5 1.0
TOTAL :  91112  COUNT :  91112
Time take =  109.653911
6 1.0
TOTAL :  256508  COUNT :  256508
Time take =  335.61342900000005
7 1.0
TOTAL :  623576  COUNT :  623576
Time take =  895.535807
8 1.0
TOTAL :  1356194  COUNT :  1356194
Time take =  2068.598578
9 1.0
TOTAL :  2703512  COUNT :  2703512
Time take =  4305.171832
10 1.0
TOTAL :  5025692  COUNT :  5025692
Time take =  8439.600536999998
    """

    #for it in partitionfunc(20, 3):
    #    print(it)

    #exit()
    start = time.clock()
    filename = 'CHANGE'

    original_table = np.array([[[25, 25], [25, 25]], [[25, 25], [25, 25]]])*2

    l1_list = []
    ratio_list = [1]

    #Iterate over a range of number to partition. The L1 norm is equal to 2*number_to_partition
    for number_to_partition in range(40 , 20, -1):

        table_dictio = {}
        l1 = [np.array([number_to_partition])]
        l2 = []
        l3 = []
        l4 = []
        l5 = []
        l6 = []
        l7 = []

        for p in partitionfunc(number_to_partition, 2):  # revlex_partitions(number_to_partition):
            l2.append(np.array(deepcopy(p)))

        for p in partitionfunc(number_to_partition, 3):  # revlex_partitions(number_to_partition):
            l3.append(np.array(deepcopy(p)))

        for p in partitionfunc(number_to_partition, 4):  # revlex_partitions(number_to_partition):
            l4.append(np.array(deepcopy(p)))

        for p in partitionfunc(number_to_partition, 5):  # revlex_partitions(number_to_partition):
            l5.append(np.array(deepcopy(p)))

        for p in partitionfunc(number_to_partition, 6):  # revlex_partitions(number_to_partition):
            l6.append(np.array(deepcopy(p)))

        for p in partitionfunc(number_to_partition, 7):  # revlex_partitions(number_to_partition):
            l7.append(np.array(deepcopy(p)))

        number_of_samples = 1000
        list_of_lists = []
        pre_list_of_lists = [l1,l2,l3,l4,l5,l6,l7]
        for liste in pre_list_of_lists:
            if len(liste) > 0 :
                list_of_lists.append(liste)

        for i in range(0, number_of_samples):

            random_number_1 = np.random.randint(0, len(list_of_lists))
            random_number_2 = np.random.randint(0, len(list_of_lists) - random_number_1)

            first_list = list_of_lists[random_number_1]
            second_list = list_of_lists[random_number_2]

            first_partition = random.choice(first_list)
            second_partition = random.choice(second_list)

            which_negative = np.random.randint(2)

            if which_negative:
                total_partition = np.concatenate((-1*first_partition, second_partition))
            else:
                total_partition = np.concatenate((first_partition, -1*second_partition))

            while len(total_partition) < 8:
                total_partition = np.concatenate((total_partition, np.array([0])))
            np.random.shuffle(total_partition)
            # We add the permutation to the original_table. If the resulting table has negative
            # values, it doesn't represent a contingency table, so we discard it
            sampled_table = np.array(total_partition).reshape((2, 2, 2)) + original_table
            if np.any(sampled_table < 0):
                continue
            else:
                # If the table is valid, we find its expected table under the hypothesis of independance
                # and evaluate its pvalue, and we add this table in a dictionary (to avoid repetitions
                # because there are many permutations that are identical)
                expected = iterative_proportional_fitting_AB_AC_BC_no_zeros(sampled_table)
                if expected is not None:
                    p = chisq_test(sampled_table, expected)[1]
                    #try :
                    #    print(table_dictio[tuple(total_partition)])
                    #except:
                    table_dictio[tuple(total_partition)] = p
                else:
                    print(sampled_table)

        keylist = list(table_dictio.keys())
        total = len(keylist)
        count = 0

        for key in keylist:
            if table_dictio[key] > 0.01:
                count += 1
                # print(key, table_dictio[key])
        print(number_to_partition, count / total)
        print('TOTAL : ', total, ' COUNT : ', count)
        print('Time taken = ', time.clock() - start)

        l1_list.append(number_to_partition * 2)
        ratio_list.append(count / total)


