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
from scipy.stats import chi2

def mle_multinomial_from_table(cont_table):
    n = np.sum(cont_table)
    p_list = []
    for element in cont_table.flatten():
        p_list.append(element/n)

    return p_list

def multinomial_problist_cont_cube(nb_trials, prob_list, s=1):
    return np.random.multinomial(nb_trials, prob_list, s).reshape(s, 2, 2, 2)

def sampled_chisq_test(cont_table, expected_table, sampled_array):
    if float(0) in expected_table:
        test_stat = 0
        pval = 1
    else:
        test_stat = np.sum((cont_table - expected_table) ** 2 / expected_table)
        cdf = np.sum((sampled_array < test_stat) * 1) / len(sampled_array)
        pval = 1 - cdf
    return test_stat, pval

def chisq_formula_vector_for_cubes(cont_tables, expected):
    # Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
    # via MLE or iterative proportional fitting.

    return np.nan_to_num(np.sum(np.sum(np.sum((cont_tables - expected) ** 2 / expected, axis = 1), axis = 1), axis=1))

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

def chisq_test(cont_tab, expected, df=1):
    #Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
    #via MLE or iterative proportional fitting.
    if np.any(expected == 0):
        print('HERE')
        return 0, 1
    #df = 7
    test_stat = np.sum((cont_tab-expected)**2/expected)
    p_val = chi2.sf(test_stat, df)

    return test_stat, p_val

def g_stat_test(cont_tab, expected, df=1):
    #Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
    #via MLE or iterative proportional fitting.
    if np.any(expected == 0):
        print('HERE')
        return 0, 1
    #df = 7
    test_stat = 2*np.sum(cont_tab*np.log(cont_tab/expected))
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

    expected_AC_B = iterative_proportional_fitting_AC_B(cont_cube)

    expected_BC_A = mle_2x2x2_BC_A(cont_cube)

    expected_AB_AC = mle_2x2x2_AB_AC(cont_cube)

    expected_AB_BC = mle_2x2x2_AB_BC(cont_cube)

    expected_AC_BC = mle_2x2x2_AC_BC(cont_cube)

    expected_ind = iterative_proportional_fitting_ind(cont_cube)
    #print(expected_ind)

    expected_AB_C = iterative_proportional_fitting_AB_C(cont_cube)
    #print(expected_AB_C)

    expected_AC_B = iterative_proportional_fitting_AC_B(cont_cube)
    #print(expected_AC_B)
    expected_BC_A = iterative_proportional_fitting_BC_A(cont_cube)
    #print(expected_BC_A)
    expected_AB_AC = iterative_proportional_fitting_AB_AC(cont_cube)

    expected_AB_BC = iterative_proportional_fitting_AB_BC(cont_cube)

    expected_AC_BC = iterative_proportional_fitting_AC_BC(cont_cube)

    expected_AB_AC_BC = iterative_proportional_fitting_AB_AC_BC_no_zeros(cont_cube)


    p_value_list.append(chisq_test(cont_cube, expected_ind, df=4)[-1])
    p_value_list.append(chisq_test(cont_cube, expected_AB_C, df=3)[-1])
    p_value_list.append(chisq_test(cont_cube, expected_AC_B, df=3)[-1])
    p_value_list.append(chisq_test(cont_cube, expected_BC_A, df=3)[-1])
    p_value_list.append(chisq_test(cont_cube, expected_AB_AC, df=2)[-1])
    p_value_list.append(chisq_test(cont_cube, expected_AB_BC, df=2)[-1])
    p_value_list.append(chisq_test(cont_cube, expected_AC_BC, df=2)[-1])
    p_value_list.append(chisq_test(cont_cube, expected_AB_AC_BC, df=1)[-1])

    chisqlist = []

    chisqlist.append(chisq_test(cont_cube, expected_ind, df=4)[0])
    chisqlist.append(chisq_test(cont_cube, expected_AB_C, df=3)[0])
    chisqlist.append(chisq_test(cont_cube, expected_AC_B, df=3)[0])
    chisqlist.append(chisq_test(cont_cube, expected_BC_A, df=3)[0])
    chisqlist.append(chisq_test(cont_cube, expected_AB_AC, df=2)[0])
    chisqlist.append(chisq_test(cont_cube, expected_AB_BC, df=2)[0])
    chisqlist.append(chisq_test(cont_cube, expected_AC_BC, df=2)[0])
    chisqlist.append(chisq_test(cont_cube, expected_AB_AC_BC, df=1)[0])

    gstatlist = []

    gstatlist.append(g_stat_test(cont_cube, expected_ind, df=4)[0])
    gstatlist.append(g_stat_test(cont_cube, expected_AB_C, df=3)[0])
    gstatlist.append(g_stat_test(cont_cube, expected_AC_B, df=3)[0])
    gstatlist.append(g_stat_test(cont_cube, expected_BC_A, df=3)[0])
    gstatlist.append(g_stat_test(cont_cube, expected_AB_AC, df=2)[0])
    gstatlist.append(g_stat_test(cont_cube, expected_AB_BC, df=2)[0])
    gstatlist.append(g_stat_test(cont_cube, expected_AC_BC, df=2)[0])
    gstatlist.append(g_stat_test(cont_cube, expected_AB_AC_BC, df=1)[0])

    return p_value_list, chisqlist, gstatlist, models

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
    nb_samples = 1000000
    filename = '33_22_13_30_24_23_26_29'

    #classroom example :
    #classroom = np.array([[[16, 7], [1, 1]], [[15, 34], [3, 8]], [[5, 3], [1, 3]]])
    #cholesterol = np.array([[[716, 79], [207, 25]], [[819, 67], [186, 22]]])
    #collision = np.array([[[350, 150], [26, 23]], [[60, 112], [19, 80]]])
    #print(test_models(collision, 3))
    #exit()
    #[[[65 42]  [39 60]] [[46 62] [40 46]]]
    original_table = np.array([[30, 20], [20, 30]])*5
    exp = mle_2x2_ind(original_table)
    print(chisq_test(original_table, exp, df=1))
    exit()
    #original_table = np.array([[[33, 22], [13, 30]], [[24, 23], [26, 29]]])
    #original_table = np.array([[[13, 14], [19, 7]], [[7, 6], [12, 26]]])

    #original_table = np.array([[[20, 24], [25, 32]], [[30, 21], [22, 26]]])
    #original_table = np.array([[[36, 13], [24, 27]], [[25, 25], [31, 19]]])
    #original_table = np.array([[[59, 32], [45, 57]], [[54, 64], [49, 40]]])
    #problist = original_table.flatten()/np.sum(original_table)
    #expected_AC_B = mle_2x2x2_AC_B(original_table)
    #exp = iterative_proportional_fitting_AB_AC_BC_no_zeros(original_table)
    #print(chisq_test(exp, expected_AC_B, df=2))
    #original_table = np.array([[[25, 25], [25, 25]], [[21, 25], [25, 29]]])
    #print(test_models(original_table, 9))
    #exit()
    #original_table = np.array([[[5, 0], [0, 5]], [[0, 5], [5, 0]]])/2
    chisqlist = []
    #for i in range(0, 100000):
    #print(i)
    #For rejection of H0 :     [[[49 27]  [26 64]] [[57 74]  [58 45]]]
    original_table = np.array([[[49, 27], [26, 64]], [[57, 74], [58, 45]]]) # For rejection of H0 and the others
    # For rejection of H0 :     [[[51 25] [19 64]][[63 85] [54 39]]]
    original_table = np.array([[[51, 25], [19, 64]], [[63, 85], [54, 39]]])  # For rejection of H0 and the others
    # For rejection of H0 :    [[[62 24]  [20 65]] [[59 93]  [47 30]]] OU [[[ 52  29]  [  8  63]] [[ 50 111]  [ 51  36]]]

    original_table = np.array([[[77, 12], [15, 87]], [[68, 67], [65, 9]]]) #Used for data_100_2 three_dep_triangle_to_simplex
    print(np.sum(original_table,axis=0))
    #exit()
    #original_table = np.array([[[49, 27], [46, 52]], [[64, 74], [45, 43]]])
    #original_table = np.array([[[47, 29], [42, 58]], [[74, 65], [40, 45]]])
    #original_table = np.array([[[155, 15], [15, 15]], [[15, 15], [15, 155]]]) # To accept empty triangles
    #original_table = np.array([[[41, 35], [41, 59]], [[76, 59], [39, 50]]])

    # Empty triangle to 2-simplex


    #original_table = np.array([[[25, 51], [63, 19]], [[85, 64], [39, 54]]])
    #print(test_models(original_table, 9))
    print(original_table/np.sum(original_table), np.sum(original_table))

    #problist = original_table.flatten() / np.sum(original_table)
    #original_table = np.random.multinomial(400, problist).reshape((2,2,2))

    print(original_table)
    print(test_models(original_table, 9))
    exit()
    #print(original_table)
    #print(np.sum(original_table))
    N = np.sum(original_table)
    #exp = iterative_proportional_fitting_AB_AC_BC_no_zeros(original_table)
    exp = iterative_proportional_fitting_AB_C(original_table)
    chisqlist.append(chisq_test(original_table, exp)[0])
    n, b, p = plt.hist(chisqlist, bins=100, density=True)
    plt.xlabel('chi')
    plt.ylabel('count')
    df = 3
    x = np.linspace(chi2.ppf(0.00, df),
                    chi2.ppf(0.999, df), 100)
    plt.plot(x, chi2.pdf(x, df),
             'r-', lw=5, alpha=0.6, label='chi2 pdf')
    plt.show()
    exit()
    expec = np.tile(exp, (nb_samples, 1, 1)).reshape(nb_samples, 2, 2, 2)
    problist = mle_multinomial_from_table(exp)
    sample = multinomial_problist_cont_cube(N, problist, nb_samples)
    chisq = chisq_formula_vector_for_cubes(sample, expec)

    #plt.xlim(0, 1)
    plt.show()
    print(sampled_chisq_test(original_table, exp, chisq))
    #print(exp)
    print(chisq_test(original_table, exp))
    plt.figure(1)
    n, b, p = plt.hist(chisq, bins=20, density=True)
    plt.xlabel('chi')
    plt.ylabel('count')
    df = 7
    x = np.linspace(chi2.ppf(0.00, df),
                    chi2.ppf(0.999, df), 100)
    plt.plot(x, chi2.pdf(x, df),
             'r-', lw=5, alpha=0.6, label='chi2 pdf')
    plt.show()
    exit()
    l1_list = []
    total_ratio_list =  []
    number_of_samples = 1000

    #Iterate over a range of number to partition. The L1 norm is equal to 2*number_to_partition
    for i in range(0, 10):
        if i > 0:
            total_ratio_list.append(ratio_list)

        ratio_list = [1]
        for number_to_partition in range(1, 33, 1):

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

            list_of_lists = []
            pre_list_of_lists = [l1,l2,l3,l4,l5,l6,l7]
            for liste in pre_list_of_lists:
                if len(liste) > 0 :
                    list_of_lists.append(liste)

            for i in range(0, 2*number_of_samples):

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

            ratio_list.append(count / total)

    total_ratio_list.append(ratio_list)
    print(total_ratio_list)

    total_ratio_list = np.array(total_ratio_list)
    print(total_ratio_list)

    print(total_ratio_list.shape)

    np.save(filename, total_ratio_list)



