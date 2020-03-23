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

    #for it in partitionfunc(20, 3):
    #    print(it)

    #exit()

    start = time.clock()
    filename = 'CHANGE'

    original_table = np.array([[[250, 250], [250, 250]], [[250, 250], [250, 250]]])
    original_table = np.array([[45, 30], [10, 15]])
    original_table = np.array([[295, 317], [268, 502]])
    #original_table = np.array([[30, 20], [20, 30]])
    original_table = np.array([[25, 25], [24, 26]])


    #original_table = np.array([[12, 25], [25, 38]])
    #original_table = np.array([[20, 0], [0, 80]])
    #original_table = np.array([[25, 25], [25, 25]])
    #original_table = np.array([[75, 75], [75, 75]])

    #Dependance vs N
    #original_table = np.array([[63.69416035, 15], [10, 11.30583965]]) #dep_100
    #original_table = np.array([[135.06281265, 30], [ 20, 14.93718735]]) # dep_200
    #original_table = np.array([[206.49477294, 45], [30, 18.50522706]]) # dep_300
    #original_table = np.array([[278.09251464, 60], [40, 21.90748536]]) # dep_400

    #Independance vs N
    #original_table = np.array([[42.62613217, 15], [25, 17.37386783]]) #'ind_samechi_100'
    #original_table = np.array([[93.4870287, 30], [50, 26.5129713]]) # 'ind_samechi_200'
    #original_table = np.array([[144.48792507, 45], [75, 35.51207493]]) # 'ind_samechi_300'
    #original_table = np.array([[195.71877794,  60], [100, 44.28122206]]) # 'ind_samechi_400'

    #Independance vs N
    #original_table = np.array([[7.51280963, 20], [15, 57.48719037]]) # ind_samechi_05_100
    #original_table = np.array([[ 13.42317634, 40], [30, 116.57682366]]) # ind_samechi_05_200
    #original_table = np.array([[19.12598277, 60], [45, 175.87401723]]) # ind_samechi_05_300
    #original_table = np.array([[24.71979651, 80], [60, 235.28020349]]) # ind_samechi_05_400


    #exp = iterative_proportional_fitting_AB_AC_BC_no_zeros(original_table)
    exp = mle_2x2_ind(original_table)
    print(exp)
    print(chisq_test(original_table, exp))
    print(np.sum(original_table))
    exit()

    l1_list = []
    ratio_list = [1]
    mask1 = np.array([-1, 1, 1, 1])
    mask2 = np.array([-1, -1, 1, 1])
    mask3 = np.array([-1, -1, -1, 1])
    mask_list = [mask1, mask2, mask3]

    #Iterate over a range of number to partition. The L1 norm is equal to 2*number_to_partition
    for number_to_partition in range(2, 102, 2):

        table_dictio = {}
        for length_of_part in [2, 3, 4]:
            for p in partitionfunc(number_to_partition, length_of_part): #revlex_partitions(number_to_partition):

                l = list(deepcopy(p))

                #When the partition contains less than 4 elements, we need to add -number_to_partition
                # to the partition, because we want the sum of the resulting list to be zero. Indeed
                # if we add some counts at some place in the contingency table, we need to remove the
                # same amount elsewhere in order to conserve the total number of observations in the
                # contingency table
                if len(l) <= 3 and len(l) > 1:

                    #l.append(number_to_partition)

                    #Since in a 2X2 table, we have 4 possible states, we want our list to be 4 elements long, so we add
                    #zeroes if there aren't already 4.
                    while len(l) < 4:
                        l.append(0)

                    # When we have our list of 4 numbers that sums to zero, we check all possible permutations of these
                    # 4 numbers and add them to the original_table
                    # We also have to try every other permutations that sum to zero by changing which numbers are
                    # negative
                    mask_permut_set = set()
                    for mask in mask_list:
                        for mask_permut in permutations(mask):
                            if mask_permut not in mask_permut_set:
                                mask_permut_set.add(mask_permut)
                                if np.sum(np.array(l) * mask_permut) == 0:
                                    perm_set = set()
                                    for perm in permutations(np.array(l) * mask_permut):
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

                #If the length of the partition is equal to 4, we only need to check if, by adding negative signs to some of
                #the numbers, we get a sum of zero.
                elif len(l) == 4:
                    mask_permut_set = set()
                    for mask in mask_list:
                        for mask_permut in permutations(mask):
                            if mask_permut not in mask_permut_set:
                                mask_permut_set.add(mask_permut)
                                if np.sum(np.array(l) * mask_permut) == 0:

                                    perm_set = set()
                                    for perm in permutations(np.array(l) * mask_permut):
                                        if perm not in perm_set:
                                            perm_set.add(perm)

                                        sampled_table = np.array(perm).reshape((2, 2)) + original_table
                                        if np.any(sampled_table < 0):
                                            continue
                                        else:
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

        l1_list.append(number_to_partition)
        ratio_list.append(count/total)

    np.save(filename, ratio_list)
    ratio_list = np.load(filename + '.npy')

    l1_list = np.arange(0, 2*len(ratio_list), 2)
    print(l1_list)
    print('TIME : ', time.clock() - start)
    plt.plot(l1_list, ratio_list, marker='x')
    plt.xlabel('L1 norm')
    plt.ylabel('Success rate')
    plt.show()



