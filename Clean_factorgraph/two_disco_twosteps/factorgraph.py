import numpy as np
import scipy as sp
from scipy.stats import chi2
from loglin_model import *
import time

def mle_multinomial_from_table(cont_table):
    n = np.sum(cont_table)
    p_list = []
    for element in cont_table.flatten():
        p_list.append(element/n)

    return p_list

def multinomial_problist_cont_table(nb_trials, prob_list, s=1):
    return np.random.multinomial(nb_trials, prob_list, s).reshape(s, 2, 2)

def chisq_formula_vector(cont_tables, expected):
    # Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
    # via MLE or iterative proportional fitting.

    return np.nan_to_num(np.sum(np.sum((cont_tables - expected) ** 2 / expected, axis = 1), axis = 1))

def sampled_chisq_test(cont_table, expected_table, sampled_array):
    if float(0) in expected_table:
        test_stat = 0
        pval = 1
    else:
        test_stat = np.sum((cont_table - expected_table) ** 2 / expected_table)
        cdf = np.sum((sampled_array < test_stat) * 1) / len(sampled_array)
        pval = 1 - cdf
    return test_stat, pval


def phi_coefficient_table(cont_tab):
   row_sums = np.sum(cont_tab, axis=1)
   col_sums = np.sum(cont_tab, axis=0)

   return (cont_tab[0,0]*cont_tab[1,1] - cont_tab[1,0]*cont_tab[0,1])/np.sqrt(row_sums[0]*row_sums[1]*col_sums[0]*col_sums[1])

def chisq_test(cont_tab, expected):
    #Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
    #via MLE or iterative proportional fitting.

    df = 1
    test_stat = np.sum((cont_tab-expected)**2/expected)
    p_val = chi2.sf(test_stat, df)

    return test_stat, p_val

def pairwise_p_values_phi(bipartite_matrix):

    contingency_table = get_cont_table(0, 1, bipartite_matrix)
    expected_table = mle_2x2_ind(contingency_table)
    phi = phi_coefficient_table(contingency_table)
    chi2, p = chisq_test(contingency_table, expected_table)
    return chi2, p, phi

def get_cont_table(u_idx, v_idx, matrix):
    # Computes the 2X2 contingency table for the occurrence matrix
    row_u_present = matrix[u_idx, :]
    row_v_present = matrix[v_idx, :]

    row_u_not_present = 1 - row_u_present
    row_v_not_present = 1 - row_v_present

    # u present, v present
    table00 = np.dot(row_u_present, row_v_present)

    # u present, v NOT present
    table01 = np.dot(row_u_present, row_v_not_present)

    # u NOT present, v present
    table10 = np.dot(row_u_not_present, row_v_present)

    # u NOT present, v NOT present
    table11 = np.dot(row_u_not_present, row_v_not_present)

    return np.array([[table00, table01], [table10, table11]])



def energy_small_network_dependence():
    state1 = [0,0]
    state2 = [1,0]
    state3 = [0,1]
    state4 = [1,1]

    allstate = [state1, state2, state3, state4]
    energylist = []
    problist = []
    Z = 0
    wij = -1

    for state in allstate:
        x1 = state[0]
        x2 = state[1]
        energy = wij*(x1-x2)**2
        prob = np.exp(-energy)
        Z += prob
        energylist.append(energy)
        problist.append(prob)

    problist = np.array(problist)/Z

    return problist


def energy_small_network_independence():
    state1 = [0,0]
    state2 = [1,0]
    state3 = [0,1]
    state4 = [1,1]

    allstate = [state1, state2, state3, state4]
    energylist = []
    problist = []
    Z = 0
    wij = -0.7

    for state in allstate:
        x1 = state[0]
        x2 = state[1]
        energy = wij*(x1+x2)
        prob = np.exp(-energy)
        Z += prob
        energylist.append(energy)
        problist.append(prob)

    problist = np.array(problist)/Z

    return problist

if __name__ == '__main__':
    state1 = [0, 0]
    state2 = [1, 0]
    state3 = [0, 1]
    state4 = [1, 1]

    allstate = [state1, state2, state3, state4]


    problist = energy_small_network_independence()
    print(problist)

    drawn_states = np.random.choice(4, 1000, p=problist)


    bipartite = np.random.rand(2, len(drawn_states))

    i = 0

    for drawn in drawn_states :
        bipartite[:, i] = allstate[drawn]
        i += 1

    table = get_cont_table(0,1, bipartite)
    print(table)







    N = np.sum(table)
    expected1 = mle_2x2_ind(table)
    print(expected1)
    problist = mle_multinomial_from_table(expected1)
    start = time.clock()
    sample = multinomial_problist_cont_table(N, problist, 1000000)
    expec = mle_2x2_ind_vector(sample, N)
    chisq = np.nan_to_num(chisq_formula_vector(sample, expec))
    print(pairwise_p_values_phi(bipartite))
    print(sampled_chisq_test(table, expected1, chisq))
    print('Time for one table: ', time.clock()-start)

    #    json.dump(pvaldictio, open("pvaldictio.json", 'w'))
