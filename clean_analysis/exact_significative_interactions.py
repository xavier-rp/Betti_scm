import numpy as np
import pandas as pd
import collections
import json
import copy
import scipy as sp
import scipy.misc
import scipy.stats
import itertools
import time
import pickle
import os
from loglin_model import *
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import csv
from scipy.stats import chi2
from Exact_chi_square_1_deg import *


def pvalue_AB_AC_BC(cont_cube):
    expected = iterative_proportional_fitting_AB_AC_BC_no_zeros(cont_cube)
    if expected is not None:
        return chisq_test(cont_cube, expected)[1]
    else:
        return expected

def pvalue_ABC_ABD_ACD_BCD(hyper_cont_cube):
    expected = ipf_ABC_ABD_ACD_BCD_no_zeros(hyper_cont_cube)
    if expected is not None:
        return chisq_test(hyper_cont_cube, expected)[1]
    else:
        return expected

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

    expected_AB_AC_BC = iterative_proportional_fitting_AB_AC_BC(cont_cube)


    p_value_list.append(chisq_test(cont_cube, expected_ind)[0])
    p_value_list.append(chisq_test(cont_cube, expected_AB_C)[0])
    p_value_list.append(chisq_test(cont_cube, expected_AC_B)[0])
    p_value_list.append(chisq_test(cont_cube, expected_BC_A)[0])
    p_value_list.append(chisq_test(cont_cube, expected_AB_AC)[0])
    p_value_list.append(chisq_test(cont_cube, expected_AB_BC)[0])
    p_value_list.append(chisq_test(cont_cube, expected_AC_BC)[0])
    p_value_list.append(chisq_test(cont_cube, expected_AB_AC_BC)[0])

    for i in range(8):
        if p_value_list[i] < alpha:
            return p_value_list[i], models[i]

    return models[-1]

def to_occurrence_matrix(matrix, savepath=None):
    """
    Transform a matrix into a binary matrix where entries are 1 if the original entry was different from 0.
    Parameters
    ----------
    matrix (np.array)
    savepath (string) : path and filename under which to save the file
    Returns
    -------
        The binary matrix or None if a savepath is specified.
    """
    if savepath is None:
        return (matrix > 0)*1
    else:
        np.save(savepath, (matrix>0)*1)

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

def get_cont_cube(u_idx, v_idx, w_idx, matrix):
    # Computes the 2X2X2 contingency table for the occurrence matrix

    row_u_present = matrix[u_idx, :]
    row_v_present = matrix[v_idx, :]
    row_w_present = matrix[w_idx, :]

    row_u_not = 1 - row_u_present
    row_v_not = 1 - row_v_present
    row_w_not = 1 - row_w_present

    #All present :
    table000 =np.sum(row_u_present*row_v_present*row_w_present)

    # v absent
    table010 = np.sum(row_u_present*row_v_not*row_w_present)

    # u absent
    table100 = np.sum(row_u_not*row_v_present*row_w_present)

    # u absent, v absent
    table110 = np.sum(row_u_not*row_v_not*row_w_present)

    # w absent
    table001 = np.sum(row_u_present*row_v_present*row_w_not)

    # v absent, w absent
    table011 = np.sum(row_u_present*row_v_not*row_w_not)

    # u absent, w absent
    table101 = np.sum(row_u_not*row_v_present*row_w_not)

    # all absent
    table111 = np.sum(row_u_not*row_v_not*row_w_not)

    return np.array([[[table000, table010], [table100, table110]], [[table001, table011], [table101, table111]]], dtype=np.float64)


def phi_coefficient_table(cont_tab):
   row_sums = np.sum(cont_tab, axis=1)
   col_sums = np.sum(cont_tab, axis=0)

   return (cont_tab[0,0]*cont_tab[1,1] - cont_tab[1,0]*cont_tab[0,1])/np.sqrt(row_sums[0]*row_sums[1]*col_sums[0]*col_sums[1])

def phi_coefficient_chi(cont_tab, chi):

   n = np.sum(cont_tab)

   return np.sqrt(chi/n)

def chisq_test(cont_tab, expected):
    #Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
    #via MLE or iterative proportional fitting.

    df = 1
    test_stat = np.sum((cont_tab-expected)**2/expected)
    p_val = chi2.sf(test_stat, df)

    return test_stat, p_val

def read_pairwise_p_values(filename, alpha=0.01):

    graph = nx.Graph()

    with open(filename, 'r') as csvfile:

        reader = csv.reader(csvfile)
        next(reader)

        for row in tqdm(reader):

            try:
                p = float(row[-1])
                if p < alpha:
                    # Reject H_0 in which we suppose that u and v are independent
                    # Thus, we accept H_1 and add a link between u and v in the graph to show their dependency
                    graph.add_edge(int(row[0]), int(row[1]), phi=float(row[-2]), p_value=p)
            except:
                pass


    return graph

def save_all_triangles(G, savename, bufferlimit=100000):
    G = copy.deepcopy(G)
    with open(savename + '.csv', 'w',  newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows([['node index 1', 'node index 2', 'node index 3', 'phi 1-2', 'phi 1-3', 'phi 2-3']])
    buffer = []
    # Iterate over all possible triangle relationship combinations
    count = 0
    for node in list(G.nodes):
        if G.degree[node] < 2:
            G.remove_node(node)
        else:
            for n1, n2 in itertools.combinations(G.neighbors(node), 2):

                # Check if n1 and n2 have an edge between them
                if G.has_edge(n1, n2):

                    buffer.append([node, n1, n2, G.get_edge_data(node, n1)['phi'],
                                   G.get_edge_data(node, n2)['phi'], G.get_edge_data(n1, n2)['phi']])
                    count += 1

            G.remove_node(node)

            if count == bufferlimit:
                with open(savename + '.csv', 'a',  newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerows(buffer)
                    count = 0
                    # empty the buffer
                    buffer = []

    with open(savename + '.csv', 'a',  newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(buffer)

def triangles_p_values_AB_AC_BC(csvfile, savename, matrix, bufferlimit=100000):

    buffer = []

    with open(csvfile, 'r') as csvfile, open(savename + '.csv', 'w',  newline='') as fout:
        reader = csv.reader(csvfile)
        writer = csv.writer(fout)
        writer.writerows([['node index 1', 'node index 2', 'node index 3', 'phi 1-2', 'phi 1-3', 'phi 2-3', 'p-value']])
        count = 0
        none_count = 0
        next(reader)
        for row in tqdm(reader):

            cont_cube = get_cont_cube(int(row[0]), int(row[1]), int(row[2]), matrix)

            p_value = pvalue_AB_AC_BC(cont_cube)

            if p_value is None:
                none_count += 1
            buffer.append([int(row[0]), int(row[1]), int(row[2]), float(row[3]), float(row[4]), float(row[5]), p_value])
            count += 1

            if count == bufferlimit:
                writer.writerows(buffer)
                count = 0
                # empty the buffer
                buffer = []


        writer.writerows(buffer)
        return none_count

def count_triangles_csv(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        row_count = -1
        for row in tqdm(reader):
            row_count +=1

    return row_count

def extract_2_simplex_from_csv_dictionary(alpha,  matrix, dictionary, csvfilename, savename):

    with open(csvfilename, 'r') as csvfile, open(savename + '.csv', 'w',  newline='') as fout:
        reader = csv.reader(csvfile)
        writer = csv.writer(fout)
        writer.writerows([['node index 1', 'node index 2', 'node index 3', 'phi 1-2', 'phi 1-3', 'phi 2-3', 'p-value']])
        next(reader)
        for row in tqdm(reader):
            computed_cont_table = get_cont_cube(int(row[0]), int(row[1]), int(row[2]), matrix)
            table_str = str(int(computed_cont_table[0, 0, 0])) + '_' + str(
                int(computed_cont_table[0, 0, 1])) + '_' + str(int(computed_cont_table[0, 1, 0])) + '_' + str(
                int(computed_cont_table[0, 1, 1])) + '_' + str(int(computed_cont_table[1, 0, 0])) + '_' + str(
                int(computed_cont_table[1, 0, 1])) + '_' + str(int(computed_cont_table[1, 1, 0])) + '_' + str(
                int(computed_cont_table[1, 1, 1]))
            p = dictionary[table_str][1]
            try:
                if p < alpha:
                    writer = csv.writer(fout)
                    writer.writerow([int(row[0]), int(row[1]), int(row[2]), float(row[3]), float(row[4]), float(row[5]), p])
            except:
                pass


def extract_converged_triangles(csvfilename, savename):

    with open(csvfilename, 'r') as csvfile, open(savename + '.csv', 'w',  newline='') as fout:
        reader = csv.reader(csvfile)
        writer = csv.writer(fout)
        writer.writerows([['node index 1', 'node index 2', 'node index 3', 'phi 1-2', 'phi 1-3', 'phi 2-3', 'p-value']])
        next(reader)
        for row in tqdm(reader):
            try :
                p = float(row[6])
                writer = csv.writer(fout)
                writer.writerow([int(row[0]), int(row[1]), int(row[2]), float(row[3]), float(row[4]), float(row[5]), p])
            except:
                pass


def find_unique_tables(matrix, save_name):

    table_set = set()

    # Finds all unique tables

    for one_simplex in tqdm(itertools.combinations(range(matrix.shape[0]), 2)):

        computed_cont_table = get_cont_table(one_simplex[0], one_simplex[1], matrix)
        #computed_cont_table = computed_cont_table.astype(int)

        table_str = str(computed_cont_table[0, 0]) + '_' + str(computed_cont_table[0, 1]) + '_' + str(
            computed_cont_table[1, 0]) + '_' + str(computed_cont_table[1, 1])

        if table_str not in table_set:
            table_set.add(table_str)

    table_set = list(table_set)
    print('How many different tables : ', len(table_set))
    json.dump(table_set, open(save_name + "_table_list.json", 'w'))


def pvalues_for_tables(file_name, nb_samples):

    with open(file_name + "_table_list.json") as json_file:
        table_set = json.load(json_file)

        #### From the different tables : generate the chisqdist :

        pvaldictio = {}

        # Max index used in range() :
        for it in tqdm(range(len(table_set))):
            table_id = table_set[it]
            table = np.random.rand(2,2)
            table_id_list = str.split(table_id, '_')
            table[0, 0] = int(table_id_list[0])
            table[0, 1] = int(table_id_list[1])
            table[1, 0] = int(table_id_list[2])
            table[1, 1] = int(table_id_list[3])

            N = np.sum(table)
            expected1 = mle_2x2_ind(table)
            problist = mle_multinomial_from_table(expected1)
            sample = multinomial_problist_cont_table(N, problist, nb_samples)

            expec = np.tile(expected1, (nb_samples, 1)).reshape(nb_samples, 2,2)
            chisq = chisq_formula_vector(sample, expec)

            pvaldictio[table_id] = sampled_chisq_test(table, expected1, chisq)

        json.dump(pvaldictio, open(file_name + "_exact_pval_dictio.json", 'w'))

def save_pairwise_p_values_phi_dictionary(bipartite_matrix, dictionary, savename):

    # create a CSV file
    with open(savename+'.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows([['node index 1', 'node index 2','phi-coefficient', 'p-value']])

        buffer = []
        for one_simplex in tqdm(itertools.combinations(range(bipartite_matrix.shape[0]), 2)):
            contingency_table = get_cont_table(one_simplex[0], one_simplex[1], bipartite_matrix)
            table_str = str(contingency_table[0, 0]) + '_' + str(contingency_table[0, 1]) + '_' + \
                        str( contingency_table[1, 0]) + '_' + str(contingency_table[1, 1])

            phi = phi_coefficient_table(contingency_table)

            chi2, p = dictionary[table_str]
            buffer.append([one_simplex[0], one_simplex[1], phi, p])
            writer = csv.writer(csvFile)
            writer.writerows(buffer)

            # empty the buffer
            buffer = []

        writer = csv.writer(csvFile)
        writer.writerows(buffer)


def find_unique_cubes(matrix, save_name):

    table_set = set()

    for two_simplex in tqdm(itertools.combinations(range(matrix.shape[0]), 3)):
        cont_cube = get_cont_cube(two_simplex[0], two_simplex[1], two_simplex[2], matrix)

        table_str = str(int(cont_cube[0, 0, 0])) + '_' + str(int(cont_cube[0, 0, 1])) + '_' + str(int(cont_cube[0, 1, 0])) + '_' + str(int(cont_cube[0, 1, 1])) + '_' + str(int(cont_cube[1, 0, 0])) + '_' + str(int(cont_cube[1, 0, 1])) + '_' + str(int(cont_cube[1, 1, 0])) + '_' + str(int(cont_cube[1, 1, 1]))
        if table_str not in table_set:
            table_set.add(table_str)

    table_set = list(table_set)
    print('How many different cubes : ', len(table_set))
    json.dump(table_set, open(save_name + "_cube_list.json", 'w'))

def pvalues_for_cubes(file_name, nb_sample):

    with open(file_name + '_cube_list.json') as json_file:
        table_set = json.load(json_file)

        #### From the different tables : generate the chisqdist :

        pvaldictio = {}

        for it in tqdm(range(len(table_set))):

            table_id = table_set[it]
            table = np.random.rand(2, 2, 2)
            table_id_list = str.split(table_id, '_')
            table[0, 0, 0] = int(table_id_list[0])
            table[0, 0, 1] = int(table_id_list[1])
            table[0, 1, 0] = int(table_id_list[2])
            table[0, 1, 1] = int(table_id_list[3])
            table[1, 0, 0] = int(table_id_list[4])
            table[1, 0, 1] = int(table_id_list[5])
            table[1, 1, 0] = int(table_id_list[6])
            table[1, 1, 1] = int(table_id_list[7])

            N = np.sum(table)
            expected_original = iterative_proportional_fitting_AB_AC_BC_no_zeros(table)

            if expected_original is not None:
                problist = mle_multinomial_from_table(expected_original)
                sample = multinomial_problist_cont_cube(N, problist, nb_sample)
                expec = np.tile(expected_original, (nb_samples, 1, 1)).reshape(nb_sample, 2, 2, 2)
                chisq = chisq_formula_vector_for_cubes(sample, expec)
                s, p = sampled_chisq_test(table, expected_original, chisq)
                pvaldictio[table_id] = p

            else:
                pvaldictio[table_id] = 'None'

        json.dump(pvaldictio, open(data_name + "_exact_cube_pval_dictio.json", 'w'))


def save_triplets_p_values_dictionary(bipartite_matrix, dictionary, savename):

    # create a CSV file
    with open(savename +'.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows([['node index 1', 'node index 2', 'node index 3', 'p-value']])


        for two_simplex in tqdm(itertools.combinations(range(bipartite_matrix.shape[0]), 3)):
            cont_cube = get_cont_cube(two_simplex[0], two_simplex[1], two_simplex[2], bipartite_matrix)

            table_str = str(int(cont_cube[0, 0, 0])) + '_' + str(int(cont_cube[0, 0, 1])) + '_' + str(
                int(cont_cube[0, 1, 0])) + '_' + str(int(cont_cube[0, 1, 1])) + '_' + str(
                int(cont_cube[1, 0, 0])) + '_' + str(int(cont_cube[1, 0, 1])) + '_' + str(
                int(cont_cube[1, 1, 0])) + '_' + str(int(cont_cube[1, 1, 1]))

            try :
                chi2, p = dictionary[table_str]
            except:
                p = dictionary[table_str]

            writer.writerow([two_simplex[0], two_simplex[1], two_simplex[2], p])


def two_simplex_from_csv(csvfilename, alpha, savename):

    with open(csvfilename, 'r') as csvfile, open(savename + '.csv', 'w',  newline='') as fout:
        reader = csv.reader(csvfile)
        writer = csv.writer(fout)
        writer.writerows([['node index 1', 'node index 2', 'node index 3', 'p-value']])
        next(reader)
        for row in tqdm(reader):
            try :
                p = float(row[-1])
                if p < alpha:
                    writer = csv.writer(fout)
                    writer.writerow([int(row[0]), int(row[1]), int(row[2]),  p])
            except:
                pass

def build_facet_list(matrix, two_simplices_file, one_simplices_file, alpha):
    #open('facet_list.txt', 'a').close()

    with open('facet_list.txt', 'w') as facetlist:

        nodelist = np.arange(0, matrix.shape[0])

        for node in nodelist:
            facetlist.write(str(node) +'\n')

        with open(one_simplices_file, 'r') as csvfile:

            reader = csv.reader(csvfile)
            next(reader)

            for row in tqdm(reader):

                try:
                    p = float(row[-1])
                    if p < alpha:
                        # Reject H_0 in which we suppose that u and v are independent
                        # Thus, we accept H_1 and add a link between u and v in the graph to show their dependency
                        facetlist.write(str(row[0]) + ' ' + str(row[1]) + '\n')
                except:
                    pass

        with open(two_simplices_file, 'r') as csvfile:

            reader = csv.reader(csvfile)
            next(reader)

            for row in tqdm(reader):

                try:
                    p = float(row[-1])
                    if p < alpha:
                        # Reject H_0 in which we suppose that u and v are independent
                        # Thus, we accept H_1 and add a link between u and v in the graph to show their dependency
                        facetlist.write(str(row[0]) + ' ' + str(row[1]) + ' ' + str(row[2]) + '\n')
                except:
                    pass
    return

def triangles_p_values_AB_AC_BC_dictionary(csvfile, savename, dictionary, matrix, bufferlimit=100000):

    buffer = []

    with open(csvfile, 'r') as csvfile, open(savename + '.csv', 'w',  newline='') as fout:
        reader = csv.reader(csvfile)
        writer = csv.writer(fout)
        writer.writerows([['node index 1', 'node index 2', 'node index 3', 'p-value']])
        next(reader)
        for row in tqdm(reader):

            cont_cube = get_cont_cube(int(row[0]), int(row[1]), int(row[2]), matrix)

            table_str = str(int(cont_cube[0, 0, 0])) + '_' + str(int(cont_cube[0, 0, 1])) + '_' + str(
                int(cont_cube[0, 1, 0])) + '_' + str(int(cont_cube[0, 1, 1])) + '_' + str(
                int(cont_cube[1, 0, 0])) + '_' + str(int(cont_cube[1, 0, 1])) + '_' + str(
                int(cont_cube[1, 1, 0])) + '_' + str(int(cont_cube[1, 1, 1]))

            try :

                chi2, p = dictionary[table_str]

            except:

                p = dictionary[table_str]

            writer.writerow([row[0], row[1], row[2], p])

def extract_2_simplex_from_csv(csvfilename, alpha, savename):

    with open(csvfilename, 'r') as csvfile, open(savename + '.csv', 'w',  newline='') as fout:
        reader = csv.reader(csvfile)
        writer = csv.writer(fout)
        writer.writerows([['node index 1', 'node index 2', 'node index 3', 'p-value']])
        next(reader)
        for row in tqdm(reader):
            try :
                p = float(row[-1])
                if p < alpha:
                    writer = csv.writer(fout)
                    writer.writerow([int(row[0]), int(row[1]), int(row[2]), p])
            except:
                pass

def statistics_for_cubes(file_name, nb_sample):

    with open(file_name + '_cube_list.json') as json_file:
        table_set = json.load(json_file)

        #### From the different tables : generate the chisqdist :

        pvaldictio = {}

        for it in tqdm(range(len(table_set))):

            table_id = table_set[it]
            table = np.random.rand(2, 2, 2)
            table_id_list = str.split(table_id, '_')
            table[0, 0, 0] = int(table_id_list[0])
            table[0, 0, 1] = int(table_id_list[1])
            table[0, 1, 0] = int(table_id_list[2])
            table[0, 1, 1] = int(table_id_list[3])
            table[1, 0, 0] = int(table_id_list[4])
            table[1, 0, 1] = int(table_id_list[5])
            table[1, 1, 0] = int(table_id_list[6])
            table[1, 1, 1] = int(table_id_list[7])

            N = np.sum(table)
            expected_original = iterative_proportional_fitting_AB_AC_BC_no_zeros(table)

            if expected_original is not None:
                problist = mle_multinomial_from_table(expected_original)
                sample = multinomial_problist_cont_cube(N, problist, nb_sample)
                expec = np.tile(expected_original, (nb_samples, 1, 1)).reshape(nb_sample, 2, 2, 2)
                chisq = chisq_formula_vector_for_cubes(sample, expec)

                return chisq


if __name__ == '__main__':
    old_way = False
    dirName = 'vOTUS_fortest'
    data_name = 'vOTUS'

    alpha = 0.01
    nb_samples = 1000000
    matrix1 = np.load('vOTUS_occ.npy').T
    matrix1 = matrix1.astype(np.int64)

    #build_facet_list(matrix1, r'D:\Users\Xavier\Documents\Analysis_master\Analysis\clean_analysis\vOTUS\vOTUS_exact_cube_pvalues.csv', r'D:\Users\Xavier\Documents\Analysis_master\Analysis\clean_analysis\vOTUS\vOTUS_exact_pvalues.csv', 0.01 )
    #exit()
    # Create target Directory if don't exist
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")

    data_name = os.path.join(dirName, data_name)


    ####### First step : Extract all the unique tables

    print('Step 1 : Extract all the unique tables')

    # Finds all unique tables
    find_unique_tables(matrix1, data_name)

    ######## Second step : Extract pvalues for all tables with an exact Chi3 distribution

    print('Step 2: Extract pvalues for all tables with an exact Chi3 distribution')

    pvalues_for_tables(data_name, nb_samples)

    ######## Third step : Find table for all links and their associated pvalue

    print('Step 3 : Find table for all links and their associated pvalue')

    with open(data_name + '_exact_pval_dictio.json') as jsonfile:
        dictio = json.load(jsonfile)

        save_pairwise_p_values_phi_dictionary(matrix1, dictio, data_name + '_exact_pvalues')


    ######## Fourth step : Choose alpha and extract the network

    print('Step 4 : Generate network and extract edge_list for a given alpha')

    g = read_pairwise_p_values(data_name + '_exact_pvalues.csv', alpha)
    nx.write_edgelist(g, data_name + '_exact_edge_list_' + str(alpha)[2:] + '.txt', data=True)

    print('Number of nodes : ', g.number_of_nodes())
    print('Number of links : ', g.number_of_edges())

    ######## Fifth step : Extract all the unique cubes

    print('Step 5 : Extract all the unique cubes')

    find_unique_cubes(matrix1, data_name)

    ###### Sixth step : Extract pvalues for all cubes with an exact CHI 3 distribution

    print('Step 6: Extract pvalues for all tables with an exact CHI 3 distribution')

    pvalues_for_cubes(data_name, nb_samples)

    ######## Seventh step : Find cube for all triplets and their associated pvalue

    if not old_way:

        print('Step 7 : Find cube for all triplets and their associated pvalue')

        with open(data_name + "_exact_cube_pval_dictio.json") as jsonfile:
            dictio = json.load(jsonfile)

            save_triplets_p_values_dictionary(matrix1, dictio, data_name + '_exact_cube_pvalues')

        two_simplex_from_csv(data_name + '_exact_cube_pvalues.csv', alpha, data_name + '_exact_two_simplices_'  + str(alpha)[2:])

        exit()

    else:
        print('OLD WAY : ')

    ############## OLD WAY : FIND EMPTY TRIANGLES AND TEST THEM : ###################


    ######## Fifth step : Find all triangles in the previous network

        print('Finding all empty triangles in the network')

        g = read_pairwise_p_values(data_name + '_exact_pvalues.csv', alpha)

        save_all_triangles(g, data_name + '_exact_triangles_' + str(alpha)[2:])

        print('Number of triangles : ', count_triangles_csv(data_name + '_exact_triangles_' + str(alpha)[2:] + '.csv'))

    ######## Sixth step : Find all the p-values for the triangles under the hypothesis of homogeneity

        print('Find all the p-values for the triangles under the hypothesis of homogeneity')

        with open(data_name + "_exact_cube_pval_dictio.json") as jsonfile:
            dictio = json.load(jsonfile)

            triangles_p_values_AB_AC_BC_dictionary(data_name + '_exact_triangles_' + str(alpha)[2:] + '.csv', data_name + '_exact_triangles_' + str(alpha)[2:] + '_pvalues.csv', dictio, matrix1)

    ######## Fifth step : Extract all 2-simplices

        print('Extract 2-simplices')

        extract_2_simplex_from_csv(data_name + '_exact_triangles_' + str(alpha)[2:] + '_pvalues.csv', alpha, data_name + '_exact_2-simplices_' + str(alpha)[2:])

    ################# DONE ###################



    #### to json for d3js :

    #g = read_pairwise_p_values('exact_chi1_pvalues_birds.csv', 0.01)
    #label_list = []
    #with open('groupes_otu.csv', 'r') as csvfile:
    #   reader = csv.reader(csvfile)
    #   for row in reader:
    #       try:
    #           label_list.append(row[1])
    #       except:
    #           if row[0] != 'Bacteria':
    #               label_list.append(row[0])
    #           else:
    #               label_list.append(label_list[-1])

    #node_dictio_list = []
    #for noeud in g.nodes:
    #   node_dictio_list.append({"id": str(noeud), "group": 2})
    #   #node_dictio_list.append({"id":str(noeud)})

    #link_dictio_list = []
    #for lien in g.edges:
    #   link_dictio_list.append({"source": str(lien[0]), "target": str(lien[1]), "value": 1})

    #triplex_dictio_list = []

    #with open('all_cube_pval_dictionary') as jsonfile:
    #    dictio = json.load(jsonfile)

    #matrix1 = np.loadtxt('incidenceMatrix.txt').T
    #matrix1 = matrix1.astype(int)
    #table_set = set()
    #for two_simplex in tqdm(itertools.combinations(range(matrix1.shape[0]), 3)):

    #    computed_cont_table = get_cont_cube(two_simplex[0], two_simplex[1], two_simplex[2], matrix1)
    #    table_str = str(int(computed_cont_table[0, 0, 0])) + '_' + str(
    #        int(computed_cont_table[0, 0, 1])) + '_' + str(int(computed_cont_table[0, 1, 0])) + '_' + str(
    #        int(computed_cont_table[0, 1, 1])) + '_' + str(int(computed_cont_table[1, 0, 0])) + '_' + str(
    #        int(computed_cont_table[1, 0, 1])) + '_' + str(int(computed_cont_table[1, 1, 0])) + '_' + str(
    #        int(computed_cont_table[1, 1, 1]))
    #    if table_str not in table_set:
    #        table_set.add(table_str)
    #    try:
    #        if float(dictio[table_str][1]) < 0.01:
    #            print(two_simplex[0], two_simplex[1], two_simplex[2], dictio[table_str][1])
    #            triplex_dictio_list.append({"nodes": [str(two_simplex[0]), str(two_simplex[1]), str(two_simplex[2])]})
    #            link_dictio_list.append({"source": str(two_simplex[0]), "target": str(two_simplex[1]), "value": 1})
    #            link_dictio_list.append({"source": str(two_simplex[1]), "target": str(two_simplex[2]), "value": 1})
    #            link_dictio_list.append({"source": str(two_simplex[0]), "target": str(two_simplex[2]), "value": 1})
    #    except:
    #        pass


    #json_diction = {"nodes": node_dictio_list, "links" : link_dictio_list, "triplex" : triplex_dictio_list}
    #with open('d3js_simplicialcomplex_01.json', 'w') as outfile:
    #   json.dump(json_diction, outfile)
    #exit()
    # Extract nodes with groups :
    ######groupe_set = set()
    ######with open('groupes_otu.csv', 'r') as csvfile:
    ######    reader = csv.reader(csvfile)
    ######    for row in reader:
    ######        try:
    ######            groupe_set.add(row[1])
    ######        except:
    ######            if row[0] != 'Bacteria':
    ######                groupe_set.add(row[0])
    ######print(len(groupe_set))
    ######print(groupe_set)
    # node_dictio_list = []
    # label_list = []
    # with open('groupes_otu.csv', 'r') as csvfile:
    #    reader = csv.reader(csvfile)
    #    for row in reader:
    #        try:
    #            label_list.append(row[1])
    #        except:
    #            if row[0] != 'Bacteria':
    #                label_list.append(row[0])
    #            else:
    #                label_list.append(label_list[-1])

    ############## D3JS NEGATIVE INTERACTIONS
    # link_dictio_list = []
    # node_set = set()
    # count = 0
    # for lien in g.edges:
    #    if g.get_edge_data(lien[0], lien[1])['phi'] < 0 :
    #        count += 1
    #        #link_dictio_list.append({"source": str(lien[0]), "value": g.get_edge_data(lien[0], lien[1])['phi'], "target": str(lien[1])})
    #        link_dictio_list.append({"source": str(lien[0]), "value": 1, "target": str(lien[1])})
    #        node_set.add(lien[0])
    #        node_set.add(lien[1])
    # print('NUmber of negative interactions : ', count)

    # node_dictio_list = []
    # for noeud in list(node_set):
    #    node_dictio_list.append({"id": str(noeud), "group": label_list[noeud]})
    #    #node_dictio_list.append({"id": str(noeud)})

    # json_diction = {"nodes": node_dictio_list, "links": link_dictio_list}
    # with open('d3js_exact_chi1_negative_interactions_final_otu_with_groups.json', 'w') as outfile:
    #    json.dump(json_diction, outfile)

    # exit()
    # link_dictio_list = []
    # node_set = set()
    # for lien in g.edges:
    #    if g.get_edge_data(lien[0], lien[1])['phi'] > 0:
    #        #link_dictio_list.append(
    #        #    {"source": str(lien[0]), "value": g.get_edge_data(lien[0], lien[1])['phi'], "target": str(lien[1])})
    #        link_dictio_list.append(
    #            {"source": str(lien[0]), "value": 1, "target": str(lien[1])})
    #        node_set.add(lien[0])
    #        node_set.add(lien[1])

    # node_dictio_list = []
    # for noeud in list(node_set):
    #    node_dictio_list.append({"id": str(noeud), "group": label_list[noeud]})
    #    #node_dictio_list.append({"id": str(noeud)})

    # json_diction = {"nodes": node_dictio_list, "links": link_dictio_list}
    # with open('positive_interactions_otu_grouped.json', 'w') as outfile:
    #    json.dump(json_diction, outfile)

    # exit()


    #print('Step 6: Extract pvalues for all tables with an exact CHI 3 distribution')


    #def pvalues_for_cubes_fig(file_name, nb_sample):

    #    with open(file_name + '_cube_list.json') as json_file:
    #        table_set = json.load(json_file)

    #        #### From the different tables : generate the chisqdist :

    #        for it in tqdm(range(len(table_set))):

    #            table_id = table_set[it]
    #            table = np.random.rand(2, 2, 2)
    #            table_id_list = str.split(table_id, '_')
    #            table[0, 0, 0] = int(table_id_list[0])
    #            table[0, 0, 1] = int(table_id_list[1])
    #            table[0, 1, 0] = int(table_id_list[2])
    #            table[0, 1, 1] = int(table_id_list[3])
    #            table[1, 0, 0] = int(table_id_list[4])
    #            table[1, 0, 1] = int(table_id_list[5])
    #            table[1, 1, 0] = int(table_id_list[6])
    #            table[1, 1, 1] = int(table_id_list[7])

    #            N = np.sum(table)
    #            expected_original = iterative_proportional_fitting_AB_AC_BC_no_zeros(table)

    #            if expected_original is not None:
    #                problist = mle_multinomial_from_table(expected_original)
    #                sample = multinomial_problist_cont_cube(N, problist, nb_sample)
    #                expec = np.tile(expected_original, (nb_samples, 1, 1)).reshape(nb_sample, 2, 2, 2)
    #                chisq = chisq_formula_vector_for_cubes(sample, expec)

    #                return chisq


    #chisq = pvalues_for_cubes_fig(data_name, nb_samples)

    #df = 1


    #fig, ax = plt.subplots(1, 1)
    #bins2 = np.arange(0, max(chisq) + 1, 0.1)

    ## plt.hist(chisqlist_chi1, bins=bins2, weights=np.repeat(1/len(chisqlist_chi1), len(chisqlist_chi1)), alpha=0.5, label='EXACT')
    #plt.hist(chisq, bins=bins2, density=True, alpha=0.5, label='Exact')
    #x = np.arange(0, 20, 0.01)
    #ax.plot(x, chi2.pdf(x, 1), label='1 Degree')
    #ax.plot(x, chi2.pdf(x, 3), label='3 Degrees')
    #ax.plot(x, chi2.pdf(x, 7), label='7 Degrees')
    ## bins2 = np.arange(0, max(chisqlist) + 5, 1)
    ## print(max(chisqlist))
    ## plt.hist(chisqlist, bins=bins2, alpha=0.5)
    #plt.xlim([0,20])
    #plt.ylim([0,1])
    #plt.legend()
    #plt.show()