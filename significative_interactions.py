import numpy as np
import pandas as pd
import collections
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

def pvalue_AB_AC_BC(cont_cube):
    expected = iterative_proportional_fitting_AB_AC_BC_no_zeros(cont_cube)
    if expected is not None:
        return chisq_test(cont_cube, expected)[1]
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

def save_pairwise_p_values_phi(bipartite_matrix, savename, bufferlimit=100000):

    # create a CSV file
    with open(savename+'.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows([['node index 1', 'node index 2', 'p-value', 'phi-coefficient']])

    buffer = []
    count = 0
    for one_simplex in tqdm(itertools.combinations(range(bipartite_matrix.shape[0]), 2)):
        contingency_table = get_cont_table(one_simplex[0], one_simplex[1], bipartite_matrix)
        expected_table = mle_2x2_ind(contingency_table)
        phi = phi_coefficient_table(contingency_table)
        chi2, p = chisq_test(contingency_table, expected_table)
        buffer.append([one_simplex[0], one_simplex[1], p, phi])
        count += 1
        if count == bufferlimit:
            with open(savename+'.csv', 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerows(buffer)
                count = 0
                # empty the buffer
                buffer = []

    with open(savename + '.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(buffer)

def read_pairwise_p_values(filename, alpha=0.01):

    graph = nx.Graph()

    with open(filename, 'r') as csvfile:

        reader = csv.reader(csvfile)
        next(reader)

        for row in tqdm(reader):

            if row[-1] != 'nan':
                p = float(row[-2])
                if p < alpha:
                    # Reject H_0 in which we suppose that u and v are independent
                    # Thus, we accept H_1 and add a link between u and v in the graph to show their dependency
                    graph.add_edge(int(row[0]), int(row[1]), phi=float(row[-1]))

    return graph

def find_n_simplices_vector(vector, n):

    # Indices of the non zero vector entries. We use this because it would be a waste of time to check between every
    # group of n entries since we know that we wont count groups with entries that are zero.
    all_line_indices = np.where(vector > 0)[0]
    # Thus, all the len(all_line_indices) choose n groups that we can build have non zero entries in the vector. With
    # itertools, we can get the indices of the nodes in each group and save them
    tuple_list = []
    #count = 0

    for simplex in itertools.combinations(all_line_indices, n):
        tuple_list.append(frozenset(simplex))
        #count += 1

    return tuple_list#, count

def find_simplices_matrix(matrix, n, savepath=''):
    for column in range(matrix.shape[1]):
        with open(savepath + 'matrix_' +str(column)+'.pickle', 'wb') as file:
            print('Finding simplices in column : ', column)
            pickle.dump(find_n_simplices_vector(matrix[:,column], n), file)

def count_labeled_simplices(loadpath, savepath=''):
    """
    :param loadpath:
    :param savepath:
    :return:
    """


    # Initializes variables
    original_list = None
    comparison_list = None
    simplex_dictionary = {}

    #for each file in the target directory
    for filename in os.listdir(loadpath):
        if filename.endswith(".pickle"):
            print('Treating file : ', filename)

            with open(loadpath + filename, 'rb') as f:
                data = pickle.load(f)
                # if it is None, it's the first data we load in
                if original_list is None:
                    original_list = data

                    #We add all these simplices to the dictionary with entry 1, meaning we counted it once
                    for simplex in original_list:
                        simplex_dictionary[simplex] = 1
                else:
                    comparison_list = data
        # This condition ensures that we begin once both original_list and comparison_list are lists
        if comparison_list is not None:
            #While loop is used because of the fact that we use the .remove method, which causes problems with for loops
            i = 0
            while i < len(comparison_list):
                #Assign a simplex
                simplex = comparison_list[i]

                try:
                    # If the simplex is already in the dictionary, we just add 1 to the counter.
                    simplex_dictionary[simplex] += 1
                    #comparison_list.remove(simplex)
                except:
                    # If it is not, we need to add it in the dictionary
                    simplex_dictionary[simplex] = 1
                    i += 1


                #if simplex in original_list:
                #    # If the simplex is already in the dictionary, we just add 1 to the counter.
                #    simplex_dictionary[simplex] += 1
                #    comparison_list.remove(simplex)
                #else:
                #    # If it is not, we need to add it in the dictionary
                #    simplex_dictionary[simplex] = 1
                #    i += 1

            # The next original list is a concatenation of the previous original list and the comparison list in which
            # we removed the duplicates (meaning that the resulting comparison_list only contains simplices that did not
            # appear in the dictionary before this iteration).
            #original_list = original_list + comparison_list
    # save the dictionary.
    with open(savepath + 'simplexdictionary.pickle', 'wb') as file:
        pickle.dump(simplex_dictionary, file)


def find_n_simplices_vector_dictio(vector, n, dictionary):


    # Indices of the non zero vector entries. We use this because it would be a waste of time to check between every
    # group of n entries since we know that we wont count groups with entries that are zero.
    all_line_indices = np.where(vector > 0)[0]
    # Thus, all the len(all_line_indices) choose n groups that we can build have non zero entries in the vector. With
    # itertools, we can get the indices of the nodes in each group and save them
    tuple_list = []
    #count = 0

    for simplex in itertools.combinations(all_line_indices, n):
        try:
            # If the simplex is already in the dictionary, we just add 1 to the counter.
            dictionary[frozenset(simplex)] += 1
            # comparison_list.remove(simplex)
        except:
            # If it is not, we need to add it in the dictionary
            dictionary[frozenset(simplex)] = 1

    return dictionary

def find_simplices_matrix_dictio(matrix, n, savepath=''):
    simplex_dictionary = {}
    for column in range(matrix.shape[1]):
        print('Finding simplices in column : ', column)
        simplex_dictionary = find_n_simplices_vector_dictio(matrix[:, column], n, simplex_dictionary)

    with open(savepath + 'simplexdictionary.pickle', 'wb') as file:
        print('Writting dictionary.')
        pickle.dump(simplex_dictionary, file)

def histo_nb_appearences(simplex_dictionary, savename='bcounts'):
    np.save(savename, np.bincount(np.array(list(simplex_dictionary.values()))))
    return


def get_keys_for_value(simplex_dictionary, value, savename='keylist.pickle'):
    with open(savename, 'wb') as f:
        pickle.dump([key for (key, val) in simplex_dictionary.items() if val == value], f)

def histo_prob_dist(bincount, total_number_of_groups):
    bincount[0] = total_number_of_groups - np.sum(bincount)
    bincount = bincount/np.sum(bincount)
    return bincount

def cumulative_dist(prob_dist):

    cumulative = []
    for i in range(len(prob_dist)):
        if i > 0:
            cumulative.append(np.sum(prob_dist[:i]))

    return np.array(cumulative)

def significant_edge(graph, u_idx, v_idx, contingency_table, alpha = 0.05):
    chi2, p, dof, ex = sp.stats.chi2_contingency(contingency_table)
    if p > alpha :
        # Cannot reject H_0 in which we suppose that u and v are independent
        # Thus, we do not add the link between u and v in the graph
        pass
    else:
        #print('Rejected H0')
        # Reject H_0 and accept H_1 in which we suppose that u and v are dependent
        # Thus, we add a link between u and v in the graph.
        graph.add_edge(u_idx, v_idx)

def save_triplets_p_values(bipartite_matrix, nodelist, savename, bufferlimit=100000):

    # create a CSV file
    with open(savename+'.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(['node index 1', 'node index 2', 'node index 3', 'p-value'])

    buffer = []
    count = 0
    for two_simplex in tqdm(itertools.combinations(nodelist, 3)):
        contingency_table = get_cont_cube(two_simplex[0], two_simplex[1], two_simplex[2], bipartite_matrix)

        p = chisq_test(contingency_table)
        buffer.append([two_simplex[0], two_simplex[1], two_simplex[2], p])
        count += 1
        if count == bufferlimit:
            with open(savename+'.csv', 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerows(buffer)
                count = 0
                # empty the buffer
                buffer = []

    with open(savename + '.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(buffer)


def disconnected_network(nodes):
    g = nx.Graph()
    g.add_nodes_from(np.arange(nodes))
    return g

def build_network(g, matrix, alph):
    nb_of_simplices = sp.misc.comb(matrix1.shape[0], 2)
    count = 0
    for one_simplex in tqdm(itertools.combinations(range(matrix1.shape[0]), 2)):
        #print(str(count) + ' ouf of ' + str(nb_of_simplices))
        contigency_table = get_cont_table(one_simplex[0], one_simplex[1], matrix)
        significant_edge(g, one_simplex[0], one_simplex[1], contigency_table, alpha=alph)
        count += 1

    return g

def save_all_triangles(G, savename, bufferlimit=100000):
    G = copy.deepcopy(G)
    with open(savename + '.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows([['node index 1', 'node index 2', 'node index 3']])
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

                    buffer.append([node, n1, n2])
                    count += 1

            G.remove_node(node)

            if count == bufferlimit:
                with open(savename + '.csv', 'a') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerows(buffer)
                    count = 0
                    # empty the buffer
                    buffer = []

    with open(savename + '.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(buffer)

def save_all_new_triangles(G, csvtocomparewith, savename, bufferlimit=100000):
    G = copy.deepcopy(G)
    with open(savename + '.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows([['node index 1', 'node index 2', 'node index 3']])
    buffer = []
    # Iterate over all possible triangle relationship combinations
    count = 0
    with open(csvtocomparewith + '.csv', 'w') as compare:
        existinglines = {tuple(line) for line in csv.reader(compare, delimiter=',')}
        for node in list(G.nodes):
            if G.degree[node] < 2:
                G.remove_node(node)
            else:
                for n1, n2 in itertools.combinations(G.neighbors(node), 2):

                    # Check if n1 and n2 have an edge between them
                    if G.has_edge(n1, n2):
                        buffer.append([node, n1, n2])
                        count += 1

                G.remove_node(node)

                if count == bufferlimit:
                    with open(savename + '.csv', 'a') as csvFile:
                        writer = csv.writer(csvFile)
                        writer.writerows(buffer)
                        count = 0
                        # empty the buffer
                        buffer = []

        with open(savename + '.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(buffer)

def save_all_n_cliques(G, n, savename, bufferlimit=100000):
    G = copy.deepcopy(G)
    with open(savename + '.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        headerlist = []
        for i in range(n):
            headerlist.append('node index ' + str(i))
        writer.writerows([headerlist])
    buffer = []
    # Iterate over all possible triangle relationship combinations
    count = 0
    for node in list(G.nodes):
        if G.degree[node] < n - 1:
            G.remove_node(node)
        else:
            for neighbors in itertools.combinations(G.neighbors(node), n - 1):

                # Check if all pairs have an edge between them
                switch = True
                for pair in itertools.combinations(neighbors, 2):

                    if G.has_edge(pair[0], pair[1]):
                        pass

                    else :
                        switch = False
                        break
                if switch :
                    nodes_to_add_list = [node]
                    for node_i in neighbors:
                        nodes_to_add_list.append(node_i)
                    buffer.append(nodes_to_add_list)
                    count += 1
                    if count == bufferlimit:
                        with open(savename + '.csv', 'a') as csvFile:
                            writer = csv.writer(csvFile)
                            writer.writerows(buffer)
                            count = 0
                            # empty the buffer
                            buffer = []


            G.remove_node(node)

    with open(savename + '.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(buffer)

def triangles_p_values_AB_AC_BC(csvfile, savename, matrix, bufferlimit=100000):

    buffer = []

    with open(csvfile, 'r') as csvfile, open(savename, 'w') as fout:
        reader = csv.reader(csvfile)
        writer = csv.writer(fout)
        writer.writerows([['node index 1', 'node index 2', 'node index 3', 'p-value']])
        count = 0
        none_count = 0
        next(reader)
        for row in tqdm(reader):

            cont_cube = get_cont_cube(int(row[0]), int(row[1]), int(row[2]), matrix)

            p_value = pvalue_AB_AC_BC(cont_cube)

            if p_value is not None:
                buffer.append([int(row[0]), int(row[1]), int(row[2]), p_value])
                count += 1
            else :
                none_count += 1

            if count == bufferlimit:
                with open(savename + '.csv', 'a') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerows(buffer)
                    count = 0
                    # empty the buffer
                    buffer = []


        writer = fout.writer(csvFile)
        writer.writerows(buffer)
        return none_count


def new_triangles_p_values_AB_AC_BC(csvfile, csvtocomparewith, savename, matrix, bufferlimit=100000):

    buffer = []

    with open(csvfile, 'r') as csvfile, open(savename, 'w') as fout, open(csvtocomparewith + '.csv', 'w') as compare:
        reader = csv.reader(csvfile)
        writer = csv.writer(fout)
        existinglines = {tuple(line) for line in csv.reader(compare, delimiter=',')}
        writer.writerows([['node index 1', 'node index 2', 'node index 3', 'p-value']])
        count = 0
        none_count = 0
        next(reader)
        for row in tqdm(reader):

            if row not in existinglines:

                cont_cube = get_cont_cube(int(row[0]), int(row[1]), int(row[2]), matrix)

                p_value = pvalue_AB_AC_BC(cont_cube)

                if p_value is not None:
                    buffer.append([int(row[0]), int(row[1]), int(row[2]), p_value])
                    count += 1
                else :
                    buffer.append([int(row[0]), int(row[1]), int(row[2]), str(p_value)])
                    none_count += 1

                if count == bufferlimit:
                    with open(savename + '.csv', 'a') as csvFile:
                        writer = csv.writer(csvFile)
                        writer.writerows(buffer)
                        count = 0
                        # empty the buffer
                        buffer = []


            writer = fout.writer(csvFile)
            writer.writerows(buffer)
            return none_count

def count_triangles_csv(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        row_count = -1
        for row in tqdm(reader):
            row_count +=1

    return row_count

def extract_2_simplex_from_csv(csvfilename, alpha, savename):

    with open(csvfilename, 'r') as csvfile, open(savename + '.csv', 'w') as fout:
        reader = csv.reader(csvfile)
        writer = csv.writer(fout)
        writer.writerows([['node index 1', 'node index 2', 'node index 3', 'p-value']])
        next(reader)
        for row in tqdm(reader):
            if row[-1] != 'nan':
                p = float(row[-1])
                if p < alpha:
                    writer = csv.writer(fout)
                    writer.writerow([int(row[0]), int(row[1]), int(row[2]), p])



if __name__ == '__main__':

    ######## First step : extract all links, their p-value and their phi-coefficient
    #### Load matrix
    #matrix1 = np.loadtxt('final_OTU.txt', skiprows=0, usecols=range(1, 39))

    #### Transform into occurrence matrix
    #matrix1 = to_occurrence_matrix(matrix1, savepath=None)

    #### Save all links and values to CSV file
    #save_pairwise_p_values_phi(matrix1, 'change')
    #exit()

    ######## Second step : Choose alpha and extract the network
    g = read_pairwise_p_values('pairwise_pvalues_phi_final_otu.csv', 0.001)

    print("Network density : ", nx.density(g))
    print("Is connected : ", nx.is_connected(g))
    print("Triadic closure : ", nx.transitivity(g))
    degree_sequence = sorted([d for n, d in g.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt/np.sum(cnt), width=0.80, color='b')
    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    #ax.set_xticks([d + 0.4 for d in deg])
    #ax.set_xticklabels(deg)
    plt.show()

    # Bunch of stats can be found here : https://networkx.github.io/documentation/stable/reference/functions.html

    exit()

    ######## Third step : Find all triangles in the previous network

    g = read_pairwise_p_values('pairwise_pvalues_phi_final_otu.csv', 0.001)
    exit()

    #### TODO Changer la fonction pour sauvegarder les phis associés à chaque pair de lien au sein du triangle
    save_all_triangles(g, 'triangles_001_final_otu')

    exit()


    #with open('/home/xavier/Documents/Projet/Betti_scm/triangles_alpha001.csv', 'r') as file1:
    #    existinglines = {tuple(line) for line in csv.reader(file1, delimiter=',')}
    #existinglines = set(existinglines)
    #print(('295', '2270', '480')in existinglines)
    #print(len(existinglines))

    #exit()
    #extract_2_simplex_from_csv("/home/xavier/Documents/Projet/Betti_scm/triangles_pvalues_alpha001.csv", 0.001, "extracted")
    #exit()
    #xijk = np.ones((2, 2, 2))

    #xijk[0, 0, 0] = 156
    #xijk[0, 1, 0] = 84
    #xijk[0, 0, 1] = 84
    #xijk[0, 1, 1] = 156

    #xijk[1, 0, 0] = 107
    #xijk[1, 1, 0] = 133
    #xijk[1, 0, 1] = 31
    #xijk[1, 1, 1] = 209

    #print(iterative_proportional_fitting_ind(xijk, delta=0.000001))
    #exit()

    #G = nx.Graph()
    #G = nx.complete_graph(3)
    #save_all_triangles(G, 'testcomplete')
    #print(count_triangles_csv('/home/xavier/Documents/Projet/Betti_scm/testcomplete.csv'))
    #print(type(G.nodes))
    #exit()
    #print(nx.triangles(G, 0))

    #print(nx.triangles(G))

    #print(list(nx.triangles(G, (0, 1)).values()))
    #exit()
    #G.add_nodes_from([1,2,3,4,5,6,7])
    #G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9])
    #G.add_nodes_from([1, 2, 3, 4, 5])
    #G.add_edges_from(
    #    [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)])
    #for clique in nx.algorithms.clique.find_cliques(G):
    #    print(clique)
    #G.add_edges_from([(1,2),(2,3),(3,1),(3,4),(3,5),(3,6),(3,7),(4,5),(4,6),(4,7),(5,6),(5,7),(6,7)])
    #G.add_edges_from([(1, 2), (2, 3), (3, 1), (1, 4), (1, 5), (4, 5), (2, 6), (2, 7), (7, 6), (3, 8), (3, 9), (9, 8)])
    #g = read_pairwise_p_values('/home/xavier/Documents/Projet/Betti_scm/pairwise_p_values_vectorized.csv', 0.001)
    #ls = list(g.nodes)
    #matrix1 = np.loadtxt('final_OTU.txt', skiprows=0, usecols=range(1, 39))
    #matrix1 = to_occurrence_matrix(matrix1, savepath=None)
    #cont = get_cont_cube(1, 331, 2151, matrix1)
    #cont[0,1,1] = 0.5
    #cont[1,0,1] = 0.5
    #cont[1,1,0] = 0.5
    #print(cont)
    #print(pvalue_AB_AC_BC(cont))
    #exit()
    #print(triangles_p_values_AB_AC_BC('/home/xavier/Documents/Projet/Betti_scm/triangles_alpha001.csv', 'triangles_pvalues_alpha001donterase', matrix1))
    #exit()
    #save_all_triangles(g, 'triangles_alpha01')
    #8535037 it[33:13, 4281.43it / s]
    #8311398
    #print(count_triangles_csv('/home/xavier/Documents/Projet/Betti_scm/triangles_alpha001.csv'))
    #print(count_triangles_csv('/home/xavier/Documents/Projet/Betti_scm/triangles_pvalues_alpha001.csv'))
    #print(sum(nx.triangles(g).values()) / 3)
    #exit()
    #print(get_nb_cliques_by_length(g,3))
    #exit()
    #print(sum(nx.triangles(g).values()) / 3)
    # For alpha = 0.01 : 2605 nodes (6 unconnected nodes) and 305361 links. For alpha = 0.001 : 2570 nodes (41 unconnected nodes) and 186280 links
    #print(len(list(g.nodes)))
    #print(len(list(g.edges)))
    #i = 0
    #nb_3_cliques = 0
    #for clique in nx.algorithms.clique.find_cliques(g):
        #nb_3_cliques += find_nb_n_cliques(clique, 3)
        #i += 1
    #    print(clique)
    #print(i, nb_3_cliques)
    #exit()
    #    pass
        #print(clique)
    #pos = nx.spring_layout(g)
    #nx.draw_networkx_nodes(g, pos, nodelist=list(g.nodes), node_color='r', node_size=20)
    #nx.draw_networkx_edges(g, pos)
    #plt.show()
    #exit()
    #graf = nx.Graph()
    #g.add_edge(2,3)
    #graf = nx.read_edgelist('alpha001_reduced_graph_edgelist')
    #print(len(list(graf.nodes)))
    #exit()

    #nodelist = list(g.nodes)
    #print(len(nodelist))
    #matrix1 = np.loadtxt('final_OTU.txt', skiprows=0, usecols=range(1, 39))
    #matrix1 = to_occurrence_matrix(matrix1, savepath=None)
    #exit()
    #print(matrix1[917,:])
    #chisq_test(get_cont_table(1,261, matrix1))
    #cont_tab = get_cont_table(0, 768, matrix1)
    #print(chisq_test(cont_tab, mle_2x2_ind(cont_tab)))
    #exit()

    #save_triplets_p_values(matrix1, nodelist, 'triplet_p_value')
    #exit()
    #save_pairwise_p_values_new(matrix1, 'pairwise_p_values_vectorizedwhaton')
    #exit()

    #alph = 0.01
    #g = nx.Graph() #disconnected_network(matrix1.shape[0])
    #build_network(g, matrix1, alph=alph)

    #nx.write_edgelist(g, 'alpha0001_reduced_graph_edgelist')





    #exit()
    #### plot distributions
    #fignum = 1
    #for i in range(0, 3):

        #plt.figure(fignum)
        ##fig, ax = plt.subplots()
        #bcount = np.load(str(i) + 'bcounts.npy')

        ## plot probability distribution
        #matrix1 = np.loadtxt('final_OTU.txt', skiprows=0, usecols=range(1, 39))
        #nb_species = matrix1.shape[0]
        #spec_choose_3 = sp.misc.comb(nb_species, i + 1)
        #prob_dist = histo_prob_dist(bcount, spec_choose_3)
        ## print(np.sum(prob_dist))
        ## print(np.sum(prob_dist *np.arange(0, 35)**2 / 35))
        #plt.bar(np.arange(len(bcount)), prob_dist)
        #plt.title(str(i)+'-simplices')
        ## plt.show()
        #fignum += 1
        # plot cumulative distribution
        #plt.figure(fignum)
        #matrix1 = np.loadtxt('final_OTU.txt', skiprows=0, usecols=range(1, 39))
        #nb_species = matrix1.shape[0]
        #spec_choose_3 = sp.misc.comb(nb_species, i + 1)
        #cumul = cumulative_dist(prob_dist)
        #print(cumul)
        #plt.plot(np.arange(len(cumul)), cumul, label=str(i)+'-simplices')
        #plt.xlabel('Number of appearences')
        #plt.ylabel('Cumulative probability')
        #plt.title(str(i)+'-simplices')
        # plt.show()
        #fignum += 1

        # plot complementary cumulative dist
        #plt.figure(fignum)
        #matrix1 = np.loadtxt('final_OTU.txt', skiprows=0, usecols=range(1, 39))
        #nb_species = matrix1.shape[0]
        #spec_choose_3 = sp.misc.comb(nb_species, i + 1)
        #cumul = cumulative_dist(prob_dist)
        #plt.plot(np.arange(len(cumul)), 1 - cumul)
        #plt.xlabel('Number of appearences')
        #plt.ylabel('1 - Cumulative probability')
        #plt.title(str(i) + '-simplices')
        ## plt.show()
        #fignum += 1
    #plt.legend()
    #plt.show()
    #exit()

    ########## plot count histogram (without 0)
    #fig, ax = plt.subplots()
    #ax.bar(np.arange(len(bcount)), bcount)
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    #plt.xlabel('Number of appearences')
    #plt.ylabel('Number of simplicies')
    #plt.show()

    #exit()



    #fig, ax = plt.subplots()
    #bcount = np.load('bcount.npy')

    ## plot probability distribution
    #matrix1 = np.loadtxt('final_OTU.txt', skiprows=0, usecols=range(1, 39))
    #nb_species = matrix1.shape[0]
    #spec_choose_3 = sp.misc.comb(nb_species, 3)
    #prob_dist = histo_prob_dist(bcount, spec_choose_3)
    #print(np.sum(prob_dist))
    #print(np.sum(prob_dist *np.arange(0, 35)**2 / 35))
    #ax.bar(np.arange(len(bcount)), prob_dist)
    #plt.show()

    ## plot cumulative distribution
    #matrix1 = np.loadtxt('final_OTU.txt', skiprows=0, usecols=range(1, 39))
    #nb_species = matrix1.shape[0]
    #spec_choose_3 = sp.misc.comb(nb_species, 3)
    #prob_dist = histo_prob_dist(bcount, spec_choose_3)
    #cumul = cumulative_dist(prob_dist)
    #print(cumul)
    #plt.plot(np.arange(len(cumul)), cumul)
    #plt.xlabel('Number of appearences')
    #plt.ylabel('Cumulative probability')
    #plt.show()

    ## plot complementary cumulative dist
    #matrix1 = np.loadtxt('final_OTU.txt', skiprows=0, usecols=range(1, 39))
    #nb_species = matrix1.shape[0]
    #spec_choose_3 = sp.misc.comb(nb_species, 3)
    #prob_dist = histo_prob_dist(bcount, spec_choose_3)
    #cumul = cumulative_dist(prob_dist)
    #plt.plot(np.arange(len(cumul)), 1 - cumul)
    #plt.xlabel('Number of appearences')
    #plt.ylabel('1 - Cumulative probability')
    #plt.show()


    # plot count histogram (without 0)
    #fig, ax = plt.subplots()
    #ax.bar(np.arange(len(bcount)), bcount)
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    #plt.xlabel('Number of appearences')
    #plt.ylabel('Number of simplicies')
    #plt.show()


    #exit()


    #exit()
    #start = time.clock()
    #matrix1 = np.loadtxt('SubOtu4000.txt', skiprows=1, usecols=range(1, 35))
    #find_simplices_matrix_dictio(matrix1, 3)
    #with open(r'simplexdictionary.pickle', 'rb') as f:
    #    data = pickle.load(f)
    #    get_keys_for_value(data, 34)
    #print('Time taken : ', time.clock()-start)

    #find_simplices_matrix(matrix1, 3)
    #                      savepath=r'C:\Users\Xavier\Desktop\Notes de cours\Maîtrise\Projet OTU\Betti_scm-master (1)\testspace\\')
    #matrix1 = np.loadtxt('SubOtu4000.txt', skiprows=1, usecols=range(1, 35))
    #print(len(np.where(matrix1[5,:]>0)[0]))
    #print(len(np.where(np.sum(matrix1, axis=1) > 0.1)[0]))
    #exit()
    #count_labeled_simplices(loadpath=r'C:\Users\Xavier\Desktop\Notes de cours\Maîtrise\Projet OTU\Betti_scm-master (1)\testspace\\')
    #print('Time taken : ', time.clock()-start)
    #exit()
    #with open(r'C:\Users\Xavier\Desktop\Notes de cours\Maîtrise\Projet OTU\Betti_scm-master (1)\Betti_scm-master\simplexdictionary.pickle', 'rb') as f:
    #    data = pickle.load(f)
    #    print(len(data))

    #print('wait')

    #matrix1 = np.loadtxt('SubOtu4000.txt', skiprows=1, usecols=range(1, 35))
    #find_simplices_matrix(matrix1, 1, savepath=r'C:\Users\Xavier\Desktop\Notes de cours\Maîtrise\Projet OTU\Betti_scm-master (1)\testspace\\')
    #count_labeled_simplices(loadpath=r'C:\Users\Xavier\Desktop\Notes de cours\Maîtrise\Projet OTU\Betti_scm-master (1)\testspace\\')
    #vector = matrix1[:, 0].flatten()
    #start = time.clock()
    #print(count_n_simplices(vector, 3), time.clock()-start)


    #### plot distributions

    # for i in range(3):
    #    fig, ax = plt.subplots()
    #    bcount = np.load(str(i) + 'bcounts.npy')

    #    # plot probability distribution
    #    matrix1 = np.loadtxt('final_OTU.txt', skiprows=0, usecols=range(1, 39))
    #    nb_species = matrix1.shape[0]
    #    spec_choose_3 = sp.misc.comb(nb_species, i + 1)
    #    prob_dist = histo_prob_dist(bcount, spec_choose_3)
    #    # print(np.sum(prob_dist))
    #    # print(np.sum(prob_dist *np.arange(0, 35)**2 / 35))
    #    ax.bar(np.arange(len(bcount)), prob_dist)
    #    # plt.show()

    #    # plot cumulative distribution
    #    plt.figure(i + 1)
    #    matrix1 = np.loadtxt('final_OTU.txt', skiprows=0, usecols=range(1, 39))
    #    nb_species = matrix1.shape[0]
    #    spec_choose_3 = sp.misc.comb(nb_species, i + 1)
    #    prob_dist = histo_prob_dist(bcount, spec_choose_3)
    #    cumul = cumulative_dist(prob_dist)
    #    print(cumul)
    #    plt.plot(np.arange(len(cumul)), cumul)
    #    plt.xlabel('Number of appearences')
    #    plt.ylabel('Cumulative probability')
    #    # plt.show()

    #    # plot complementary cumulative dist
    #    plt.figure(i + 2)
    #    matrix1 = np.loadtxt('final_OTU.txt', skiprows=0, usecols=range(1, 39))
    #    nb_species = matrix1.shape[0]
    #    spec_choose_3 = sp.misc.comb(nb_species, i + 1)
    #    prob_dist = histo_prob_dist(bcount, spec_choose_3)
    #    cumul = cumulative_dist(prob_dist)
    #    plt.plot(np.arange(len(cumul)), 1 - cumul)
    #    plt.xlabel('Number of appearences')
    #    plt.ylabel('1 - Cumulative probability')
    #    # plt.show()
    # plt.show()
    # exit()

    ## plot count histogram (without 0)
    # fig, ax = plt.subplots()
    # ax.bar(np.arange(len(bcount)), bcount)
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # plt.xlabel('Number of appearences')
    # plt.ylabel('Number of simplicies')
    # plt.show()

    # exit()



    ########## ESPACE DES FIGURES #############

    # 1- ratio (nbr 1-simplexes détectés) / (nbr de 1-simplexes possibles); (pour différents alpha)
    total = sp.special.comb(2611, 2)
    alphalist = np.linspace(0.0001, 0.01, 100)
    one_simplex_count_list = []
    pvalue_list = []

    with open('/home/xavier/Documents/Projet/Betti_scm/pairwise_p_values_vectorized.csv', 'r') as link_file:
        reader = csv.reader(link_file)
        next(reader)
        row_count = 0
        for row in reader:
            if row[-1] != 'nan':
                pvalue_list.append(float(row[-1]))
    for alpha in alphalist:
        print(alpha)
        one_simplex_count_list.append(np.sum(np.array(pvalue_list) < alpha)/total)

    plt.plot(alphalist, one_simplex_count_list)
    plt.xlabel('alpha')
    plt.ylabel('Proportion')
    plt.title('')
    plt.show()











