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


def chisq_test(cont_tab, expected):
    #Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
    #via MLE or iterative proportional fitting.

    if float(0) in expected:
        test_stat = 0
        p_val = 1
    else:
        df = 1
        test_stat = np.sum((cont_tab-expected)**2/expected)
        p_val = chi2.sf(test_stat, df)

    return test_stat, p_val


def save_pairwise_p_values_phi_dictionary(bipartite_matrix, dictionary, savename):

    # create a CSV file
    with open(savename+'.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows([['node index 1', 'node index 2', 'p-value', 'phi-coefficient']])

        buffer = []
        count = 0
        for one_simplex in tqdm(itertools.combinations(range(bipartite_matrix.shape[0]), 2)):
            contingency_table = get_cont_table(one_simplex[0], one_simplex[1], bipartite_matrix).astype(int)
            table_str = str(contingency_table[0, 0]) + '_' + str(contingency_table[0, 1]) + '_' + \
                        str( contingency_table[1, 0]) + '_' + str(contingency_table[1, 1])

            phi = phi_coefficient_table(contingency_table)

            chi2, p = dictionary[table_str]
            buffer.append([one_simplex[0], one_simplex[1], p, phi])
            count += 1
            writer = csv.writer(csvFile)
            writer.writerows(buffer)
            count = 0
            # empty the buffer
            buffer = []

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
                    graph.add_edge(int(row[0]), int(row[1]), p_value=row[2], phi=float(row[-1]))

    return graph

def save_triplets_p_values(bipartite_matrix, nodelist, savename, bufferlimit=100000):

    # create a CSV file
    with open(savename+'.csv', 'w',  newline='') as csvFile:
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
            with open(savename+'.csv', 'a',  newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerows(buffer)
                count = 0
                # empty the buffer
                buffer = []

    with open(savename + '.csv', 'a',  newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(buffer)

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

def save_all_n_cliques(G, n, savename, bufferlimit=100000):
    G = copy.deepcopy(G)
    with open(savename + '.csv', 'w',  newline='') as csvFile:
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
                        with open(savename + '.csv', 'a',  newline='') as csvFile:
                            writer = csv.writer(csvFile)
                            writer.writerows(buffer)
                            count = 0
                            # empty the buffer
                            buffer = []


            G.remove_node(node)

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

def extract_2_simplex_from_csv(csvfilename, alpha, savename):

    with open(csvfilename, 'r') as csvfile, open(savename + '.csv', 'w',  newline='') as fout:
        reader = csv.reader(csvfile)
        writer = csv.writer(fout)
        writer.writerows([['node index 1', 'node index 2', 'node index 3', 'phi 1-2', 'phi 1-3', 'phi 2-3', 'p-value']])
        next(reader)
        for row in tqdm(reader):
            try :
                p = float(row[6])
                if p < alpha:
                    writer = csv.writer(fout)
                    writer.writerow([int(row[0]), int(row[1]), int(row[2]), float(row[3]), float(row[4]), float(row[5]), p])
            except:
                pass

            #if row[-1] != 'nan' and row[-1] != 'None':
            #    p = float(row[-1])
            #    if p < alpha:
            #        writer = csv.writer(fout)
            #        writer.writerow([int(row[0]), int(row[1]), int(row[2]), float(row[3]), float(row[4]), float(row[5]), p])

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

def extract_phi_for_triangles(csvfilename):
    with open(csvfilename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        pure_negative_count = 0
        pure_positive_count = 0
        one_pos_two_neg = 0
        two_pos_one_neg = 0

        for row in reader:
            try:
                philist = [float(row[3]), float(row[4]), float(row[5])]
                philistmask = (np.array(philist) > 0) * 1
                sum = np.sum(philistmask)
                if sum == 3:
                    pure_positive_count += 1
                elif sum == 0:
                    pure_negative_count += 1
                elif sum == 1:
                    one_pos_two_neg += 1
                else:
                    two_pos_one_neg += 1
            except:
                pass

    return [pure_negative_count, one_pos_two_neg, two_pos_one_neg, pure_positive_count]


class Analyser():

    def __init__(self, biadjacencymatrix, savename, exact=False):

        self.bam = biadjacencymatrix
        self.savename= savename
        self.exact = exact

    def analyse_asymptotic(self, alpha):

        self.save_pairwise_p_values_phi(self.savename)

        g = read_pairwise_p_values(self.savename + '.csv', alpha)

        nx.write_edgelist(g, self.savename + '_asymptotic_edgelist_' + str(alpha)[2:] + '.txt')

        print('Number of nodes : ', g.number_of_nodes())
        print('Number of links : ', g.number_of_edges())
        print("Network density : ", nx.density(g))
        print("Is connected : ", nx.is_connected(g))


    def analyse_exact(self, alpha, samples=1000000, chi1=True):

        self.get_unique_tables()

        self.find_exact_pvalues_for_tables(samples, chi1)

        save_pairwise_p_values_phi_dictionary(self.bam, self.pval_dictio, self.savename)

        g = read_pairwise_p_values(self.savename + '.csv', alpha)

        nx.write_edgelist(g, self.savename + '_exact_edgelist_' + str(alpha)[2:] + '.txt')
        print('Number of nodes : ', g.number_of_nodes())
        print('Number of links : ', g.number_of_edges())
        print("Network density : ", nx.density(g))
        print("Is connected : ", nx.is_connected(g))

    def get_unique_tables(self):
        table_set = set()

        ############ Count number of different tables :
        for one_simplex in itertools.combinations(range(self.bam.shape[0]), 2):
            computed_cont_table = get_cont_table(one_simplex[0], one_simplex[1], self.bam)
            computed_cont_table = computed_cont_table.astype(int)
            table_str = str(computed_cont_table[0, 0]) + '_' + str(computed_cont_table[0, 1]) + '_' + str(
                computed_cont_table[1, 0]) + '_' + str(computed_cont_table[1, 1])
            if table_str not in table_set:
                table_set.add(table_str)
        table_set = list(table_set)
        print('How many different tables : ', len(table_set))

        self.table_set = table_set

        #json.dump(table_set, open("table_list.json", 'w'))

    def find_exact_pvalues_for_tables(self, nb_samples=1000000, chi1=True):

        #### From the different tables : generate the chisqdist :

        pvaldictio = {}

        # Max index used in range() :
        for it in range(len(self.table_set)):
            table_id = self.table_set[it]
            table = np.random.rand(2, 2)
            table_id_list = str.split(table_id, '_')
            table[0, 0] = int(table_id_list[0])
            table[0, 1] = int(table_id_list[1])
            table[1, 0] = int(table_id_list[2])
            table[1, 1] = int(table_id_list[3])

            N = np.sum(table)
            expected1 = mle_2x2_ind(table)
            problist = mle_multinomial_from_table(expected1)
            start = time.clock()
            sample = multinomial_problist_cont_table(N, problist, nb_samples)
            if chi1:
                expec = mle_2x2_ind_vector(sample, N)
                chisq = chisq_formula_vector(sample, expec)
            else:
                expec = np.tile(expected1, (nb_samples, 1)).reshape(nb_samples, 2, 2)
                chisq = chisq_formula_vector(sample, expec)

            pvaldictio[table_id] = sampled_chisq_test(table, expected1, chisq)
            print('Time for one table: ', time.clock() - start)

        self.pval_dictio = pvaldictio

        #json.dump(pvaldictio, open(dictionaryname, 'w'))


    def save_pairwise_p_values_phi(self, savename):

        # create a CSV file
        with open(savename + '.csv', 'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows([['node index 1', 'node index 2', 'p-value', 'phi-coefficient']])

            count = 0
            for one_simplex in tqdm(itertools.combinations(range(self.bam.shape[0]), 2)):
                contingency_table = get_cont_table(one_simplex[0], one_simplex[1], self.bam)
                expected_table = mle_2x2_ind(contingency_table)
                phi = phi_coefficient_table(contingency_table)
                chi2, p = chisq_test(contingency_table, expected_table)
                count += 1
                writer = csv.writer(csvFile)
                writer.writerow([one_simplex[0], one_simplex[1], p, phi])

    def compare_graph(self, alpha, path_to_factorgraph):

        with open(path_to_factorgraph, 'rb') as fg_file:
            factorgraph = pickle.load(fg_file)


        g = read_pairwise_p_values(self.savename + '.csv', alpha)

        count_fp = 0
        print('The original factor graph does not have these statistical edges :')
        for edge in g.edges():
            if not factorgraph.has_edge(edge[0], edge[1]):

                with open(self.savename + '.csv', 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader)
                    for row in reader:
                        if (int(row[0]) == edge[0] and int(row[1]) == edge[1]) or (int(row[0]) == edge[1] and int(row[1]) == edge[0]):
                            p_val = float(row[-2])
                            break


                print(edge, ' p_value : ', p_val)
                count_fp += 1
        print('Total of false positives : ', count_fp)

        print('The statistical graph does not have these original edges :')
        count_fn = 0
        for edge in factorgraph.edges():
            if not g.has_edge(edge[0], edge[1]):

                with open(self.savename + '.csv', 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader)
                    for row in reader:
                        if (int(row[0]) == edge[0] and int(row[1]) == edge[1]) or (int(row[0]) == edge[1] and int(row[1]) == edge[0]):
                            p_val = float(row[-2])
                            break


                print(edge, ' p_value : ', p_val)
                count_fn += 1
        print('Total of false negatives : ', count_fn)

        count_tp = 0
        print('The original factor graph share these statistical edges :')
        for edge in g.edges():
            if factorgraph.has_edge(edge[0], edge[1]):

                with open(self.savename + '.csv', 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader)
                    for row in reader:
                        if (int(row[0]) == edge[0] and int(row[1]) == edge[1]) or (int(row[0]) == edge[1] and int(row[1]) == edge[0]):
                            p_val = float(row[-2])
                            break


                print(edge, ' p_value : ', p_val)
                count_tp += 1
        print('Total of True positives : ', count_tp, ' out of ', len(factorgraph.edges))


        tpr = count_tp / (count_tp + count_fn)

        count_tn = sp.misc.comb(len(factorgraph.nodes), 2) - count_tp - count_fp - count_fn

        fpr = count_fp / (count_fp + count_tn)

        print(count_tn)

        print(fpr, tpr, tpr-fpr)

        print(alpha)

        return [fpr, tpr]







if __name__ == '__main__':


    bam = np.load('bipartite.npy')

    print(bam[1])

    analyser = Analyser(bam, 'test_analyser_exact')

    #analyser.analyse_exact(0.01)

    x = []
    y = []
    for alpha in np.logspace(-10, 0):
        output = analyser.compare_graph(alpha, 'factorgraph.pkl')
        x.append(output[0])
        y.append(output[1])
    print(x)
    plt.plot(x, y)
    plt.plot([0,1], [0,1])
    plt.show()
    exit()

    # TODO : ADD NEW SAVE PAIRWISEPVALUE DICTIONARY,
    # TODO SEND DICTIONARY
    ####################### Exact p-values
    #with open('exact_chisq_1deg_cc/complete_pval_dictionary_1deg.json') as jsonfile:
    #    dictio = json.load(jsonfile)

    #    matrix1 = np.loadtxt('final_OTU.txt', skiprows=0, usecols=range(1, 39))

    #    matrix1 = to_occurrence_matrix(matrix1, savepath=None)

    #    save_pairwise_p_values_phi_dictionary(matrix1, dictio, 'exact_chi1_pvalues_final_otu')
    #exit()


    #find_tetra_from_all_comb(matrix1)

    #exit()

    #find_4_cliques(r'extracted_2-simplices_001_final_otu.csv', '4cliquecsv')
    #exit()


    ######## First step : extract all links, their p-value and their phi-coefficient
    #### Load matrix
    #matrix1 = np.loadtxt('final_OTU.txt', skiprows=0, usecols=range(1, 39))

    ##### Transform into occurrence matrix
    #matrix1 = to_occurrence_matrix(matrix1, savepath=None)

    #print(get_cont_table(1, 2419, matrix1))
    #print(get_cont_table(2, 19, matrix1))
    #print(get_cont_table(2, 129, matrix1))
    #print(get_cont_table(3,2059, matrix1))
    #exit()
    #### Save all links and values to CSV file
    #save_pairwise_p_values_phi(matrix1, 'windows_final_otu_pairwise_pvalue_phi')
    #exit()

    ######## Second step : Choose alpha and extract the network
    #G = read_pairwise_p_values('windows_final_otu_pairwise_pvalue_phi.csv', 0.001)
    #g = read_pairwise_p_values('final_otu_exact_pairwise_pvalues_phi.csv', 0.001)
    #g = read_pairwise_p_values('exact_1-simplices_1deg.csv', 0.001)
    #g = read_pairwise_p_values('exact_chi1_pvalues_final_otu.csv', 0.001)
    #count = 0
    #for edge in g.edges():
    #    if not G.has_edge(edge[0], edge[1]):
    #        print(edge)
    #        count += 1
    #print(count)
    ##exit()
    #print('Number of nodes : ', g.number_of_nodes())
    #print('Number of links : ', g.number_of_edges())
    ###### Compute a few interesting quantities
    #print("Network density : ", nx.density(g))
    #print("Is connected : ", nx.is_connected(g))
    ##print("Triadic closure : ", nx.transitivity(g))
    #degree_sequence = sorted([d for n, d in g.degree()], reverse=True)  # degree sequence
    #print("Average degree : ", np.sum(np.array(degree_sequence))/len(degree_sequence))
    #degreeCount = collections.Counter(degree_sequence)
    #deg, cnt = zip(*degreeCount.items())

    #fig, ax = plt.subplots()
    ##plt.bar(deg, cnt/np.sum(cnt), width=10, color='b')
    #n, bins, patches = plt.hist(degree_sequence, np.arange(0, 600, 10), density=True, facecolor='g', alpha=0.75)
    #plt.title("Degree Histogram")
    #plt.ylabel("Count")
    #plt.xlabel("Degree")
    ##ax.set_xticks([d + 0.4 for d in deg])
    ##ax.set_xticklabels(deg)
    #plt.show()

    #exit()

    # Bunch of stats can be found here : https://networkx.github.io/documentation/stable/reference/functions.html

    #### to json for d3js :

    #g = read_pairwise_p_values('exact_chi1_pvalues_final_otu.csv', 0.001)
    #label_list = []
    #with open('groupes_otu.csv', 'r') as csvfile:
    #    reader = csv.reader(csvfile)
    #    for row in reader:
    #        try:
    #            label_list.append(row[1])
    #        except:
    #            if row[0] != 'Bacteria':
    #                label_list.append(row[0])
    #            else:
    #                label_list.append(label_list[-1])

    #node_dictio_list = []
    #for noeud in g.nodes:
    #    node_dictio_list.append({"id": str(noeud), "group": label_list[noeud]})
    #    #node_dictio_list.append({"id":str(noeud)})

    #link_dictio_list = []
    #for lien in g.edges:
    #    link_dictio_list.append({"source": str(lien[0]), "value": 1, "target": str(lien[1])})

    #json_diction = {"nodes": node_dictio_list, "links" : link_dictio_list}
    #with open('d3js_exact_chi1_final_otu_with_groups.json', 'w') as outfile:
    #    json.dump(json_diction, outfile)
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
    #node_dictio_list = []
    #label_list = []
    #with open('groupes_otu.csv', 'r') as csvfile:
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
    #link_dictio_list = []
    #node_set = set()
    #count = 0
    #for lien in g.edges:
    #    if g.get_edge_data(lien[0], lien[1])['phi'] < 0 :
    #        count += 1
    #        #link_dictio_list.append({"source": str(lien[0]), "value": g.get_edge_data(lien[0], lien[1])['phi'], "target": str(lien[1])})
    #        link_dictio_list.append({"source": str(lien[0]), "value": 1, "target": str(lien[1])})
    #        node_set.add(lien[0])
    #        node_set.add(lien[1])
    #print('NUmber of negative interactions : ', count)

    #node_dictio_list = []
    #for noeud in list(node_set):
    #    node_dictio_list.append({"id": str(noeud), "group": label_list[noeud]})
    #    #node_dictio_list.append({"id": str(noeud)})


    #json_diction = {"nodes": node_dictio_list, "links": link_dictio_list}
    #with open('d3js_exact_chi1_negative_interactions_final_otu_with_groups.json', 'w') as outfile:
    #    json.dump(json_diction, outfile)

    #exit()
    #link_dictio_list = []
    #node_set = set()
    #for lien in g.edges:
    #    if g.get_edge_data(lien[0], lien[1])['phi'] > 0:
    #        #link_dictio_list.append(
    #        #    {"source": str(lien[0]), "value": g.get_edge_data(lien[0], lien[1])['phi'], "target": str(lien[1])})
    #        link_dictio_list.append(
    #            {"source": str(lien[0]), "value": 1, "target": str(lien[1])})
    #        node_set.add(lien[0])
    #        node_set.add(lien[1])

    #node_dictio_list = []
    #for noeud in list(node_set):
    #    node_dictio_list.append({"id": str(noeud), "group": label_list[noeud]})
    #    #node_dictio_list.append({"id": str(noeud)})

    #json_diction = {"nodes": node_dictio_list, "links": link_dictio_list}
    #with open('positive_interactions_otu_grouped.json', 'w') as outfile:
    #    json.dump(json_diction, outfile)

    #exit()


    ######## Third step : Find all triangles in the previous network

    #g = read_pairwise_p_values('exact_chi1_pvalues_final_otu.csv', 0.001)

    #save_all_triangles(g, 'exact_chi1_triangles_001_final_otu')

    #print(count_triangles_csv('exact_chi1_triangles_001_final_otu.csv'))

    #exit()

    ######## Fourth step : Find all the p-values for the triangles under the hypothesis of homogeneity

    #### Matrix is needed for the analysis
    matrix1 = np.loadtxt('final_OTU.txt', skiprows=0, usecols=range(1, 39))

    #### Transform into occurrence matrix
    matrix1 = to_occurrence_matrix(matrix1, savepath=None)

    triangles_p_values_AB_AC_BC('exact_chi1_triangles_001_final_otu.csv', 'exact_triangles_001_final_otu_pvalues.csv', matrix1)

    ######## Fifth step : Exctract all 2-simplices

    #extract_2_simplex_from_csv('windows_triangles_001_final_otu_pvalues.csv', 0.001, 'extracted_2-simplices_001_final_otu')

    # THIS ONE GIVES ALL TRIANGLES THAT CONVERGED REGARDLESS OF THEIR P-VALUE (NO ALPHA NEEDED)
    #extract_converged_triangles('windows_triangles_001_final_otu_pvalues.csv', 'converged_triangles')

    #exit()

    ######### Sixth step : Rebuild a network from triangles only and find all the 4-cliques

    #g = triangles_to_network(r'extracted_2-simplices_001_final_otu.csv')

    #find_4_cliques(g, '4clique_final_otu')

    #exit()

    ######## Seventh step : Find all the p-values for the tetrahedrons under the hypothesis of homogeneity

    ##### Matrix is needed for the analysis
    #matrix1 = np.loadtxt('final_OTU.txt', skiprows=0, usecols=range(1, 39))

    ##### Transform into occurrence matrix
    #matrix1 = to_occurrence_matrix(matrix1, savepath=None)

    #print(tetrahedron_p_values_ABC_ABD_ACD_BCD('4clique_final_otu.csv', '4clique_p_values_final_otu', matrix1))

    #exit()

    ######## Eigth step : Extract all 3-simplices

    #extract_3_simplex_from_csv('4clique_p_values_final_otu.csv', 0.001, 'extracted_3-simplices_001_final_otu')

    #exit()
    ################# DONE ###################
