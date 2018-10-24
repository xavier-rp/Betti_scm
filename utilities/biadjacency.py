import numpy as np
import scipy as sp
import scipy.sparse
import networkx as nx
from sklearn.preprocessing import normalize
import networkx.algorithms

def to_nx_edge_list_format(filename_or_matrix='SubOtu4000.txt', out_name='edgelist_SubOtu4000.txt'):
    #Takes a biadjacency matrix in a txt file OR a numpy matrix and writes it in the konect format (edgelist)

    if isinstance(filename_or_matrix, str):
        ### Instruction : skiprows and usecols have to be updated depending on the file you use
        # Skiprows and usecols have to be adjusted depending on the file used.
        # For final_OUT, use skiprows=0 and usecols=range(1,39)
        # For SubOtu4000 use skiprows=1 and usecols=range(1,35)
        mat = np.loadtxt(filename_or_matrix, skiprows=1, usecols=range(1, 35))
    else:
        #This means that filename_or_matrix is a numpy matrix
        mat = filename_or_matrix

    #Necessary to be able to use the networkX function
    A = sp.sparse.csr_matrix(mat)
    print(A)
    G = nx.algorithms.bipartite.from_biadjacency_matrix(A)
    nx.write_edgelist(G, out_name, data=False)


def read_edge_list_konect_minus(path):
    """Read edge list, but the nodes have to be zero indexed manually by specifying  -x, where x is an integer,
    in the line edge_list.add((int(e[0]), int(e[1])-500)). In this example, the nodes of the second set had to be
    reindexed using -500, because the smallest index for the second set was 500. """
    with open(path, 'r') as f:
        edge_list = set()
        for line in f:
            if not line.lstrip().startswith("%"):  # ignore commets
                e = line.strip().split()
                edge_list.add((int(e[0]), int(e[1]) - 47))  # 0 index
        return list(edge_list)

def normalize_columns(filename_or_matrix='SubOtu4000.txt', savename=None):
    #Normalizes each column of a biadjacency matrix
    if savename is None:
        if isinstance(filename_or_matrix, str):
            matrix = np.loadtxt(filename_or_matrix, skiprows=1, usecols=range(1, 35))
        else:
            matrix = filename_or_matrix
        normed_matrix = normalize(matrix, axis=0, norm='l1')
        return normed_matrix

    else:#TODO SAVE TO TXT
        matrix = np.loadtxt(filename_or_matrix, skiprows=1, usecols=range(1, 35))
        normed_matrix = normalize(matrix, axis=0, norm='l1')

def reduce_matrix(matrix):
    # Removes lines that only contain zeros. Needed to make sure that everything gets properly zero-indexed.
    i = 0
    index_list = []
    while i < matrix.shape[0]:
        if np.all(matrix[i,:] == 0):
            index_list.append(i)
        i += 1
    reduced_matrix = np.delete(matrix, index_list, axis=0)
    return reduced_matrix

def matrix_filter(matrix, threshold=0.1, savename=None):
    # Sets the elements of the matrix to zero if it is lower than the threshold
    # and (re)normalizes each column of the matrix

    if savename is None:
        low_values_flags = matrix < threshold
        matrix[low_values_flags] = 0

        matrix = normalize_columns(matrix)
        return matrix
    else: # TODO SAVE TO TXT
        pass

def matrix_filter_sum_to_prop(matrix, prop_threshold=0.5, allow_ties=True, savename=None):
    # Sorts the elements of each column from highest to lowest proportion sums element by element until a propotion
    # threshold is reached or bursted.

    # Keeps the n highest elements of each column of the matrix. If there are ties, it will also include them. It then
    # (re)normalizes each column the matrix

    thresholdlist = []
    i = 0

    while i < len(matrix[0, :]):
        sorted_column = sorted(matrix[:, i], reverse=True)
        total_sum = 0
        j = 0
        while total_sum < prop_threshold:
            total_sum += sorted_column[j]
            j += 1

        # We add the last element added to a threshold list that we are going to use to set every elements lower than
        # this value in the column to zero
        thresholdlist.append(sorted_column[j-1])

        i += 1
    nb_elements_in_column = len(matrix[:, i-1])
    if savename is None:
        i = 0
        while i < len(thresholdlist):
            print(thresholdlist[i])
            if len(np.where(matrix[:, i] == thresholdlist[i])) == 1:
                low_values_flags = matrix[:, i] < thresholdlist[i]
                matrix[low_values_flags, i] = 0
            elif len(np.where(matrix[:, i] == thresholdlist[i])) > 1 and allow_ties is True:
                low_values_flags = matrix[:, i] < thresholdlist[i]
                matrix[low_values_flags, i] = 0
            else:
                low_values_flags = matrix[:, i] < thresholdlist[i]
                matrix[low_values_flags, i] = 0

                idx = np.where(matrix[:, i] == thresholdlist[i])
                np.random.shuffle(idx)
                k = 0
                while k < len(idx) - 1:
                    matrix[idx, i] = 0
                    k += 1

            print(nb_elements_in_column - np.count_nonzero(low_values_flags))
            i += 1
        print(np.count_nonzero(matrix))
        matrix = normalize_columns(matrix)
        print(np.count_nonzero(matrix))
        return matrix


def matrix_filter_keep_n(matrix, n=1, savename=None):
    # Keeps the n highest elements of each column of the matrix. If there are ties, it will also include them. It then
    # (re)normalizes each column the matrix

    thresholdlist = []
    i = 0
    transpose = matrix.T
    print(np.count_nonzero(matrix))
    while i < len(transpose[:,0]):
        thresholdlist.append(sorted(transpose[i,:])[-n])
        i += 1
    nb_elements_in_column = len(matrix[:, i-1])
    if savename is None:
        i = 0
        while i < len(thresholdlist):
            print(thresholdlist[i])
            low_values_flags = matrix[:, i] < thresholdlist[i]
            matrix[low_values_flags, i] = 0
            print(nb_elements_in_column-np.count_nonzero(low_values_flags))
            i += 1
        print(np.count_nonzero(matrix))
        matrix = normalize_columns(matrix)
        print(np.count_nonzero(matrix))
        return matrix
    else: # TODO SAVE TO TXT
        pass


def matrix_adaptive_filter(matrix, ntimes=0, savename=None):
    # Sets the elements of the matrix to zero if it is lower than the threshold
    # and (re)normalizes each column the matrix
    # The adaptive filter plays around the mean. If ntime == 0, all species in a site that are present in a proportion
    # lower than the mean are set to zero. If ntimes == negative integer, we do the same, but for species that are less
    # present than mean - -ntimes*sigma (more restritive), where sigma is the standard variation. if ntimes == positive integer, we cut
    # those under mean - ntimes*sigma (less restrictive)
    thresholdlist = []
    i = 0
    transpose = matrix.T
    print(np.count_nonzero(matrix))
    while i < len(transpose[:,0]):

        mean = np.mean(transpose[i,:])
        std = np.std(transpose[i, :])
        thresholdlist.append(mean - ntimes*std)
        i += 1
    if savename is None:
        i = 0
        while i < len(thresholdlist):
            print(thresholdlist[i])
            low_values_flags = matrix[:, i] < thresholdlist[i]
            matrix[low_values_flags, i] = 0
            i += 1
        print(np.count_nonzero(matrix))
        matrix = normalize_columns(matrix)
        print(np.count_nonzero(matrix))
        return matrix
    else: # TODO SAVE TO TXT
        pass

def facet_list_to_graph(facet_list):
    """Convert a facet list to a bipartite graph"""
    # ON peut convertir le graphe bipartite en autre graphe en effectuant une multiplication par la transposée
    g = nx.Graph()
    for f, facet in enumerate(facet_list):
        for v in facet:
            g.add_edge('v' + str(v), 'f' + str(f))  # differentiate node types
    return g

def facet_list_to_bipartite(facet_list):
    """Convert a facet list to a bipartite graph"""
    # ON peut convertir le graphe bipartite en autre graphe en effectuant une multiplication par la transposée
    g = nx.Graph()
    for f, facet in enumerate(facet_list):
        g.add_node('f'+str(f), bipartite = 1)
        for v in facet:
            g.add_node('v'+ str(v), bipartite = 0)
            g.add_edge('v' + str(v), 'f' + str(f))  # differentiate node types
    return g

def facet_list_to_bipartite2(facet_list):
    """Convert a facet list to a bipartite graph"""
    # ON peut convertir le graphe bipartite en autre graphe en effectuant une multiplication par la transposée
    g = nx.Graph()
    usedlist = []
    tuplelist = []
    i=500
    for f, facet in enumerate(facet_list):
        print(f, facet)
        g.add_node(f, bipartite = 1)
        for v in facet:
            if v not in usedlist:
                usedlist.append(v)
                tuplelist.append((i,v))
                v = i
                i += 1
            else:
                idx = usedlist.index(v)
                v = tuplelist[idx][0]
            g.add_node(v, bipartite = 0)
            g.add_edge(v, f)  # differentiate node types
    return g

if __name__ == '__main__':
    """
    This module is used to transform biadjacency matrix in .txt to an edge list.

    It can also normalize the matrix by column and remove values under a certain threshold in each column
    """



    matrix1 = np.loadtxt('final_OTU.txt', skiprows=0, usecols=range(1,39))
    mat = normalize_columns(matrix1)
    i = 0
    array = []
    while i < mat.shape[1]:
        array.append(max(mat[:,i]))
        print(min(mat[:,i]), max(mat[:,i]), np.sum(mat[:,i]))
        i += 1
    mean = sum(array)/len(array)
    print(mean)
    print(np.count_nonzero(mat))
    #matrix = matrix_adaptive_filter(mat, ntimes=-2)
    #matrix = matrix_adaptive_filter(mat, ntimes=-10)
    #matrix = matrix_filter_keep_n(mat, n=20)
    #matrix = matrix_filter_sum_to_prop(mat, prop_threshold=0.2)
    matrix = matrix_filter(mat, threshold=0.04)
    print(matrix == mat)
    matrix = reduce_matrix(matrix)

    print(to_nx_edge_list_format(matrix, out_name='edglis'))
    #print('stop')

    #polli = []
    #with open("/home/xavier/Documents/Projet/scm/datasets/pollinators_facet_list.txt", 'r') as f:
    #    for l in f:
    #        polli.append([int(x) for x in l.strip().split()])
    #G = facet_list_to_bipartite2(polli)
    #nx.write_edgelist(G, 'polli_edge.txt', data=False)

    print('Done.\n Don\'t forget to change the saving file name so your file does not get written over. ')


    #to_konect_format(matrix, out_name='severeadaptivthresh_finalOTU.txt')