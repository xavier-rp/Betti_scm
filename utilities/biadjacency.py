import numpy as np
import scipy as sp
import scipy.sparse
import networkx as nx
from sklearn.preprocessing import normalize
import networkx.algorithms
"""
Author: Xavier Roy-Pomerleau <xavier.roy-pomerleau.1@ulaval.ca>

See section '' main ''

This module is used to transform biadjacency matrix in .txt to an edge list.

It can also normalize the matrix by column and remove values under a certain threshold in each column

"""

def to_nx_edge_list_format(filename_or_matrix='SubOtu4000.txt', out_name='edgelist_SubOtu4000.txt'):
    """
    This function takes a biadjacency matrix in a txt file OR a numpy matrix/array and writes its edge list in the NetworkX
    format to a file.
    Parameters
    ----------
    filename_or_matrix (str OR numpy.array) : file name from which to open the matrix, or numpy matrix. The edge list
                                              of this matrix is saved in out_name.
    out_name (str) : path and name of the file in which we save the edge list

    -------

    """


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

def normalize_columns(filename_or_matrix='SubOtu4000.txt', savename=None):
    """
    This function normalizes each column of the (biadjacency) matrix. It uses a function from sklearn.preprocessing
    called normalize.

    Parameters
    ----------
    filename_or_matrix (str OR numpy.array) : file name from which to open the matrix, or numpy matrix. The edge list
                                              of this matrix is saved in out_name.
    savename (str) : WIP Name under which to save the normalized matrix.

    Return (Numpy array) : The matrix in which each column are normalized.
    -------

    """
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
    """
    This function removes the lines that only contain zeros from a matrix. This happens when a matrix was filtered using
    one of the few functions of this module. As an example, if the matrix is a biadjacency matrix in which the lines
    represent the species and the columns represent the site in which we find them, and we set to zero the proportion of
    each specie that are lower than 0.05, we might create a line in which every element is zero.

    If this line is not removed, it can cause indexing errors when writting the edgelist from such matrix.

    Parameters
    ----------
    matrix numpy array : Matrix in which we want to remove the lines that only contain zeros.

    Returns numpy array : Matrix in which we removed the lines.
    -------

    """

    i = 0
    # Find the indices of the lines that only contain zeros
    index_list = []
    while i < matrix.shape[0]:
        if np.all(matrix[i,:] == 0):
            index_list.append(i)
        i += 1
    # Delete these lines
    reduced_matrix = np.delete(matrix, index_list, axis=0)
    return reduced_matrix

def matrix_filter(matrix, threshold=0.1, savename=None):
    """
    This function sets the elements of a matrix that are lower than ''threshold'' to zero and normalizes each column of
    the matrix using normalize_columns().


    Parameters
    ----------
    matrix numpy array : Matrix on which to apply filter
    threshold (float) : Threshold under which the elements of the matrix are set to zero.
    savename (str) : WIP TODO

    Returns (numpy array) : The filtered matrix in which we renormalized each column.
    -------

    """
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
    """
    This function sorts the elements of each column by descending order and sums them until "prop_threshold" is reached
    or bursted. It then keeps the '' n '' highest elements that were used to reach the threshold in each column and sets
    every other element to zero. Finally, it normalizes each column of the matrix using normalize_columns().

    In the context of biadjacency matrix in which each column is normalized, the number that we sum are proportion of a
    specie in one site (columne) and sums to 1.

    Parameters
    ----------
    matrix (numpy array) : Matrix on which to apply filter. Its columns must already be normalized
                           (using normalize_columns()).
    prop_threshold (float) : Number between 0 and 1 to which we sum the elements of a column. (0 for nothing, 1 for
                             every elements)
    allow_ties (bool) : If we reach the threshold and that the next element is equal to the last, we also keep it in
                        the column, even though we already reached the threshold. This was considered since there might
                        be two species that are present in the same proportion, but we don't want to pick at random the
                        ones that are going to stay in the matrix.
    savename (str) : WIP TODO

    Returns (numpy array) : The filtered matrix in which we renormalized each column.
    -------

    """


    thresholdlist = []
    i = 0
    # Sort each column in descending order
    while i < len(matrix[0, :]):
        sorted_column = sorted(matrix[:, i], reverse=True)
        total_sum = 0
        j = 0
        # Sum the elements of the column until we burst the threshold
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
        # For each column, we find the elements that are lower than the threshold found in the previous step and set
        # them to zero. If allow_ties is False, we take the '' n '' elements that are equal and that would have made
        # the program burst the threshold in the same way and pick one of them at random.
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
    """
    This function keep the '' n '' highest elements of each column of the matrix. If there are ties, it will also
    include them. All other elements in the column are then set to zero and each column is normalized.

    Parameters
    ----------
    matrix (numpy array) : Matrix on which to apply filter.
    n (int) : Number of highest elements to keep in each column
    savename : WIP TODO

    Returns (numpy array) : The filtered matrix in which we renormalized each column.
    -------

    """

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
    """
    This function sets the elements of a matrix that are lower than ''threshold'' to zero and normalizes each column of
    the matrix using normalize_columns(). The threshold is computed from the mean of each column and the specified
    number "ntimes" of standard deviation : threshold = mean of column - ntimes * standard deviation. Negative floats
    are thus more restrictive (we keeps less and less species since the threshold is high) and positive floats are
    more permissive.

    Parameters
    ----------
    matrix (numpy array) : Matrix on which to apply filter
    ntimes (float) : Number of times we multiply the standard deviation
    savename WIP TODO

    Returns (numpy array) : The filtered matrix in which we renormalized each column.
    -------

    """

    thresholdlist = []
    i = 0
    transpose = matrix.T
    print(np.count_nonzero(matrix))
    # Compute the mean and standard deviation and put the resulting threshold in a list
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
    matrix = matrix_filter(mat, threshold=0.04)
    print(matrix == mat)
    matrix = reduce_matrix(matrix)
    to_nx_edge_list_format(matrix, out_name='edglis')

    print('Done.\n Don\'t forget to change the saving file name so your file does not get written over. ')
