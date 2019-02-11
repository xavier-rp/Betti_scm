import numpy as np
import itertools
import time

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

def group_is_present(vector, indices):
    """
    Verify if the entries at matrix[indices[0], indices[1]] are simultaneously non zero.
    Parameters
    ----------
    matrix (np.array) : Co-occurrence matrix.
    indices (array of int) : (indices of lines)

    Returns
    -------
    1 if the entries are simulteaneously non zero
    0 otherwise
    """

    line_indices = indices
    presence_array = (vector > 0)*1
    if np.sum(presence_array) == len(line_indices):
        return 1
    else:
        return 0

def count_n_simplices_blind(vector, n):


    # Indices of the non zero vector entries. We use this because it would be a waste of time to check between every
    # group of n entries since we know that we wont count groups with entries that are zero.
    all_line_indices = np.where(vector > 0)[0]

    # Thus, all the len(all_line_indices) choose n groups that we can build have non zero entries in the vector. With
    # itertools, we can get the indices of the nodes in each group and save them
    tuple_list = []
    count = 0
    for simplex in itertools.combinations(all_line_indices, n):
        #tuple_list.append(simplex)
        count += 1
    return count, tuple_list

if __name__ == '__main__':
    matrix1 = np.loadtxt('/home/xavier/Documents/Projet/Betti_scm/utilities/SubOtu4000.txt', skiprows=1, usecols=range(1, 35))
    vector = matrix1[:, 0].flatten()
    start = time.clock()
    print(count_n_simplices_blind(vector, 5), time.clock()-start)









