import numpy as np
import scipy as sp
import scipy.misc
import itertools
import time
import pickle
import json
import os
import matplotlib.pyplot as plt

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

if __name__ == '__main__':

    fig, ax = plt.subplots()
    bcount = np.load('bcount.npy')

    # plot probability distribution
    #matrix1 = np.loadtxt('SubOtu4000.txt', skiprows=1, usecols=range(1, 35))
    #nb_species = matrix1.shape[0]
    #spec_choose_3 = sp.misc.comb(nb_species, 3)
    #prob_dist = histo_prob_dist(bcount, spec_choose_3)
    #ax.bar(np.arange(len(bcount)), prob_dist)
    #plt.show()

    # plot cumulative distribution
    matrix1 = np.loadtxt('SubOtu4000.txt', skiprows=1, usecols=range(1, 35))
    nb_species = matrix1.shape[0]
    spec_choose_3 = sp.misc.comb(nb_species, 3)
    prob_dist = histo_prob_dist(bcount, spec_choose_3)
    cumul = cumulative_dist(prob_dist)
    print(cumul)
    plt.plot(np.arange(len(cumul)), cumul)
    plt.xlabel('Number of appearences')
    plt.ylabel('Cumulative probability')
    plt.show()

    # plot complementary cumulative dist
    matrix1 = np.loadtxt('SubOtu4000.txt', skiprows=1, usecols=range(1, 35))
    nb_species = matrix1.shape[0]
    spec_choose_3 = sp.misc.comb(nb_species, 3)
    prob_dist = histo_prob_dist(bcount, spec_choose_3)
    cumul = cumulative_dist(prob_dist)
    plt.plot(np.arange(len(cumul)), 1 - cumul)
    plt.xlabel('Number of appearences')
    plt.ylabel('1 - Cumulative probability')
    plt.show()


    # plot count histogram (without 0)
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(bcount)), bcount)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlabel('Number of appearences')
    plt.ylabel('Number of simplicies')
    plt.show()


    exit()


    #exit()
    start = time.clock()
    #matrix1 = np.loadtxt('SubOtu4000.txt', skiprows=1, usecols=range(1, 35))
    #find_simplices_matrix_dictio(matrix1, 3)
    with open(r'simplexdictionary.pickle', 'rb') as f:
        data = pickle.load(f)
        get_keys_for_value(data, 34)
    print('Time taken : ', time.clock()-start)

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









