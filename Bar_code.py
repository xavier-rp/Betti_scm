import gudhi
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
import json
from utilities.biadjacency import *
from utilities.bipartite_to_max_facets import to_max_facet
from utilities.prune import to_pruned_file

def compute_bettis_for_persistence(first_index, last_index, path, highest_dim):

    bettilist =[]

    for idx in range(first_index, last_index+1):
        site_facet_list = []
        with open(path + str(idx) + '.txt', 'r') as file:
            print('Computing bettis for : ', path + str(idx) + '.txt')
            for l in file:
                site_facet_list.append([int(x) for x in l.strip().split()])
            st = gudhi.SimplexTree()
            i = 0
            for facet in site_facet_list:
                # This condition and the loop it contains break a facet in all its n choose k faces. This is more
                # memory efficient than inserting a big facet because of the way GUDHI's SimplexTree works. The
                # dimension of the faces has to be at least one unit bigger than the skeleton we use (see next instruction)
                if len(facet) > highest_dim:
                    for face in itertools.combinations(facet, highest_dim):
                        st.insert(face)
                else:
                    st.insert(facet)
                # print(i)
                i += 1

            # Instruction : Change the dimension of the skeleton you want to use. From a skeleton of dimension d, we can
            # only compute Betti numbers lower than d. The dimension of the skeleton has to be coherent with the
            # dimension of the largest facet allowed, meaning that if we break a facet into its N choose K faces, where
            # K == 3, than the dimension of the skeleton has to be d <= K-1
            skel = st.get_skeleton(highest_dim-1)
            sk = gudhi.SimplexTree()
            for tupl in skel:
                sk.insert(tupl[0])

            # This function has to be launched in order to compute the Betti numbers
            sk.persistence()
            bettilist.append(sk.betti_numbers())
    length_longest_sublist = max(len(l) for l in bettilist)
    for sublist in bettilist:
        if len(sublist) < length_longest_sublist:
            sublist.extend(0 for i in range(length_longest_sublist - len(sublist)))

    return np.array(bettilist)

def plot_betti_persistence(bettiarray, threshold_array):
    for betticolumn_index in range(0, len(bettiarray[0])):
        bar_heights = []
        bar_starting_pos = []
        number_of_betti = []
        previous_change_index = 0
        i = 1
        while i < len(bettiarray[:, betticolumn_index]):
            if bettiarray[i, betticolumn_index] != bettiarray[i-1, betticolumn_index]:
                number_of_betti.append(bettiarray[i-1, betticolumn_index])
                bar_starting_pos.append(threshold_array[previous_change_index])
                bar_heights.append(threshold_array[i]-threshold_array[previous_change_index])
                previous_change_index = i
            i += 1
        i -= 1
        number_of_betti.append(bettiarray[i, betticolumn_index])
        bar_starting_pos.append(threshold_array[previous_change_index])
        bar_heights.append(threshold_array[i] - threshold_array[previous_change_index])

        #plt.figure(betticolumn_index)
        plt.rcdefaults()
        fig, ax = plt.subplots()

        # Example data
        y_pos = number_of_betti
        left = bar_starting_pos
        ax.barh(y_pos, bar_heights, align='center',
                color='green', left=left)
        ax.set_yticks(y_pos)
        ax.invert_xaxis()
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Number of Betti ' + str(betticolumn_index) + ' in the simplicial complex')
        ax.set_title('Persistence of the number of Betti ' + str(betticolumn_index))

    plt.show()

if __name__ == '__main__':

    thresholdlist = np.linspace(0.01, 0.2, 10000)
    i = 1
    path = '/home/xavier/Documents/Projet/Betti_scm/persistencetest/simpletest'
    #for thresh in thresholdlist:
    #    matrix1 = np.loadtxt('final_OTU.txt', skiprows=0, usecols=range(1, 39))
    #    mat = normalize_columns(matrix1)
    #    matrix = matrix_filter(mat, threshold=thresh)
    #    print(matrix == mat)
    #    matrix = reduce_matrix(matrix)
    #    to_nx_edge_list_format(matrix, out_name=path+str(i)+'.txt')
    #    i += 1
    ilist = np.arange(1, 10001)

    #for i in ilist:
    #    to_max_facet(path+str(i)+'.txt', 1, path+'facet'+str(i)+'.txt')
    #    to_pruned_file(path+'facet'+str(i)+'.txt', path+'facet'+'pruned'+str(i)+'.txt')

    bettiarr = compute_bettis_for_persistence(ilist[0], ilist[-1], path+'facetpruned', 4)
    print(bettiarr.shape)

    plot_betti_persistence(bettiarr, thresholdlist)









