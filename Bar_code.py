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
    """
    This function returns an array in which the columns are the Betti numbers and the lines are the filtered facet list
    of a bipartite graph.

    Parameters
    ----------
    first_index (int) : Index of the first instance from which we want to compute the Betti numbers. Instances between
                        this index and last_index will be treated as well.
    last_index (int) : Index of the last instance from which we want to compute the Betti numbers. Instances between
                       this index and first_index will be treated as well.
    path (str) : path of the files we want to analyse. Do not put the extension (.txt will be added by default) and do
                 not specify the index (automatically added in a loop using range(first_index, last_index+1).
    highest_dim (int) : Dimension of the largest facet allowed / Dimension of the j-skeleton used to compute the Betti
                        numbers (see GUDHI's documentation on skeleton). If a facet of dimension higher than highest_dim
                        exists in the facet list, it will be decomposed in its N choose highest_dim facets. This number
                        limits which Betti numbers can be computed (for instance if highest_dim == 2, only Betti 0 and 1
                        can be computed).

    Returns (numpy array) : an array in which the columns are the Betti numbers and the lines are the filtered facet
                            lists of a bipartite graph.
    -------

    """

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
                if len(facet) > highest_dim+1:
                    for face in itertools.combinations(facet, highest_dim+1):
                        st.insert(face)
                else:
                    st.insert(facet)
                # print(i)
                i += 1

            # Instruction : Change the dimension of the skeleton you want to use. From a skeleton of dimension d, we can
            # only compute Betti numbers lower than d. The dimension of the skeleton has to be coherent with the
            # dimension of the largest facet allowed, meaning that if we break a facet into its N choose K faces, where
            # K == 3, than the dimension of the skeleton has to be d <= K-1
            skel = st.get_skeleton(highest_dim)
            sk = gudhi.SimplexTree()
            for tupl in skel:
                sk.insert(tupl[0])

            # This function has to be launched in order to compute the Betti numbers
            sk.persistence()
            bettis = sk.betti_numbers()
            if len(bettis) < highest_dim:
                bettis.extend(0 for i in range(highest_dim - len(bettis)))
            bettilist.append(bettis)
            np.save(save_path + '_bettilist', np.array(bettilist))

    # Some filtered facet list might only contain facets smaller than highest_dim. In this case, GUDHI does not compute
    # the highest Bettis we were expecting by setting highest_dim. In this case, however, the Bettis that were not
    # computed are necessarily zero, since the dimension of the facets does not allow the formation of higher dimensional
    # voids. For example, if there cannot be tetrahedrons, Betti 3 is necesserily zero, because we cannot glue tetrahedrons
    # together and create a 4 dimensional void. The following loop adds zeros to the Bettis that were not computed if
    # such a situation arises and ensures that there a no issues in the construction/dimensions of the returned numpy array.
    length_longest_sublist = max(len(l) for l in bettilist)
    for sublist in bettilist:
        if len(sublist) < length_longest_sublist:
            print('HERE')
            sublist.extend(0 for i in range(length_longest_sublist - len(sublist)))

    return np.array(bettilist)

def plot_betti_persistence(bettiarray, threshold_array, sumfilt=True):
    """
    This function plots a persistence bar code of the number of Betti X throughout the filtrations. It plots N bar code
    where N is the number of columns in bettiarray (which correspond to a specific Betti number).
    Parameters
    ----------
    bettiarray (numpy array) : an array in which the columns are the Betti numbers and the lines are the filtered facet
                               lists of a bipartite graph. Use compute_bettis_for_persistence() to get this array.
    threshold_array (list) : List of the threshold used to filter the bipartite matrix. Its number of elements must match
                             the number of lines of bettiarray

    Returns None. It plots and shows the bar codes.
    -------

    """
    # This variable contains the number of subplots that are going to be present.
    subplotnb = str(len(bettiarray[0]))

    # Each column corresponds to a Betti number. The column contains the number of Betti X at each threshold.
    # For each column (i.e. Betti), we find for how many thresholds (and which one) we had a certain value of
    # Betti X.
    for betticolumn_index in range(0, len(bettiarray[0])):

        bar_heights = []
        bar_starting_pos = []
        number_of_betti = []
        previous_change_index = 0
        i = 1

        # This loop computes the length of the bars, meaning the distance between the two thresholds through which
        # a certain number of Betti X persists. It also stores the position of the bar on the x axis.
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


        # This section plots each bar code for every Betti number.

        plt.rcdefaults()

        # Declares a subplot in the form subplot(XYZ) where X is the number of subplots, Y the selected column
        # on the plot (only 1) and the line (for each betti, we need to use a new line on the figure).
        ax = plt.subplot(int(subplotnb+'1'+str(betticolumn_index+1)))
        y_pos = number_of_betti
        left = bar_starting_pos
        ax.barh(y_pos, bar_heights, align='center', left=left)
        ax.set_yticks(y_pos)
        if not sumfilt:
            ax.invert_xaxis()
        if betticolumn_index == 0:
            ax.set_title('Persistence of the Betti numbers')
        ax.set_ylabel('Betti ' + str(betticolumn_index))


    plt.xlabel('test')
    # This command makes it possible to remove space between the subplots.
    plt.subplots_adjust(hspace=.0)
    ax.set_xlabel('Threshold')

    plt.show()

def proportion_of_species(path_pruned, maxnumber=2611):
    """
    This function finds the proportion of species that are considered in a filtration of the biadjacency matrix.
    To do so, it uses the pruned facet list associated with this filtration and finds the highest node index.
    By adding 1 to this index, the number we obtain corresponds to the number of species (we add 1 because the
    pruned facet list are zero indexed).

    Parameters
    ----------
    path (str) : Path to the pruned facet list
    maxnumber (int) : Total number of species in a given biadjacency matrix. (final_OTU=2611).

    Returns The proportion of species present in the facet list.
    -------

    """

    nodes_indices = []

    with open(path_pruned) as f:
        for l in f:
            nodes_indices.append(max([int(x) for x in l.strip().split()]))

    # We add 1 since the species are zero indexed.
    number_of_species = max(nodes_indices) + 1

    return number_of_species/maxnumber

def find_original_number_of_species(path_to_original_matrix, skip_rows=0, use_cols=range(1,39)):
    """
    Find the total number of species in the original matrix
    Parameters
    ----------
    path_to_original_matrix (str) : Path to the original matrix
    skip_rows (int) : Number of rows to be skipped, ideally 0
    use_cols (range) : Column to use ideally all

    Returns The total number of species.
    -------

    """

    matrix = np.loadtxt(path_to_original_matrix, skiprows=0, usecols=range(1, 39))

    return matrix.shape[0]


def plot_species_prop(proportions, thresholdlist):

    plt.plot(thresholdlist, proportions)
    plt.xlabel('Threshold')
    plt.ylabel('Species proportion %')

def fill_folder_with_facetlists(path_to_matrix, save_path, thresholdlist, skip_row=0, use_col=range(1, 39), func='sum'):
    """
    This function generates filtered instances from a complete matrix and a threshold list. It transforms the matrix into
    a filtered edgelist, then a facet list, then a pruned facet list. It also computes the proportion
    Parameters
    ----------
    path_to_matrix (str) : Path to the complete cooccurrence matrix
    save_path (str) : Path and name of the isntances (ex : blabla/bla/instance )
    thresholdlist (list of float) : List of thresholds to be used to filter the matrix
    skip_row (int) : Number of rows to skip in the complete matrix. For final_OTU, use skiprows=0; SubOtu4000 use skiprows=1
    use_col (range) : range of column to use. For final_OTU use usecols=range(1, 39); SubOtu4000 use range(1, 35)
    func (str) : Either 'sum' or 'thresh'. If 'sum' we use the matrix_filter_sum_to_prop() function. If it's 'thresh' we
                 use the function matrix_filter()

    Returns None ; It saves the filtered instances the same folder specified by path
    -------

    """
    i = 1
    for thresh in thresholdlist:
        print('Working on threshold ', i, ' which value is ', thresh)
        matrix1 = np.loadtxt(path_to_matrix, skiprows=skip_row, usecols=use_col)
        mat = normalize_columns(matrix1)
        if func == 'sum':
            matrix = matrix_filter_sum_to_prop(mat, prop_threshold=thresh)
        elif func == 'thresh':
            matrix = matrix_filter(mat, thresh)
        matrix = reduce_matrix(matrix)
        to_nx_edge_list_format(matrix, out_name=save_path+str(i)+'.txt')
        i += 1
    ilist = np.arange(1, i)

    for i in ilist:
        to_max_facet(save_path+str(i)+'.txt', 1, save_path+'facet'+str(i)+'.txt')
        to_pruned_file(save_path+'facet'+str(i)+'.txt', save_path+'facet'+'pruned'+str(i)+'.txt')


if __name__ == '__main__':
    # Instruction : Change the array for the thresholdlist. Keep in mind that it depends on the type of filter that you
    # are going to use later on.

    thresholdlist = np.linspace(0.001, 0.01, 1000)
    thresholdlist = np.arange(0.1, 1, 0.01)
    ilist = np.arange(1, len(thresholdlist) + 1)

    # Instruction :  Change path / index that we're going to use to name the filtered instances using the thresholds from the threshold list
    save_path = '/home/xavier/Documents/Projet/Betti_scm/persistencetest/simpletest'
    proportions = []

    # Once this function has been run, running it again is a waste and should be put in commentary
    # Instruction : Change the parameters so they are coherent with what is desired. / If the folder is already full of
    # generated instances put in commentary
    fill_folder_with_facetlists('final_OTU.txt', save_path, thresholdlist)

    for i in ilist:
        proportions.append(proportion_of_species(save_path + 'facet' + 'pruned' + str(i) + '.txt'))

    plt.plot(thresholdlist, proportions)
    plt.show()

    #Once we have every instances (with fill_folder_with_facetlists()) we can compute the betti numbers of the instances.
    #np.save(save_path+'_thresholdlist', thresholdlist)
    #bettiarr = compute_bettis_for_persistence(ilist[0], ilist[-1], save_path+'facetpruned', 3)

    exit()
    #np.savez(path+'01-08-001', bettiarr, thresholdlist)


    # Instruction : Load the data for the bar code and change the argument of the function plot_betti_persistence so they
    #               are coherent with the filter that was used
    data = np.load(save_path + '001-02-10000-dimskel3' +'.npz')
    bettiarr = data['arr_0']
    thresholdlist = data['arr_1']
    plot_betti_persistence(bettiarr, thresholdlist, sumfilt=True)









