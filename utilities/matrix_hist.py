import numpy as np
import matplotlib.pyplot as plt


def to_konect_format(filename_or_matrix='SubOtu4000.txt', out_name='edgelist_SubOtu4000.txt'):
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

    G = nx.algorithms.bipartite.from_biadjacency_matrix(A)
    nx.write_edgelist(G, out_name, data=False)


def count_number_of_individuals(filename_or_matrix='final_OTU.txt', density=True):
    if isinstance(filename_or_matrix, str):
        ### Instruction : skiprows and usecols have to be updated depending on the file you use
        # Skiprows and usecols have to be adjusted depending on the file used.
        # For final_OUT, use skiprows=0 and usecols=range(1,39)
        # For SubOtu4000 use skiprows=1 and usecols=range(1,35)
        mat = np.loadtxt(filename_or_matrix, skiprows=0, usecols=range(1, 39))
    else:
        # This means that filename_or_matrix is a numpy matrix
        mat = filename_or_matrix

    total_number_of_individuals = []
    for specie_index in range(0, mat.shape[0]):
        total_number_of_individuals.append(np.sum(mat[specie_index,:]))

    if density:
        total_number_of_individuals = total_number_of_individuals/np.sum(total_number_of_individuals)

    return np.arange(0, mat.shape[0]), total_number_of_individuals


if __name__ == '__main__':

    species_idx, total_number = count_number_of_individuals()
    print(len(species_idx), len(total_number))
    print(max(total_number), np.where(total_number == max(total_number)), species_idx[np.where(total_number == max(total_number))])
    plt.bar(species_idx, total_number, log=True)
    plt.xlabel('Specie\'s index')
    plt.ylabel('Proportion')
    plt.show()