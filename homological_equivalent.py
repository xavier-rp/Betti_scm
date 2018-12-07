#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import networkx as nx
import gudhi
import numpy as np
import json
import matplotlib.pyplot as plt
import time
import itertools
from utilities.biadjacency import to_nx_edge_list_format
from utilities.bipartite_to_max_facets import to_max_facet
from utilities.prune import to_pruned_file
"""
Author: Xavier Roy-Pomerleau <xavier.roy-pomerleau.1@ulaval.ca>

In this module we compute the Betti numbers of a dataset and its randomized instances. We also plot the Betti number
distribution and the graph representing the real data.

"""

def facet_list_to_bipartite(facet_list):
    """
    Convert a facet list to a bipartite graph. It also returns one of the set of the bipartite graph. This is usefull
    in the case where the graph is disconnected in the sense that no path exists between one or more node(s) and all
    the others. This case raises this exception when we want to plot the graph with networkX

    AmbiguousSolution : Exception – Raised if the input bipartite graph is disconnected and no container with all nodes
    in one bipartite set is provided. When determining the nodes in each bipartite set more than one valid solution is
    possible if the input graph is disconnected.

    In this case, we can use TODO to specify one of the node set (here the facets).
    Parameters
    ----------
    facet_list (list of list) : Facet list that we want to convert to a bipartite graph

    Returns
    g (NetworkX graph object) : Bipartite graph with two node sets well identified
    facet_node_list (list of labeled nodes) : Set of nodes that represent facets in the bipartite graph
    -------
    """
    g = nx.Graph()
    facet_node_list = []
    specie_node_list = []
    for f, facet in enumerate(facet_list):
        g.add_node('f'+str(f), bipartite = 1)
        facet_node_list.append('f'+str(f))
        for v in facet:
            g.add_node('v'+ str(v), bipartite = 0)
            specie_node_list.append('v'+str(v))
            g.add_edge('v' + str(v), 'f' + str(f))  # differentiate node types
    return g, facet_node_list, specie_node_list


def compute_betti(facetlist, highest_dim):
    """
    This function computes the betti numbers of the j-skeleton of the simplicial complex given in the shape of a facet
    list.
    Parameters
    ----------
    facetlist (list of list) : Sublists are facets with the index of the nodes in the facet.
    highest_dim (int) : Highest dimension allowed for the facets. If a facet is of higher dimension in the facet list,
                        the algorithm breaks it down into it's N choose k faces. The resulting simplicial complexe is
                        called the j-skeleton, where j = highest_dim.

    Returns
    -------
    The (highest_dim - 1) Betti numbers of the simplicial complexe.
    """

    st = gudhi.SimplexTree()
    for facet in facetlist:

        # This condition and the loop it contains break a facet in all its n choose k faces. This is more
        # memory efficient than inserting a big facet because of the way GUDHI's SimplexTree works. The
        # dimension of the faces has to be at least one unit bigger than the skeleton we use.
        if len(facet) > highest_dim+1:
            for face in itertools.combinations(facet, highest_dim+1):
                st.insert(face)
        else:
            st.insert(facet)


    # This function has to be launched in order to compute the Betti numbers
    st.persistence()

    return st.betti_numbers()

def compute_and_store_bettis(path, highest_dim, save_path):
    """
    Computes and saves the bettis numbers of the j-skeleton (j = highest_dim) of a facetlist stored in a .txt file.

    Parameters
    ----------
    path (str) : Path to the facetlist stored in a txt file. The shape of the data correspond to the shape of the outputs
                 of the null_model.py module.
    highest_dim (int) : Highest dimension allowed for the facets. If a facet is of higher dimension in the facet list,
                        the algorithm breaks it down into it's N choose k faces. The resulting simplicial complexe is
                        called the j-skeleton, where j = highest_dim. See The Simplex Tree: An Efficient Data Structure
                        for General Simplicial Complexes by Boissonat, Maria for information about j-skeletons.
    save_path (str) : Path where to save the bettinumbers (npy file)

    Returns
    -------

    """
    bettilist = []
    facetlist = []
    print('Working on ' + path)
    with open(path, 'r') as file:
        for l in file:
            facetlist.append([int(x) for x in l.strip().split()])

        bettis = compute_betti(facetlist, highest_dim)
        if len(bettis) < highest_dim:
            bettis.extend(0 for i in range(highest_dim - len(bettis)))
        bettilist.append(bettis)
        print(bettis)
        np.save(save_path + '_bettilist', np.array(bettilist))


def compute_and_store_bettis_from_instances(instance_path, idx_range, highest_dim, save_path):
    """
    Computes and saves the bettis numbers of the j-skeletons (j = highest_dim) of many facetlist generated with null_model.py
    and stored in .json files.
    Parameters
    ----------
    instance_path (str) : path to the instance, must not include the index of the instance nor its extension (added automatically)
    idx_range (range) : range of the indices of the instances
    highest_dim (int) : Highest dimension allowed for the facets. If a facet is of higher dimension in the facet list,
                        the algorithm breaks it down into it's N choose k faces. The resulting simplicial complexe is
                        called the j-skeleton, where j = highest_dim. See The Simplex Tree: An Efficient Data Structure
                        for General Simplicial Complexes by Boissonat, Maria for information about j-skeletons.
    save_path (str) : path where to save the array of betti numbers.

    Returns
    -------

    """

    bettilist = []

    for idx in idx_range:
        with open(instance_path + str(idx) + '.json', 'r') as file :
            print('Working on ' + instance_path + str(idx) + '.json')

            bettis = compute_betti(json.load(file), highest_dim)
            if len(bettis) < highest_dim:
                bettis.extend(0 for i in range(highest_dim - len(bettis)))
            bettilist.append(bettis)
            np.save(save_path + '_bettilist', np.array(bettilist))

def plot_betti_dist(bettiarray_instance, bettiarray_data):
    """
    This function plots the distributions of the betti numbers of the instances.
    Parameters
    ----------
    bettiarray_instance (np.array) : saved array using compute_and_store_bettis_from_instances
    bettiarray_data (np.array) : saved array using compute_and_store_bettis

    Returns
    -------

    """
    for column_index in range(0, bettiarray_instance.shape[1]):
        plt.figure(column_index)
        n, b, p = plt.hist(bettiarray_instance[:, column_index], bins=np.arange(0, max(bettiarray_instance[:, column_index]) + 0.5), density=True)
        plt.plot([bettiarray_data[0, column_index], bettiarray_data[0, column_index]], [0, max(n)], color="#ff5b1e")
        plt.text(-0.29, 25, 'Real system', color="#ff5b1e")
        # plt.ylim(0, 30)
        plt.xlabel('Number of Betti ' + str(column_index))
        plt.ylabel('Normalized count')

    plt.show()

if __name__ == '__main__':
    path = '/home/xavier/Documents/Projet/Betti_scm/datasets/diseasome_facet_list.txt'

    facetlist = []
    with open(path, 'r') as file:
        for l in file:
            facetlist.append([int(x) for x in l.strip().split()])

    print(compute_betti(facetlist, 2))

    g, facet_nodes, specie_node_list = facet_list_to_bipartite(facetlist)
    print(len(facet_nodes), len(specie_node_list))
    #sets = nx.algorithms.bipartite.sets(g, facet_nodes)
    matrix = nx.algorithms.bipartite.biadjacency_matrix(g, row_order=facet_nodes).transpose()
    to_nx_edge_list_format(matrix, out_name='/home/xavier/Documents/Projet/Betti_scm/datasets/diseasome_edgelist.txt')

    for i in [0, 1]:
        edgelistpath = '/home/xavier/Documents/Projet/Betti_scm/datasets/diseasome_edgelist.txt'
        to_max_facet(edgelistpath, i, '/home/xavier/Documents/Projet/Betti_scm/datasets/diseasome_pruned.txt')
        to_pruned_file('/home/xavier/Documents/Projet/Betti_scm/datasets/diseasome_pruned.txt', '/home/xavier/Documents/Projet/Betti_scm/datasets/diseasome_pruned.txt')

        facetlist = []
        with open('/home/xavier/Documents/Projet/Betti_scm/datasets/diseasome_pruned.txt', 'r') as file:
            for l in file:
                facetlist.append([int(x) for x in l.strip().split()])

        print(compute_betti(facetlist, 2))


