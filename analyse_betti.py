#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import networkx as nx
import gudhi
import numpy as np
import scipy as sp
import scipy.misc
import simplicial
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import time
import itertools
"""
Author: Xavier Roy-Pomerleau <xavier.roy-pomerleau.1@ulaval.ca>

In this module we compute the Betti numbers of a dataset and its randomized instances. We also plot the Betti number
distribution and the graph representing the real data.

"""

def facet_list_to_graph(facet_list):
    """Convert a facet list to a bipartite graph"""
    g = nx.Graph()
    for f, facet in enumerate(facet_list):
        for v in facet:
            g.add_edge('v' + str(v), 'f' + str(f))  # differentiate node types
    return g

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
    for f, facet in enumerate(facet_list):
        g.add_node('f'+str(f), bipartite = 1)
        facet_node_list.append('f'+str(f))
        for v in facet:
            g.add_node('v'+ str(v), bipartite = 0)
            g.add_edge('v' + str(v), 'f' + str(f))  # differentiate node types
    return g, facet_node_list


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

def highest_possible_betti(facetlist):

    # Sort the facet list by ascending order of facet size
    facetlist.sort(key=len)

    # This loop counts the number of facets of size 1
    i = 0
    while i < len(facetlist):
        if len(facetlist[i]) != 1:
            break
        i += 1
    # If we only have facets of size 1, there are no facets bigger than 1
    if i == len(facetlist):
        facets_bigger_than_one = []
    # If the previous loop broke at
    else:
        facets_bigger_than_one = facetlist[i:]

    length = len(facets_bigger_than_one)
    if length >= 3:
        min_size = len(facets_bigger_than_one[0])
        max_size = len(facets_bigger_than_one[-1])

        nb_facets_histogram_by_size = []
        size_list = []
        count_list = []
        previous_facet = facets_bigger_than_one[0]
        i = 1
        count = 0
        # This loop counts the number of facets of the same size and stores them in a list [size, count] that is
        # also stored in a list (nb_facets_histogram_by_size)
        while i < length:
            present_facet = facets_bigger_than_one[i]
            if len(previous_facet) == len(present_facet):
                count += 1
            else:
                count += 1
                nb_facets_histogram_by_size.append([len(previous_facet), count])
                size_list.append(len(previous_facet))
                count_list.append(count)
                count = 0
            previous_facet = present_facet
            i += 1
        count += 1
        nb_facets_histogram_by_size.append([len(previous_facet), count])
        size_list.append(len(previous_facet))
        count_list.append(count)
        size_list = np.array(size_list)
        count_list = np.array(count_list)


        # If there are N facets, we can, in theory, have a non zero Betti N-2, provided that these N facets have
        # a minimum size of N - 1.
        # Therefore : If the maximal size that we can find in the list is lower than the minimum size (i.e. lower than
        # N-1) required to have a non zero Betti N - 2, we know that all the Betti numbers > max_size - 1 are zero.
        # Indeed, to build Betti numbers of dimension 'd', we need facets of size d + 1. So if our max size is 'k' the
        # first Betti number we need to look at is k - 1 (i.e. max_size - 1)
        if max_size < length - 1:
            considered_size = max_size
        # If there are facets of size higher than or equal to the minimum size required to have a non zero Betti N - 2,
        # the first Betti number we need to look at is built with facets of size N - 1 and bigger. This means that
        # we don't have to look at the Betti numbers > N - 2, that would in principle be non trivial if we didn't know
        # the number of facets
        else:
            considered_size = length - 1

        i = 0
        for sublist in nb_facets_histogram_by_size :
            if sublist[0] >= considered_size:
                break
            # i is the index of the first sublist that matches the condition
            i += 1

        for size in np.arange(considered_size, 2, - 1):
            nb_of_respecting_facets = np.sum(count_list[np.where(size_list >= size)])
            if nb_of_respecting_facets >= size + 1:
                break
        betti = size - 1


    else:
        betti = 0

    return betti
if __name__ == '__main__':
    # Instruction : Change the path to the instances. Must not contain the index nor the extension of the file (added
    #               automatically in a function)
    instance_path = '/home/xavier/Documents/Projet/Betti_scm/crime/crime_instance'

    # Instruction : Change the path to the original data. This path must be complete (with the extension)
    data_path = '/home/xavier/Documents/Projet/Betti_scm/datasets/crime_facet_list.txt'

    #facetlist = []
    #with open(data_path, 'r') as file:
    #    for l in file:
    #        facetlist.append([int(x) for x in l.strip().split()])
    #print(len(facetlist[0]))
    #exit()
    #G, site_node_list = facet_list_to_bipartite(facetlist)

    #sets = nx.algorithms.bipartite.sets(G, top_nodes=site_node_list)

    #proj = nx.algorithms.projected_graph(G, sets[1])
    #print(proj.number_of_edges())
    #exit()
    #pos = nx.spring_layout(proj)

    #nx.draw_networkx(proj, pos=pos)
    #nx.draw_networkx_nodes(G, pos, nodelist=sets[0], node_color='b', node_size=70)
    #nx.draw_networkx_nodes(G, pos, nodelist=sets[1], node_color='r', node_size=90, node_shape='^')
    #nx.draw_networkx_edges(G, pos)

    #plt.show()


    #print(proj.number_of_edges)

    # Instruction : Change the range so it matches the indices of the instances that we want to process
    idx_range = range(1,101)

    # Instruction : Change the dimension of the j-skeleton used to compute Betti numbers
    highest_dim = 2

    # Instruction : Change the path where you save the Betti numbers of the instances
    savepath_betti_instance = instance_path

    # Instruction : Change the path where you save the Betti numbers of the original dataset
    savepath_betti_data = '/home/xavier/Documents/Projet/Betti_scm/datasets/facet_list_c1_as_simplicestest'

    compute_and_store_bettis_from_instances(instance_path, idx_range, highest_dim, savepath_betti_instance)

    compute_and_store_bettis(data_path, highest_dim, savepath_betti_data)

    bettiarray_instance = np.load(savepath_betti_instance + '_bettilist.npy')
    bettiarray_data = np.load(savepath_betti_data + '_bettilist.npy')

    plot_betti_dist(bettiarray_instance, bettiarray_data)

    """
    #OLD CODE, MIGHT STIL BE USEFULL
    start = time.time()

    # Initialise the list that will contain the Betti numbers
    betti0list = []
    betti1list = []
    betti2list = []

    # Instruction : Change the range to include the instances you want to use.
    for idx in range(1, 101):
        with open('/home/xavier/Documents/Projet/Betti_scm/crime/alice1_crime_instance' + str(idx) + '.json', 'r') as file:
            facetlist = json.load(file)

            st = gudhi.SimplexTree()
            i = 0
            for facet in facetlist:
                #print(facet)
                # This condition and the loop it contains break a facet in all its n choose k faces. This is more
                # memory efficient than inserting a big facet because of the way GUDHI's SimplexTree works. The
                # dimension of the faces has to be at least one unit bigger than the skeleton we use (see next instruction)
                if len(facet) > 4:
                    for face in itertools.combinations(facet, 4):
                        st.insert(face)
                else:
                    st.insert(facet)
                #print(i)
                i += 1
            startskel = time.time()

            # Instruction : Change the dimension of the skeleton you want to use. From a skeleton of dimension d, we can
            # only compute Betti numbers lower than d. The dimension of the skeleton has to be coherent with the
            # dimension of the largest facet allowed, meaning that if we break a facet into its N choose K faces, where
            # K == 3, than the dimension of the skeleton has to be d <= K-1
            skel = st.get_skeleton(3)
            sk = gudhi.SimplexTree()
            for tupl in skel:
                sk.insert(tupl[0])
                startpers = time.time()

            # This function has to be launched in order to compute the Betti numbers
            sk.persistence()
            startbet = time.time()
            Bettis = sk.betti_numbers()

            betti0list.append(Bettis[0])
            betti1list.append(Bettis[1])
            #betti2list.append(Bettis[2])

            print('Betti : ', Bettis, compute_betti(facetlist, 3))
    print(np.mean(betti0list), np.mean(betti1list))
    print('Total time : ', time.time() - start)



    ### Instruction : change the file path to the actual data.

    site_facet_list = []

    with open("/home/xavier/Documents/Projet/Betti_scm/datasets/facet_list_c1_as_simplices.txt") as f:
        for l in f:
            site_facet_list.append([int(x) for x in l.strip().split()])


    st = gudhi.SimplexTree()
    i = 0
    for facet in site_facet_list:
        print(facet)
        # Instruction change the number here so it is coherent with the analogous loop above
        if len(facet) > 3:
            for face in itertools.combinations(facet, 3):
                st.insert(face)
        else:
            st.insert(facet)
        print(i)
        i += 1

    print('_____________________________________________________________')

    startskel = time.time()
    # Instruction : change the dimension of the skeleton
    skel = st.get_skeleton(2)
    sk = gudhi.SimplexTree()
    for tupl in skel:
        sk.insert(tupl[0])

    startpers = time.time()


    sk.persistence()
    print('Time to persistence analysis : ', time.time() - startpers)

    startbet = time.time()
    bettireal = sk.betti_numbers()
    print('BETTIREAL : ', bettireal)

    # Plot the distributions of Betti0 and Betti1
    # Instruction : For each figure, change the bins for an appropriate value
    plt.figure(40)

    plt.hist(betti0list, bins=np.arange(0,1000)-0.5, density=True)
    #hist = np.histogram(betti0list, bins=2)
    #print(hist)
    #plt.hist()
    # sns.distplot(betti0list, hist=True)
    plt.plot([bettireal[0], bettireal[0]], [0, 1], color="#ff5b1e")
    plt.text(-0.29, 25, 'Real system', color="#ff5b1e")
    # plt.ylim(0, 30)
    plt.xlabel('Number of Betti 0')
    plt.ylabel('Normalized count')

    plt.figure(41)
    plt.hist(betti1list, bins=np.arange(0,400)-0.5, density=True)
    #plt.hist(betti1list2, density=True)
    #sns.distplot(betti1list)
    #sns.distplot(betti1list2)
    # sns.distplot(betti1list, hist=True)
    plt.plot([bettireal[1], bettireal[1]], [0, 1], color="#ff5b1e")
    plt.text(-0.29, 25, 'Real system', color="#ff5b1e")
    # plt.ylim(0, 30)
    plt.xlabel('Number of Betti 1')
    plt.ylabel('Normalized count')

    #plt.figure(42)
    #plt.hist(betti2list, bins=np.arange(0,6)-0.5, density=True)
    ## plt.hist(betti1list2, density=True)
    ## sns.distplot(betti1list)
    ## sns.distplot(betti1list2)
    ## sns.distplot(betti1list, hist=True)
    #plt.plot([bettireal[2], bettireal[2]], [0, 1], color="#ff5b1e")
    #plt.text(-0.29, 25, 'Real system', color="#ff5b1e")
    ## plt.ylim(0, 30)
    #plt.xlabel('betti2')
    #plt.ylabel('Histogram')
    plt.show()

    # Instruction : If the bipartite graph has to be plotted, remove 'exit()' so the algorithm can reach the lines.
    exit()

    G, site_node_list = facet_list_to_bipartite(site_facet_list)

    sets = nx.algorithms.bipartite.sets(G, top_nodes=site_node_list)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, nodelist=sets[0], node_color='b', node_size=70)
    nx.draw_networkx_nodes(G, pos, nodelist=sets[1], node_color='r', node_size=90, node_shape='^')
    nx.draw_networkx_edges(G, pos)

    plt.show()
    """
