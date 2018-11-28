#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import networkx as nx
import gudhi
import numpy as np
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
    i = 0
    #print('FACETLIST : ', facetlist)
    for facet in facetlist:

        # This condition and the loop it contains break a facet in all its n choose k faces. This is more
        # memory efficient than inserting a big facet because of the way GUDHI's SimplexTree works. The
        # dimension of the faces has to be at least one unit bigger than the skeleton we use.
        if len(facet) > highest_dim+1:
            for face in itertools.combinations(facet, highest_dim+1):
                st.insert(face)
        else:
            st.insert(facet)
        i += 1

    # This function has to be launched in order to compute the Betti numbers
    st.persistence()

    return st.betti_numbers()

def compute_and_store_bettis(path, highest_dim, save_path):
    bettilist = []
    facetlist = []

    with open(path, 'r') as file:
        for l in file:
            facetlist.append([int(x) for x in l.strip().split()])

        bettis = compute_betti(facetlist, highest_dim)
        if len(bettis) < highest_dim:
            bettis.extend(0 for i in range(highest_dim - len(bettis)))
        bettilist.append(bettis)
        print('icit ', bettis)
        np.save(save_path + '_bettilist', np.array(bettilist))


def compute_and_store_bettis_from_instances(instance_path, idx_range, highest_dim, save_path):

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
    instance_path = '/home/xavier/Documents/Projet/Betti_scm/crime/alice1_crime_instance'

    data_path = '/home/xavier/Documents/Projet/Betti_scm/datasets/facet_list_c1_as_simplices.txt'

    idx_range = range(1,101)

    highest_dim = 2

    savepath_betti_instance = instance_path

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
