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
    g = nx.Graph()
    site_node_list = []
    for f, facet in enumerate(facet_list):
        g.add_node('f'+str(f), bipartite = 1)
        site_node_list.append('f'+str(f))
        for v in facet:
            g.add_node('v'+ str(v), bipartite = 0)
            g.add_edge('v' + str(v), 'f' + str(f))  # differentiate node types
    return g, site_node_list

if __name__ == '__main__':
    start = time.time()
    site_facet_list = []
    site_facet_string = []
    betti0list = []
    betti1list = []
    betti0list2 = []
    betti1list2 = []
    betti2list = []


    for idx in range(1, 101):
        with open('/home/xavier/Documents/Projet/scm/crime/alice1_crime_instance' + str(idx) + '.json', 'r') as file:
            facetlist = json.load(file)
            st = gudhi.SimplexTree()
            i = 0
            for facet in facetlist:
                #print(facet)
                if len(facet) > 3:
                    for face in itertools.combinations(facet, 3):
                        st.insert(face)
                else:
                    st.insert(facet)
                #print(i)
                i += 1
            startskel = time.time()
            skel = st.get_skeleton(2)
            sk = gudhi.SimplexTree()
            for tupl in skel:
                sk.insert(tupl[0])
                startpers = time.time()

            sk.persistence()
            startbet = time.time()
            Bettis = sk.betti_numbers()
            #if i < 1001:

            betti0list.append(Bettis[0])
            betti1list.append(Bettis[1])
            #betti2list.append(Bettis[2])
            #else:
            #    betti0list2.append(Bettis[0])
            #    betti1list2.append(Bettis[1])
            print('Betti : ', Bettis)
    print(np.mean(betti0list), np.mean(betti1list))
    print('Total time : ', time.time() - start)



    ### Instruction : change the file path to the actual data.
    #with open("/home/xavier/Documents/Projet/scm/utilities/severepruned_ada_finalOTU.txt") as f:
    #with open("/home/xavier/Documents/Projet/scm/utilities/prunedfinalotu-5.txt") as f:
    #    for l in f:
    #        site_facet_list.append([int(x) for x in l.strip().split()])
    #        site_facet_string.append([str(x) for x in l.strip().split()])
    with open("/home/xavier/Documents/Projet/scm/datasets/facet_list_c1_as_simplices.txt") as f:
        for l in f:
            site_facet_list.append([int(x) for x in l.strip().split()])
            site_facet_string.append([str(x) for x in l.strip().split()])
    #G, site_node_list = facet_list_to_bipartite(site_facet_list)

    st = gudhi.SimplexTree()
    i = 0
    for facet in site_facet_list:
        print(facet)
        if len(facet) > 3:
            for face in itertools.combinations(facet, 3):
                st.insert(face)
        else:
            st.insert(facet)
        print(i)
        i += 1

    print('_____________________________________________________________')

    startskel = time.time()
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


    plt.figure(40)

    plt.hist(betti0list, bins=np.arange(0,1000)-0.5, density=True)
    #hist = np.histogram(betti0list, bins=2)
    #print(hist)
    #plt.hist()
    # sns.distplot(betti0list, hist=True)
    plt.plot([bettireal[0], bettireal[0]], [0, 1], color="#ff5b1e")
    plt.text(-0.29, 25, 'Real system', color="#ff5b1e")
    # plt.ylim(0, 30)
    plt.xlabel('betti0')
    plt.ylabel('Histogram')

    plt.figure(41)
    plt.hist(betti1list, bins=np.arange(0,400)-0.5, density=True)
    #plt.hist(betti1list2, density=True)
    #sns.distplot(betti1list)
    #sns.distplot(betti1list2)
    # sns.distplot(betti1list, hist=True)
    plt.plot([bettireal[1], bettireal[1]], [0, 1], color="#ff5b1e")
    plt.text(-0.29, 25, 'Real system', color="#ff5b1e")
    # plt.ylim(0, 30)
    plt.xlabel('betti1')
    plt.ylabel('Histogram')

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


    exit()
    sets = nx.algorithms.bipartite.sets(G, top_nodes=site_node_list)
    print(sets[0])
    proj = nx.algorithms.projected_graph(G, sets[1])
    #nx.draw_networkx(proj)
    #plt.show()

    #pos = nx.spring_layout(G)
    #nx.draw_networkx_nodes(G, pos, nodelist=sets[0], node_color='b', node_size=70)
    #nx.draw_networkx_nodes(G, pos, nodelist=sets[1], node_color='r', node_size=90, node_shape='^')
    #nx.draw_networkx_edges(G, pos)
    # nx.draw_networkx(G)
    #plt.show()

    #simplices_dictio = nx.cliques_containing_node(G)
    simplices_dictio = nx.cliques_containing_node(proj)
    simplices_valuelist = list(simplices_dictio.values())
    simplices_flat_list = [item for sublist in simplices_valuelist for item in sublist]
    funct_nsimplices = list(set([tuple(l) for l in simplices_flat_list]))

    sc = simplicial.SimplicialComplex()
    sc.addSimplex([])
    sc.addSimplex(id=1)
    sc.addSimplex(id=2)
    #sc.addSimplex(id='c')
    #sc.addSimplex(id='d')
    sc.addSimplexWithBasis([1,2])
    sc.addSimplexWithBasis(['a','b','c','d'])
    sc.addSimplexWithBasis(site_facet_string[0])

    #betti0real = simplicial_complex.bettiNumbers()
    #betti1real = simplicial_complex.betti_number(1)
    #betti2real = simplicial_complex.betti_number(2)
    sc = simplicial.SimplicialComplex([('a','b','c','d')])
    #simplicial.faces(sc)
    #simplicial.SimplicialComplex(-19)
    funct_sc = simplicial.SimplicialComplex(funct_nsimplices)
    #betti0real = funct_sc.betti_number(0)
    #betti1real = funct_sc.betti_number(1)
    #betti2real = funct_sc.betti_number(2)
    #print(betti0real, betti1real, betti2real)

    plt.show()
    #for idx in range(1, 1001):
    for idx in range(1, 100):
        #with open('/home/xavier/Documents/Projet/scm/2sigma_filtered_finalotu/2sig_filtered_finalotu' + str(
        #        idx) + '.json', 'r') as file:
        with open('/home/xavier/Documents/Projet/scm/testsame' + str(
                idx) + '.json', 'r') as file:
        #with open('/home/xavier/Documents/Projet/scm/pickle/crimefacet' + str(
        #            idx) + '.txt', 'rb') as file:
            facetlist = json.load(file)
            #facetlist = pickle.load(file)
            print(idx)
        g, site_node_list = facet_list_to_bipartite(facetlist)
        if idx == 6 :
            print('stop')
            plt.figure(22)
            sets = nx.algorithms.bipartite.sets(g, top_nodes=site_node_list)
            proj = nx.algorithms.projected_graph(g, sets[1])
            nx.draw_networkx(proj)
            plt.show()
        if idx < 2 :
            plt.figure(idx+1)
            nx.draw_networkx(proj)
            #sets = nx.algorithms.bipartite.sets(g)
            #pos = nx.spring_layout(g)
            #nx.draw_networkx_nodes(g, pos, nodelist=sets[0], node_color='b', node_size=70)
            #nx.draw_networkx_nodes(g, pos, nodelist=sets[1], node_color='r', node_size=90, node_shape='^')
            #nx.draw_networkx_edges(g, pos)
        if idx == 5 :
            pass
            #plt.show()
        sets = nx.algorithms.bipartite.sets(g, top_nodes=site_node_list)
        proj = nx.algorithms.projected_graph(g, sets[1])
        simplices_dictio = nx.cliques_containing_node(proj)
        simplices_valuelist = list(simplices_dictio.values())
        simplices_flat_list = [item for sublist in simplices_valuelist for item in sublist]
        funct_nsimplices = list(set([tuple(l) for l in simplices_flat_list]))

        funct_sc = mogutda.SimplicialComplex(funct_nsimplices)
        betti0 = funct_sc.betti_number(0)
        betti1 = funct_sc.betti_number(1)
        print(betti0, betti1)
        betti0list.append(betti0)
        betti1list.append(betti1)
    #sets = nx.algorithms.bipartite.sets(G)
    #proj = nx.algorithms.projected_graph(G, sets[0])

    print(betti0real, betti1real)
    plt.figure(40)
    plt.hist(betti0list, density=True)
    #sns.distplot(betti0list, hist=True)
    #plt.plot([bett, real], [0, 30], color="#ff5b1e")
    plt.text(-0.29, 25, 'Real system', color="#ff5b1e")
    #plt.ylim(0, 30)
    plt.xlabel('betti0')
    plt.ylabel('Histogram')


    plt.figure(41)
    plt.hist(betti1list, density=True)
    #sns.distplot(betti1list, hist=True)
    # plt.plot([bett, real], [0, 30], color="#ff5b1e")
    plt.text(-0.29, 25, 'Real system', color="#ff5b1e")
    #plt.ylim(0, 30)
    plt.xlabel('betti1')
    plt.ylabel('Histogram')
    plt.show()

    print('Done')