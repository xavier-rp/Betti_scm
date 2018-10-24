import numpy as np
import subprocess as sp
import random
import networkx as nx
import matplotlib.pyplot as plt

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
            g.add_node('v'+str(v), bipartite = 0)
            g.add_edge('v' + str(v), 'f' + str(f))  # differentiate node types
    return g

def rejection_sampling(degree_seq, size_seq, seed=0):
    """Wrapper around the rejection sampler."""
    # Write sequence to files
    with open('/tmp/degrees.txt', 'w') as f:
        print(" ".join([str(x) for x in degree_seq]), file=f)
    with open('/tmp/sizes.txt', 'w') as f:
        print(" ".join([str(x) for x in size_seq]), file=f)
    # Call sampler with subprocess
    command = ['bin/rejection_sampler', '--degree_seq_file=/tmp/degrees.txt', '--size_seq_file=/tmp/sizes.txt', '-d', str(seed)]
    proc = sp.run(command, stdout=sp.PIPE)
    # Read output as a facet list
    facet_list = []
    for line in proc.stdout.decode().split("\n"):
        if line.find("#") == -1 and line != "\n":
            facet_list.append([int(x) for x in line.strip().split()])
    return facet_list[:-1]


def simple_perc(network, occ_prob):

    ####### Si je voulais indexer les noeuds à enlever à partir de np.where, il faudrait que
    ####### la liste de noeud soit un array avec des éléments entiers pour label les noeuds
    network_copy = network.copy()
    node_list = list(network_copy.nodes())

    random.shuffle(node_list)

    randomarr = np.random.rand(len(node_list))

    nodes_to_remove_idx = np.where(randomarr > occ_prob)[0]

    #nodes_to_remove = list(node_list[i] for i in nodes_to_remove_idx)

    #network.remove_nodes_from(nodes_to_remove)
    # see which of the three methods is the fastest. t

    for idx in nodes_to_remove_idx:
        network_copy.remove_node(node_list[idx])

    #Voir Si la loop ci-dessous est plus rapide.
    return network_copy


    #for node in node_list:
    #    if np.random.rand() > occ_prob:
    #        network.remove_node(node)



def percolation(network):
    # Initial quantities we want to average
    Sr = []


    node_list = list(network.nodes())

    random.shuffle(node_list)
    #Initialise the network we'll build node by node from the network
    built_network = nx.Graph()


    #TODO : Checker si c'est bipartie, sinon il ne faut pas utiliser le number of nodes
    nb_nodes = nx.number_of_nodes(network)
    nb_new_nodes = 0

    dict_cluster_size = {}
    nodes_in_cluster = {}


    #while number of node is different
    #TODO : Potential for multiprocess
    while nb_new_nodes != nb_nodes:
        new_node = node_list[nb_new_nodes]
        #add the randomly selected node
        built_network.add_node(new_node, cluster_label=nb_new_nodes)
        new_node_cluster_label = nb_new_nodes

        dict_cluster_size[nb_new_nodes] = 1
        nodes_in_cluster[nb_new_nodes] = [new_node]


        # Find its neighbors in the original network
        neighbors = network.neighbors(new_node)

        for neighbor in neighbors:
            #check if neighbor is present in the built network
            if neighbor in built_network:
                #Adds an edge between neighbors
                built_network.add_edge(new_node, neighbor)

                neighbor_cluster_label = built_network.node[neighbor]['cluster_label']
                #Checks if both nodes are in the same cluster
                if neighbor_cluster_label != new_node_cluster_label:
                    #If nodes are not in the same cluster, check which cluster is the smallest
                    #It is faster to update the labels of nodes in the smallest cluster

                    if dict_cluster_size[new_node_cluster_label] < dict_cluster_size[neighbor_cluster_label]:
                        # appends the list of nodes of the smallest cluster to the list of nodes of the biggest
                        nodes_in_cluster[neighbor_cluster_label] = nodes_in_cluster[neighbor_cluster_label] + nodes_in_cluster[new_node_cluster_label]

                        # Changes the cluster size of the biggest cluster by adding the size of the smallest
                        dict_cluster_size[neighbor_cluster_label] = dict_cluster_size[neighbor_cluster_label] + dict_cluster_size[new_node_cluster_label]

                        # relabels the 'cluster_label'' of the nodes in the smallest cluster that are now in the biggest
                        for node in nodes_in_cluster[new_node_cluster_label]:
                            built_network.node[node]['cluster_label'] = neighbor_cluster_label

                        #delete the now inexistant small cluster from the dictionary
                        del dict_cluster_size[new_node_cluster_label]
                        del nodes_in_cluster[new_node_cluster_label]

                        new_node_cluster_label = neighbor_cluster_label

                    else:
                        # copy of the code in the if block, but neighbor_cluster_label and new_node_cluster_label are interchanged
                        # We enter this block if the node we just added belongs to a bigger cluster than the other.
                        # appends the list of nodes of the smallest cluster to the list of nodes of the biggest
                        nodes_in_cluster[new_node_cluster_label] = nodes_in_cluster[new_node_cluster_label] + nodes_in_cluster[neighbor_cluster_label]

                        # Changes the cluster size of the new cluster labeled by neighbor_cluster_label
                        dict_cluster_size[new_node_cluster_label] = dict_cluster_size[new_node_cluster_label] + \
                                                                    dict_cluster_size[neighbor_cluster_label]

                        # relabels the cluster labels of the nodes of the smallest cluster«
                        for node in nodes_in_cluster[neighbor_cluster_label]:
                            built_network.node[node]['cluster_label'] = new_node_cluster_label

                        # delete the now inexistant small cluster
                        del dict_cluster_size[neighbor_cluster_label]
                        del nodes_in_cluster[neighbor_cluster_label]


        nb_new_nodes += 1

    #Verify if the built network is identical to the network entered as a parameter.
    print('Is isomorphic? ', nx.is_isomorphic(built_network, network))

    return built_network



    #At the end. check if built_network is isomorphic with network

def projection_to_bipartite(proj):
    node_list = list(proj.nodes())
    facet_list_return = []
    facetlist = []
    for node in node_list:

        neighbor_list = list(proj.neighbors(node))
        neighbor_set = set(proj.neighbors(node))
        edge_list = list(proj.edges(node))
        #adds the node to its neighbor set. This makes it easier to identify the facets containing the current node
        neighbor_set.add(node)

        #TODO : Potential for multiprocess
        for neighbor in neighbor_list:
            neighbors_of_neighbor_list = list(proj.neighbors(neighbor))
            neighbors_of_neighbor_set = set(proj.neighbors(neighbor))
            neighbors_of_neighbor_set.add(neighbor)
            intersection = list(set.intersection(neighbor_set,neighbors_of_neighbor_set))
            if len(intersection) >= 3:
                for nde in intersection:
                    nghbr_set = set(proj.neighbors(nde))
                    nghbr_set.add(nde)
                    facet = set.intersection(nghbr_set, neighbors_of_neighbor_set)
                    if facet not in facetlist:
                        facetlist.append(facet)
            else:
                facet = set.intersection(neighbor_set, neighbors_of_neighbor_set)
                if facet not in facetlist:
                    facetlist.append(facet)
    for facet in facetlist:
        facet_list_return.append(list(facet))

    return facet_list_return

if __name__ == '__main__':

    i=0


    print(rejection_sampling([2,2,2,1,1],[3,3,2],i))

    facet_list1 = rejection_sampling([2,2,2,1,1],[3,3,2],i)
    facet_list2 = rejection_sampling([2,2,2,1,1],[3,3,2],i)
    print(type(facet_list1))
    #del(facet_list1[0])
    print(facet_list1)

    g = facet_list_to_bipartite(facet_list1)
    sets = nx.algorithms.bipartite.sets(g)
    print(sets)
    if 'f0' in sets[0]:
        setchoice = 1
    else:
        setchoice = 0
    proj = nx.algorithms.projected_graph(g, sets[setchoice])
    #nx.draw_networkx(proj)
    #plt.show()

    projection_to_bipartite(proj)

    percolation(proj)

    gx = simple_perc(proj, 0.7)
    plt.figure(1)
    nx.draw_networkx(proj)
    plt.figure(2)
    nx.draw_networkx(gx)
    plt.show()




    g2 = facet_list_to_graph(facet_list2)

    g.node['f1']['bipartite'] = 0

    print(g.nodes(data=True))

    g.node['f1']['bipartite'] = 1

    g.remove_node('f0')

    print([n for n in g.neighbors('f1')])
    #g.add_node('f'+str(18), bipartite = 1)
    #g.add_node('v'+str(18), bipartite = 0)
    #g.add_edge('f18', 'v18')


    #sets = nx.algorithms.bipartite.sets(g)
    print(g.nodes(data=True))
    print(set(n for n,d in g.nodes(data=True) if d['bipartite']==0))

    top_nodes = set(n for n,d in g.nodes(data=True) if d['bipartite']==0)
    bottom_nodes = set(g) - top_nodes
    print(bottom_nodes)


    k = list(nx.connected_component_subgraphs(g))
    print(k)
    plt.figure(5)
    nx.draw_networkx(k[0])
    plt.figure(6)
    nx.draw_networkx(k[1])
    #print(sets)
    proj = nx.algorithms.projected_graph(g, top_nodes)
    #print(list(nx.find_cliques(proj)))
    plt.figure(1)
    nx.draw_networkx(proj)
    plt.figure(2)
    nx.draw_networkx(g)


    print(nx.is_bipartite(g))

    plt.figure(3)
    sets = nx.algorithms.bipartite.sets(g2)
    print(sets)
    proj = nx.algorithms.projected_graph(g2, sets[1])
    print(list(nx.find_cliques(proj)))
    nx.draw_networkx(proj)

    plt.figure(4)
    nx.draw_networkx(g2)
    print(nx.is_bipartite(g2))
    print(nx.is_isomorphic(g,g))
    print(nx.number_of_nodes(g2))

    plt.show()