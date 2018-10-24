import numpy as np
import scipy as sp
import random
import networkx as nx

def facet_reduction(facetlist):
    """
    This function would work on a facetlist.txt
    Parameters
    ----------
    facetlist

    Returns
    -------

    """
    #Draw a random facet (or line number)

    #Assign facet to variable

    #find the length of the facet

    #Find the N choose K, where K = length facet - 1, facets of length = length facet - 1

    #Delete the randomly selected facet from the list

    #create the N choose K subfacets

    #Add the N choose K subfacets to the list.

    return facetlist

def graph_facet_reduction(bipartite_representation):
    sets = nx.algorithms.bipartite.sets(g)
    node_set_of_facets = sets[0]

    #draw a random node
    #delete random node
    random.shuffle(node_list)

    proj = nx.algorithms.projected_graph(g, sets[0])


def node_perc(facetlist):


if __name__ == '__main__':
