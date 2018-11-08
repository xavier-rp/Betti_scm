import numpy as np
import scipy as sp
import itertools
import gudhi
import json
import time
from utilities.prune import  prune


def decompose_facet(facetlist, previ=0):
    """
    Takes a random facet in a facet list and decomposes it in its N choose (N-1) faces, where N is the number of
    nodes in the facet.  It then returns the updated facet list where the original facet is deleted and replaced
    by its N choose (N-1) faces.
    Parameters
    ----------
    facetlist  (list of lists) : Facet list generated from the Null_model.py module
    previ (int) : This variable plays two roles. Its first role is to represent the first index at which we have a facet
                  of length > 1 in the sorted (by length of facet) facetlist. For instance, in [[1], [2], [3, 4]], previ
                  would be 2. Its second role is to indicate how many facet of size 1 we have, hence 2 in the previous
                  example. previ is also returned so that if we use the updated facet list another time in this function,
                  we can specify how many facet of size one we had in the previous iteration (hence the name prev i).
                  this way, the algorithm does not have to start from 0 each time we want to know the number of facets of
                  length 1. This information is mandatory, since the algorithm has to stop once the facet list only
                  contains facets of size 1.

    Returns
    -------
    facetlist (list of lists) : The updated facet list in which we deleted the randomly selected facet and added the
                                N choose (N-1) faces of the selected facet.
    previ (int) : The number of facets of size 1 before the update.
    """
    updatedfacetlist = []

    facetlist.sort(key=len)

    # this loop finds the number of facets of length 1 and stores it in the variable i.
    go_on = True
    stop_decompostion = False
    i = previ
    while go_on :
        if i == len(facetlist):
            stop_decompostion = True
            break
        if len(facetlist[i]) > 1:
            go_on = False
        else:
            i += 1
    previ = i
    if not stop_decompostion:
        # draw a facet > 1 randomly
        random_index = np.random.randint(i, len(facetlist))
        facet = facetlist[random_index]

        # delete the facet in the list and replace it by its N choose N-1 faces
        del(facetlist[random_index])
        for face in itertools.combinations(facet, len(facet)-1):
            facetlist.append(frozenset(face))
        for facet in prune(facetlist):
            print(facet)
            updatedfacetlist.append(facet)


    return updatedfacetlist, previ

def higher_order_link_percolation(origin_path, save_path):
    with open(origin_path, 'r') as file:
        facetlist = json.load(file)

    # This loop transforms the sub lists in the facet list into frozen sets.
    indx = 0
    while indx < len(facetlist):
        facetlist[indx] = frozenset(facetlist[indx])
        indx += 1
    # We stop the decomposition of the facets when each node is isolated (part of a 0-simplex). We thus need to know
    # how many isolated nodes there will be. Since the facet lists are zero-indexed, the maximal index that we can find
    # corresponds to the total number of nodes - 1
    length_totally_decomposed = max([max(facet) for facet in facetlist]) + 1
    previ = 0
    i = 1
    while max(len(facet) for facet in facetlist) != 1:
        facetlist, previ = decompose_facet(facetlist, previ)
        with open(save_path + str(i) + '.json', 'w') as outfile:
            indx = 0
            savedfacetlist = []
            while indx < len(facetlist):
                savedfacetlist.append(list(facetlist[indx]))
                indx += 1
            json.dump(savedfacetlist, outfile)
        i += 1



if __name__ == '__main__':
    start = time.time()
    path = r'/home/xavier/Documents/Projet/Betti_scm/finalOTU_thresh01/final_thresh01_instance1.json'
    higher_order_link_percolation(path, '/home/xavier/Documents/Projet/Betti_scm/percolationtest2/percotest')
    exit()
    with open(path, 'r') as f:
        facetlist = json.load(f)
    print(len(facetlist[-1]))
    previouslen=0
    nowlen = 10
    previ = 0
    while previouslen != nowlen:
        previouslen= len(facetlist)
        facetlist, previ = decompose_facet(facetlist, previ)
        nowlen = len(facetlist)
    print(nowlen, time.time()-start)