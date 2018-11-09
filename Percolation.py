import numpy as np
import scipy as sp
import itertools
import gudhi
import json
import time
from utilities.prune import  prune

def destroy_highest_facet(facetlist, previ):
    """
        Takes a random facet in a facet list and decomposes it in its N disconnected nodes, where N is the number of
        nodes in the facet.  It then returns the updated facet list where the original facet is deleted and replaced
        by its disconnected nodes.
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
                                    N nodes contained in the facet.
        previ (int) : The number of facets of size 1 before the update.
        """
    updatedfacetlist = []

    facetlist.sort(key=len)

    # this loop finds the number of facets of length 1 and stores it in the variable i.
    go_on = True
    stop_decompostion = False
    i = previ
    while go_on:
        if i == len(facetlist):
            # If there are no more facets of size > 1, we need to stop decomposing, hence this condition
            stop_decompostion = True
            break
        if len(facetlist[i]) > 1:
            go_on = False
        else:
            i += 1
    previ = i
    # If there are no more facets of size > 1, we need to stop decomposing, hence this condition
    if not stop_decompostion:

        facet = facetlist[-1]
        # delete the facet in the list and replace it by its N choose 2 faces unless it's already a 2 simplex
        del (facetlist[-1])
        for node in facet :
            facetlist.append(frozenset([node]))
        for facet in prune(facetlist):
            updatedfacetlist.append(facet)

    return updatedfacetlist, previ

def destroy_random_facet(facetlist, previ):
    """
        Takes a random facet in a facet list and decomposes it in its N disconnected nodes, where N is the number of
        nodes in the facet.  It then returns the updated facet list where the original facet is deleted and replaced
        by its disconnected nodes.
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
                                    N nodes contained in the facet.
        previ (int) : The number of facets of size 1 before the update.
        """
    updatedfacetlist = []

    facetlist.sort(key=len)

    # this loop finds the number of facets of length 1 and stores it in the variable i.
    go_on = True
    stop_decompostion = False
    i = previ
    while go_on:
        if i == len(facetlist):
            # If there are no more facets of size > 1, we need to stop decomposing, hence this condition
            stop_decompostion = True
            break
        if len(facetlist[i]) > 1:
            go_on = False
        else:
            i += 1
    previ = i
    # If there are no more facets of size > 1, we need to stop decomposing, hence this condition
    if not stop_decompostion:
        # draw a facet > 1 randomly
        random_index = np.random.randint(i, len(facetlist))
        facet = facetlist[random_index]

        # delete the facet in the list and replace it by its N choose 2 faces unless it's already a 2 simplex
        del (facetlist[random_index])
        for node in facet :
            facetlist.append(frozenset([node]))
        for facet in prune(facetlist):
            updatedfacetlist.append(facet)

    return updatedfacetlist, previ

def random_facet_to_1_simplices(facetlist, previ=0):
    """
    Takes a random facet in a facet list and decomposes it in its N choose 2 faces, where N is the number of
    nodes in the facet.  It then returns the updated facet list where the original facet is deleted and replaced
    by its N choose 2 faces.
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
                                N choose 2 faces of the selected facet.
    previ (int) : The number of facets of size 1 before the update.
    """
    updatedfacetlist = []

    facetlist.sort(key=len)

    # this loop finds the number of facets of length 1 and stores it in the variable i.
    go_on = True
    stop_decompostion = False
    i = previ
    while go_on:
        if i == len(facetlist):
            # If there are no more facets of size > 1, we need to stop decomposing, hence this condition
            stop_decompostion = True
            break
        if len(facetlist[i]) > 1:
            go_on = False
        else:
            i += 1
    previ = i
    # If there are no more facets of size > 1, we need to stop decomposing, hence this condition
    if not stop_decompostion:
        # draw a facet > 1 randomly
        random_index = np.random.randint(i, len(facetlist))
        facet = facetlist[random_index]

        # delete the facet in the list and replace it by its N choose 2 faces unless it's already a 2 simplex
        del (facetlist[random_index])
        if len(facet) > 2:
            for face in itertools.combinations(facet, 2):
                facetlist.append(frozenset(face))
            for facet in prune(facetlist):
                print(facet)
                updatedfacetlist.append(facet)
        else:
            for face in itertools.combinations(facet, 1):
                facetlist.append(frozenset(face))
            for facet in prune(facetlist):
                print(facet)
                updatedfacetlist.append(facet)

    return updatedfacetlist, previ

def highest_facet_to_1_simplices(facetlist, previ=0):
    """
    Takes the biggest facet in a facet list and decomposes it in its N choose 2 faces, where N is the number of
    nodes in the facet.  It then returns the updated facet list where the original facet is deleted and replaced
    by its N choose 2 faces.
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
                                N choose 2 faces of the selected facet.
    previ (int) : The number of facets of size 1 before the update.
    """
    updatedfacetlist = []

    facetlist.sort(key=len)

    # this loop finds the number of facets of length 1 and stores it in the variable i.
    go_on = True
    stop_decompostion = False
    i = previ
    while go_on:
        if i == len(facetlist):
            # If there are no more facets of size > 1, we need to stop decomposing, hence this condition
            stop_decompostion = True
            break
        if len(facetlist[i]) > 1:
            go_on = False
        else:
            i += 1
    previ = i
    # If there are no more facets of size > 1, we need to stop decomposing, hence this condition
    if not stop_decompostion:
        # draw a facet > 1 randomly
        random_index = np.random.randint(i, len(facetlist))
        facet = facetlist[-1]

        # delete the facet in the list and replace it by its N choose 2 faces unless it's already a 2 simplex
        del (facetlist[-1])
        if len(facet) > 2:
            for face in itertools.combinations(facet, 2):
                facetlist.append(frozenset(face))
            for facet in prune(facetlist):
                print(facet)
                updatedfacetlist.append(facet)
        else:
            for face in itertools.combinations(facet, 1):
                facetlist.append(frozenset(face))
            for facet in prune(facetlist):
                print(facet)
                updatedfacetlist.append(facet)

    return updatedfacetlist, previ

def decompose_highest_facet(facetlist, previ=0):
    """
        Takes the biggest facet in a facet list and decomposes it in its N choose (N-1) faces, where N is the number of
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
    while go_on:
        if i == len(facetlist):
            # If there are no more facets of size > 1, we need to stop decomposing, hence this condition
            stop_decompostion = True
            break
        if len(facetlist[i]) > 1:
            go_on = False
        else:
            i += 1
    previ = i
    # If there are no more facets of size > 1, we need to stop decomposing, hence this condition
    if not stop_decompostion:
        # Select the last facet in the sorted facet list by size.
        facet = facetlist[-1]

        # delete the facet in the list and replace it by its N choose N-1 faces
        del (facetlist[-1])
        for face in itertools.combinations(facet, len(facet) - 1):
            facetlist.append(frozenset(face))
        for facet in prune(facetlist):
            print(facet)
            updatedfacetlist.append(facet)

    return updatedfacetlist, previ

def decompose_random_facet(facetlist, previ=0):
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
            # If there are no more facets of size > 1, we need to stop decomposing, hence this condition
            stop_decompostion = True
            break
        if len(facetlist[i]) > 1:
            go_on = False
        else:
            i += 1
    previ = i
    # If there are no more facets of size > 1, we need to stop decomposing, hence this condition
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

def higher_order_link_percolation(origin_path, save_path, perc_func=decompose_random_facet):
    with open(origin_path, 'r') as file:
        facetlist = json.load(file)

    # This loop transforms the sub lists in the facet list into frozen sets.
    indx = 0
    while indx < len(facetlist):
        facetlist[indx] = frozenset(facetlist[indx])
        indx += 1

    previ = 0
    i = 1
    while max(len(facet) for facet in facetlist) != 1:
        facetlist, previ = perc_func(facetlist, previ)
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
    higher_order_link_percolation(path, '/home/xavier/Documents/Projet/Betti_scm/percolationtest3/percotest', perc_func=destroy_highest_facet)
    exit()