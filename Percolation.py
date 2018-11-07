import numpy as np
import scipy as sp
import itertools
import gudhi
import json
import time


def decompose_facet(facetlist, previ=0):
    # Transform maximal facets to its N choose (N-1) faces.

    #with open(path, 'r') as file:
    #    facetlist = json.load(file)

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
        # draw a facet randomly
        # TODO ADD SOMETHING TO MAKE THE FUNCTION STOP IF i == len(facetlist)
        random_index = np.random.randint(i, len(facetlist))
        facet = facetlist[random_index]

        # delete the facet in the list and replace it by its N choose N-1 faces
        del(facetlist[random_index])
        for face in itertools.combinations(facet, len(facet)-1):
            facetlist.append(list(face))

    return facetlist, previ




if __name__ == '__main__':
    start = time.time()
    path = r'/home/xavier/Documents/Projet/Betti_scm/finalOTU_thresh01/final_thresh01_instance1.json'
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