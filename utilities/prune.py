#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
"""
Prune facet list: removes all included facets in a facet list.

In the '' main '' section, this list can also be saved in a file.

Author: Alice Patania <alice.patania@isi.it>
Author: Jean-Gabriel Young <info@jgyoung.ca>
Author: Xavier Roy-Pomerleau <xavier.roy-pomerleau.1@ulaval.ca>
"""
import itertools


def prune(facet_list):
    """Remove included facets from a collection of facets.

    Notes
    =====
    Facet list should be a list of frozensets
    """
    # organize facets by sizes
    sizes = {len(f) for f in facet_list}
    facet_by_size = {s: [] for s in sizes}
    for f in facet_list:
        facet_by_size[len(f)].append(f)
    # remove repeated facets
    for s in facet_by_size:
        facet_by_size[s] = list({x for x in facet_by_size[s]})
    # remove included facets and yield
    for ref_size in sorted(list(sizes), reverse=True):
        for ref_set in sorted(facet_by_size[ref_size]):
            for s in sizes:
                if s < ref_size:
                    facet_by_size[s] = [x for x in facet_by_size[s]
                                        if not x.issubset(ref_set)]
        for facet in facet_by_size[ref_size]:
            yield facet

def to_pruned_file(path, outpath):
    import argparse as ap
    prs = ap.ArgumentParser(description='Prune facet list.')
    prs.add_argument('facet_list', type=str, nargs='?',
                     help='Path to facet list.')
    args = prs.parse_args()

    ### Instruction : Change path to facet list here :
    # args.facet_list = '/home/xavier/Documents/Projet/scm/utilities/severeadathr_finalOTU.txt'
    args.facet_list = path
    facet_list = []
    with open(args.facet_list, 'r') as f:
        for line in f:
            # frozenset is mandatory: allows for comparison
            facet = frozenset([int(x) for x in line.strip().split()])
            facet_list.append(facet)
        stringtowrite = ''
        listfacetlength = []
        for facet in prune(facet_list):
            print(" ".join([str(v) for v in facet]))
            dumbstring = " ".join([str(v) for v in facet])
            print(len(dumbstring.split()))
            listfacetlength.append(len(dumbstring.split()))

            stringtowrite = stringtowrite + " ".join([str(v) for v in facet]) + '\n'
        print('The mean facet length is : ', np.mean(np.array(listfacetlength)))

        ### Instruction : Change the path and name of the saved file here :
        with open(outpath, 'w') as outfile:
            outfile.write(stringtowrite)


if __name__ == '__main__':
    """
        Make sure to read this before launching the code :

        There are several lines to update each time this code is used. To find which parameters have to be tuned, search
        (CTRL+F) for ''Instruction'' (without the quotation marks) or ''###''.


        Make sure that the facet list is ZERO INDEXED
        """

    print('Don\'t forget to read the documentation at the beginning of the main!')

    # Options parser.
    import argparse as ap
    prs = ap.ArgumentParser(description='Prune facet list.')
    prs.add_argument('facet_list', type=str, nargs='?',
                     help='Path to facet list.')
    args = prs.parse_args()

    ### Instruction : Change path to facet list here :
    #args.facet_list = '/home/xavier/Documents/Projet/scm/utilities/severeadathr_finalOTU.txt'
    args.facet_list = '/home/xavier/Documents/Projet/scm/utilities/test.txt'
    facet_list = []
    with open(args.facet_list, 'r') as f:
        for line in f:
            # frozenset is mandatory: allows for comparison
            facet = frozenset([int(x) for x in line.strip().split()])
            facet_list.append(facet)
        stringtowrite = ''
        listfacetlength = []
        for facet in prune(facet_list):
            print(" ".join([str(v) for v in facet]))
            dumbstring = " ".join([str(v) for v in facet])
            print(len(dumbstring.split()))
            listfacetlength.append(len(dumbstring.split()))

            stringtowrite = stringtowrite + " ".join([str(v) for v in facet]) + '\n'
        print('The mean facet length is : ', np.mean(np.array(listfacetlength)))


        ### Instruction : Change the path and name of the saved file here :
        with open('/home/xavier/Documents/Projet/scm/utilities/prubntest.txt', 'w') as outfile:
            outfile.write(stringtowrite)

        print('Done.\nAlways make sure that the resulting facet list is zero indexed.\nDon\'t forget to change the saving file name so your file does not get written over.')
