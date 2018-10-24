#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
"""
Author: Jean-Gabriel Young <info@jgyoung.ca>

Convert a bipartite graph (in KONECT format) to maximal facet format.

Assumes that left side nodes and right side nodes are the different part of
the graph. Will reindex everything from 0, not necessarily conserving the
order.

"""


def read_edge_list(path):
    """Read edge list in KONECT format."""
    with open(path, 'r') as f:
        edge_list = set()
        for line in f:
            if not line.lstrip().startswith("%"):  # ignore commets
                e = line.strip().split()
                edge_list.add((int(e[0]) - 1, int(e[1]) - 1))  # 0 index
        return list(edge_list)

def read_nx_edge_list(path):
    """Read edge list that has been created with to_nx_edge_list_format() in module biadjacency.py. """
    with open(path, 'r') as f:
        """With the networkx format, nodes are indexed from 0 to cardinal(V+E), where V and E are the nodesets of the
        bipartite graph. As a result, the indices of the second node set (those in column two of the file), need to
        be reindexed by substracting the first element of the last line of the file + 1.
        """
        edge_list = set()
        linelist = f.readlines()
        lastline = linelist[-1].strip().split()
        index_to_substract = int(lastline[0]) + 1

    with open(path, 'r') as f:
        for line in f:
            if not line.lstrip().startswith("%"):  # ignore comments
                e = line.strip().split()
                edge_list.add((int(e[0]), int(e[1]) - index_to_substract))  # 0 index
        return list(edge_list)


def remap_bipartite_edge_list(edge_list):
    """Create isomoprhic edge list, with labels starting from 0."""
    remap_1 = dict()
    remap_2 = dict()
    new_id_1 = 0
    new_id_2 = 0
    for e in edge_list:
        if remap_1.get(e[0]) is None:
            remap_1[e[0]] = new_id_1
            new_id_1 += 1
        if remap_2.get(e[1]) is None:
            remap_2[e[1]] = new_id_2
            new_id_2 += 1
    return [(remap_1[e[0]], remap_2[e[1]]) for e in edge_list]


def facet_generator(sorted_edge_list, facet_col=0):
    """Generate facet list, from the sorted edge list."""
    vertex_col = int(not facet_col)
    prev_facet = sorted_edge_list[0][facet_col]
    facet_content = []
    for e in sorted_edge_list:
        curr_facet = e[facet_col]
        if curr_facet != prev_facet:
            yield facet_content
            facet_content.clear()
        facet_content.append(e[vertex_col])
        prev_facet = curr_facet
    yield facet_content

if __name__ == '__main__':
    """
    Make sure to read this before launching the code :

    There are several lines to update each time this code is used. To find which parameters have to be tuned, search
    (CTRL+F) for ''Instruction'' (without the quotation marks) or ''###''.

    """

    print('Don\'t forget to read the documentation at the beginning of the main!')


    # Options parser.
    import argparse as ap
    from operator import itemgetter
    prs = ap.ArgumentParser(description='Convert a KONECT bipartite graph ' +
                                        'to a list of maximal facets.')
    prs.add_argument('--col', '-c', type=int, default=0,
                     help='Column to use as facets (0 or 1).')
    prs.add_argument('edge_list_path', type=str, nargs='?',
                     help='Path to edge list.')
    args = prs.parse_args()


    ### Instruction : Change path to edge_list here :
    args.edge_list_path = '/home/xavier/Documents/Projet/scm/utilities/finalOTU_adaptive-5.txt'

    ### Instruction : Change column (0 or 1), so the one you chose represents the facets in the projection
    args.col = 1


    ### Instruction : Use either read_edge_list if both sets in your edge list are one-indexed or use
    ### read_edge_list_konect_minus in which you need change the number you have to substract from the index in your edge
    ### list to be zero indexed.

    #edge_list = read_edge_list(args.edge_list_path)
    edge_list = read_nx_edge_list(args.edge_list_path)
    #edge_list = read_edge_list_konect_minus(args.edge_list_path)

    if edge_list is not None:
        ### Instruction : You can allow for remaping. Can be usefull to make sure everything is zero indexed, but scrambles
        ### the meaning of the nodes

        #edge_list = remap_bipartite_edge_list(edge_list)
        edge_list = sorted(edge_list, key=itemgetter(args.col))

        # Used to save the facetlist
        stringtowrite = ''
        listfacetlength = []
        for max_facet in facet_generator(edge_list, args.col):
            print(" ".join([str(v) for v in sorted(max_facet)]))
            stringtowrite = stringtowrite + " ".join([str(v) for v in sorted(max_facet)]) +'\n'

        #print(np.mean(np.array(listfacetlength)))
        #print('__________________ \n', stringtowrite[0:])

        ### Instruction : change the path or the filename to make sure you don't overwrite
        with open('test.txt', 'w') as outfile:
            outfile.write(stringtowrite)
        print('Done.\nAlways make sure that the resulting facet list is zero indexed.\nDon\'t forget to change the saving file name so your file does not get written over.')

