import subprocess as sp
import networkx as nx
import time
import json

"""
Author: Jean-Gabriel Young <info@jgyoung.ca>
Author: Xavier Roy-Pomerleau <xavier.roy-pomerleau.1@ulaval.ca>

See '' main '' section

This module is used to generate instances of the SCM from a given facet list. Each instances can be saved with the
same name, but a different index.

Make sure to read this before launching the code :

There are several lines to update each time this code is used. To find which parameters have to be tuned, search
(CTRL+F) for ''Instruction'' (without the quotation marks) or ''###''.

"""

def rejection_sampling(command, seed=0):
    # Call sampler with subprocess
    proc = sp.run(command, stdout=sp.PIPE)
    # Read output as a facet list
    facet_list = []
    for line in proc.stdout.decode().split("\n")[1:-1]:
        #print('line ', line)
        if line.find("#") == 0:
            yield facet_list
            facet_list = []
        else:
            facet_list.append([int(x) for x in line.strip().split()])
    yield facet_list

def facet_list_to_bipartite(facet_list):
    """Convert a facet list to a bipartite graph"""
    g = nx.Graph()
    for f, facet in enumerate(facet_list):
        g.add_node('f'+str(f), bipartite = 1)
        for v in facet:
            g.add_node('v'+ str(v), bipartite = 0)
            g.add_edge('v' + str(v), 'f' + str(f))  # differentiate node types
    return g

def facet_list_to_graph(facet_list):
    """Convert a facet list to a bipartite graph"""
    # ON peut convertir le graphe bipartite en autre graphe en effectuant une multiplication par la transposée
    g = nx.Graph()
    for f, facet in enumerate(facet_list):
        for v in facet:
            g.add_edge('v' + str(v), 'f' + str(f))  # differentiate node types
    return g




if __name__ == '__main__':
    """
    This module is used to generate instances of the SCM from a given facet list. Each instances can be saved with the
    same name, but a different index.

    Make sure to read this before launching the code :

    There are several lines to update each time this code is used. To find which parameters have to be tuned, search
    (CTRL+F) for ''Instruction'' (without the quotation marks) or ''###''.

    """

    print('Don\'t forget to read the documentation at the beginning of the main!')

    start = time.time()

    ### Instruction : Change the path to the facet list and the parameters in the command. See JG-Young GitHub of the
    ### SCM for more information.

    #command = ['bin/mcmc_sampler', 'datasets/prunedfacetlist_subotu.txt', '-t', '1', '--seed=0', '-c', '-b', '500', '-f', '500']
    #command = ['bin/mcmc_sampler', '/home/xavier/Documents/Projet/scm/datasets/crime_facet_list.txt', '-t', '1000', '--exp_prop','--prop_param', '1', '-c']
    command = ['bin/mcmc_sampler', '/home/xavier/Documents/Projet/scm/datasets/facet_list_c1_as_simplices.txt', '-t', '4', '-c']

    ### Instruction : First index used to name the files. If i = 500, saved instances will have the name blabla500
    ### blabla501... and so on.
    i = 1


    for facet_list in rejection_sampling(command):
        print(i, facet_list)


        ### Instruction : Change the path to the saved files.

        with open('/home/xavier/Documents/Projet/scm/crime/alice1_crime_instance' + str(i)+'.json', 'w') as outfile:
            json.dump(facet_list, outfile)
        i += 1
    print('Done.\n Don\'t forget to change the saving file name so your file does not get written over.')

    print('time : ', time.time()-start)