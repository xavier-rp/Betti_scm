import numpy as np
import csv
from tqdm import tqdm
import networkx as nx
import json


def read_pairwise_p_values(filename, alpha=0.01):

    graph = nx.Graph()

    with open(filename, 'r') as csvfile:

        reader = csv.reader(csvfile)
        next(reader)

        for row in tqdm(reader):

            try:
                p = float(row[-1])
                if p < alpha:
                    # Reject H_0 in which we suppose that u and v are independent
                    # Thus, we accept H_1 and add a link between u and v in the graph to show their dependency
                    graph.add_edge(int(row[0]), int(row[1]), phi=float(row[-2]), p_value=p)
            except:
                pass


    return graph


if __name__ == '__main__':

    alpha = 0.001
    path_to_csv = 'bird/bird_exact_pvalues.csv'

    g = read_pairwise_p_values(path_to_csv, alpha)

    print('Number of nodes : ', g.number_of_nodes())
    print('Number of links : ', g.number_of_edges())

    g = read_pairwise_p_values(path_to_csv, alpha)

    node_dictio_list = []
    for noeud in g.nodes:
       node_dictio_list.append({"id": str(noeud)})

    link_dictio_list = []
    for lien in g.edges:
       link_dictio_list.append({"source": str(lien[0]), "target": str(lien[1]), "value": 1})


    json_diction = {"nodes": node_dictio_list, "links" : link_dictio_list}
    with open('bird/d3js_network_' + str(alpha)[2:] +'.json', 'w') as outfile:
       json.dump(json_diction, outfile)
    exit()