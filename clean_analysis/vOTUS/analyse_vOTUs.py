#from clean_analysis.exact_significative_interactions import *
#from clean_analysis.bird.compare_edge_list import *
from analyse_betti import compute_betti, compute_nb_simplices, build_simplex_tree, skeleton_to_graph
import matplotlib.pyplot as plt
import numpy as np
import csv
from tqdm import tqdm
from utilities.prune import *
import collections
import networkx as nx

def one_simplex_from_csv(csvfilename, alpha):
    one_simplex_list = []
    with open(csvfilename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in tqdm(reader):
            try :
                p = float(row[-1])
                if p < alpha:
                    one_simplex_list.append([int(row[0]), int(row[1]), p])
            except:
                pass

    return one_simplex_list

def two_simplex_from_csv(csvfilename, alpha):
    two_simplex_list = []
    with open(csvfilename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in tqdm(reader):
            try :
                p = float(row[-1])
                if p < alpha:
                    two_simplex_list.append([int(row[0]), int(row[1]), int(row[2]),  p])
            except:
                pass

    return two_simplex_list

def build_facet_list(matrix, two_simplices_file, one_simplices_file, alpha):
    #open('facet_list.txt', 'a').close()

    with open('facet_list.txt', 'w') as facetlist:

        nodelist = np.arange(0, matrix.shape[0])

        for node in nodelist:
            facetlist.write(str(node) +'\n')

        with open(one_simplices_file, 'r') as csvfile:

            reader = csv.reader(csvfile)
            next(reader)

            for row in tqdm(reader):

                try:
                    p = float(row[-1])
                    if p < alpha:
                        # Reject H_0 in which we suppose that u and v are independent
                        # Thus, we accept H_1 and add a link between u and v in the graph to show their dependency
                        facetlist.write(str(row[0]) + ' ' + str(row[1]) + '\n')
                except:
                    pass
        try:
            with open(two_simplices_file, 'r') as csvfile:

                reader = csv.reader(csvfile)
                next(reader)

                for row in tqdm(reader):

                    try:
                        p = float(row[-1])
                        if p < alpha:
                            # Reject H_0 in which we suppose that u and v are independent
                            # Thus, we accept H_1 and add a link between u and v in the graph to show their dependency
                            facetlist.write(str(row[0]) + ' ' + str(row[1]) + ' ' + str(row[2]) + '\n')
                    except:
                        pass
        except:
            pass
    return

def phi_coefficient_table(cont_tab):
    row_sums = np.sum(cont_tab, axis=1)
    col_sums = np.sum(cont_tab, axis=0)
    return (cont_tab[0,0]*cont_tab[1,1] - cont_tab[1,0]*cont_tab[0,1])/np.sqrt(row_sums[0]*row_sums[1]*col_sums[0]*col_sums[1])

def get_cont_table(u_idx, v_idx, matrix):
    # Computes the 2X2 contingency table for the occurrence matrix
    row_u_present = matrix[u_idx, :]
    row_v_present = matrix[v_idx, :]

    row_u_not_present = 1 - row_u_present
    row_v_not_present = 1 - row_v_present

    # u present, v present
    table00 = np.dot(row_u_present, row_v_present)

    # u present, v NOT present
    table01 = np.dot(row_u_present, row_v_not_present)

    # u NOT present, v present
    table10 = np.dot(row_u_not_present, row_v_present)

    # u NOT present, v NOT present
    table11 = np.dot(row_u_not_present, row_v_not_present)

    return np.array([[table00, table01], [table10, table11]])

def build_facet_list_with_phi(matrix, two_simplices_file, one_simplices_file, alpha, phi_pos=True):
    #open('facet_list.txt', 'a').close()

    with open('facet_list.txt', 'w') as facetlist:

        nodelist = np.arange(0, matrix.shape[0])

        for node in nodelist:
            facetlist.write(str(node) +'\n')

        with open(one_simplices_file, 'r') as csvfile:

            reader = csv.reader(csvfile)
            next(reader)

            for row in tqdm(reader):

                try:
                    p = float(row[-1])
                    ph = float(row[-2])
                    if phi_pos:
                        if p < alpha and ph >= 0 :
                            # Reject H_0 in which we suppose that u and v are independent
                            # Thus, we accept H_1 and add a link between u and v in the graph to show their dependency
                            facetlist.write(str(row[0]) + ' ' + str(row[1]) + '\n')
                    else:
                        if p < alpha and ph <= 0:
                            # Reject H_0 in which we suppose that u and v are independent
                            # Thus, we accept H_1 and add a link between u and v in the graph to show their dependency
                            facetlist.write(str(row[0]) + ' ' + str(row[1]) + '\n')
                except:
                    pass
        try:
            with open(two_simplices_file, 'r') as csvfile:

                reader = csv.reader(csvfile)
                next(reader)

                for row in tqdm(reader):
                    two_simp = [row[0], row[1], row[2]]

                    phi_list = []

                    for one_simp in itertools.combinations(two_simp, 2):
                        contingency_table = get_cont_table(int(one_simp[0]), int(one_simp[1]), matrix)
                        phi_list.append(phi_coefficient_table(contingency_table))

                    try:
                        p = float(row[-1])

                        if phi_pos and np.all( np.array(phi_list) >= 0):

                            if p < alpha:
                                # Reject H_0 in which we suppose that u and v are independent
                                # Thus, we accept H_1 and add a link between u and v in the graph to show their dependency
                                facetlist.write(str(row[0]) + ' ' + str(row[1]) + ' ' + str(row[2]) + '\n')
                        elif not phi_pos and np.all( np.array(phi_list) <= 0):
                            if p < alpha:
                                # Reject H_0 in which we suppose that u and v are independent
                                # Thus, we accept H_1 and add a link between u and v in the graph to show their dependency
                                facetlist.write(str(row[0]) + ' ' + str(row[1]) + ' ' + str(row[2]) + '\n')
                    except:
                        pass
        except:
            pass
    return

def build_network_from_one_simplice_file(matrix, one_simplices_file, alpha):
    #open('facet_list.txt', 'a').close()

    with open('facet_list.txt', 'w') as facetlist:

        nodelist = np.arange(0, matrix.shape[0])

        for node in nodelist:
            facetlist.write(str(node) +'\n')

        with open(one_simplices_file, 'r') as csvfile:

            reader = csv.reader(csvfile)
            next(reader)

            for row in tqdm(reader):

                try:
                    p = float(row[-1])
                    if p < alpha:
                        # Reject H_0 in which we suppose that u and v are independent
                        # Thus, we accept H_1 and add a link between u and v in the graph to show their dependency
                        facetlist.write(str(row[0]) + ' ' + str(row[1]) + '\n')
                except:
                    pass

    return

def prun(path_to_facet):
    import argparse as ap
    prs = ap.ArgumentParser(description='Prune facet list.')
    prs.add_argument('facet_list', type=str, nargs='?',
                     help='Path to facet list.')
    args = prs.parse_args()

    ### Instruction : Change path to facet list here :
    # args.facet_list = '/home/xavier/Documents/Projet/scm/utilities/severeadathr_finalOTU.txt'
    args.facet_list = path_to_facet
    facet_list = []
    with open(args.facet_list, 'r') as f:
        for line in f:
            # frozenset is mandatory: allows for comparison
            facet = frozenset([int(x) for x in line.strip().split()])
            facet_list.append(facet)
        stringtowrite = ''

        for facet in prune(facet_list):

            stringtowrite = stringtowrite + " ".join([str(v) for v in facet]) + '\n'

        with open('facet_list.txt', 'w') as outfile:
            outfile.write(stringtowrite)


    return


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

def read_pairwise_p_values_phi(filename, alpha=0.01, phi_pos=True):

    graph = nx.Graph()

    with open(filename, 'r') as csvfile:

        reader = csv.reader(csvfile)
        next(reader)

        for row in tqdm(reader):

            try:
                p = float(row[-1])
                phi = float(row[-2])
                if p < alpha and phi_pos and phi >= 0:
                    # Reject H_0 in which we suppose that u and v are independent
                    # Thus, we accept H_1 and add a link between u and v in the graph to show their dependency
                    graph.add_edge(int(row[0]), int(row[1]), phi=float(row[-2]), p_value=p)
                elif p < alpha and not phi_pos and phi <= 0:
                    # Reject H_0 in which we suppose that u and v are independent
                    # Thus, we accept H_1 and add a link between u and v in the graph to show their dependency
                    graph.add_edge(int(row[0]), int(row[1]), phi=float(row[-2]), p_value=p)
            except:
                pass

    return graph

def read_pairwise_p_values_phi_matrix(filename, matrix, alpha=0.01, phi_pos=True):

    graph = nx.Graph()

    graph.add_nodes_from(np.arange(0, matrix.shape[0]))

    with open(filename, 'r') as csvfile:

        reader = csv.reader(csvfile)
        next(reader)

        for row in tqdm(reader):

            try:
                p = float(row[-1])
                phi = float(row[-2])
                if p < alpha and phi_pos and phi >= 0:
                    # Reject H_0 in which we suppose that u and v are independent
                    # Thus, we accept H_1 and add a link between u and v in the graph to show their dependency
                    graph.add_edge(int(row[0]), int(row[1]), phi=float(row[-2]), p_value=p)
                elif p < alpha and not phi_pos and phi <= 0:
                    # Reject H_0 in which we suppose that u and v are independent
                    # Thus, we accept H_1 and add a link between u and v in the graph to show their dependency
                    graph.add_edge(int(row[0]), int(row[1]), phi=float(row[-2]), p_value=p)
            except:
                pass

    return graph

def to_facet_list():

    facetlist = []
    with open('facet_list.txt', 'r') as file:
        for l in file:
            facetlist.append([int(x) for x in l.strip().split()])

    return facetlist

def countTriangle(g):
    dictio = nx.triangles(g)
    somme = np.sum(dictio.keys())
    return somme/6

def meandegree(g):
    totaldeg = 0
    for node in list(g.nodes) :
        totaldeg += g.degree[node]

    return totaldeg/len(g)

def find_components(g):
    comp_set = set()
    for node in list(g.nodes) :
        comp = nx.node_connected_component(g, node)
        print(comp)

    return comp_set



if __name__ == '__main__':

    matrix1 = np.load(r'/home/xavier/Documents/Projet/Betti_scm/clean_analysis/vOTUS_occ.npy').T

    virus_path = r'/home/xavier/Documents/Projet/Betti_scm/clean_analysis/vOTUS/VIRUS_exact_pvalues.csv'
    matrice = np.load(r'/home/xavier/Documents/Projet/Betti_scm/clean_analysis/vOTUS_occ.npy')

    # Chemins vers les p-values pour les liens et les 2-simplexes des lacs asymptotique puis exact
    as_lakes_2_pvalues = r'/home/xavier/Documents/Projet/Betti_scm/clean_analysis/vOTUS/vOTUS_asymptotic_pvalues.csv'
    as_lakes_3_pvalues = r'/home/xavier/Documents/Projet/Betti_scm/clean_analysis/vOTUS/vOTUS_asymptotic_cube_pvalues.csv'

    as7d_lakes_3_pvalues = r'/home/xavier/Documents/Projet/Betti_scm/clean_analysis/vOTUS_7degrees/vOTUS_7degrees_asymptotic_cube_pvalues.csv'

    as3d_lakes_2_pvalues = r'/home/xavier/Documents/Projet/Betti_scm/clean_analysis/vOTUS/vOTUS_asymptotic3degrees_pvalues.csv'
    as3d_lakes_3_pvalues = r'/home/xavier/Documents/Projet/Betti_scm/clean_analysis/vOTUS/vOTUS_asymptotic3degrees_cube_pvalues.csv'


    ex_lakes_2_pvalues = r'/home/xavier/Documents/Projet/Betti_scm/clean_analysis/vOTUS/vOTUS_exact_pvalues.csv'
    ex_lakes_3_pvalues = r'/home/xavier/Documents/Projet/Betti_scm/clean_analysis/vOTUS/vOTUS_exact_cube_pvalues.csv'

    ex10k_lakes_3_pvalues = r'/home/xavier/Documents/Projet/Betti_scm/clean_analysis/vOTUS/vOTUS_exact_cube_pvalues_10000samples.csv'
    ex10m_lakes_3_pvalues = r'/home/xavier/Documents/Projet/Betti_scm/clean_analysis/vOTUS/vOTUS_exact_cube_pvalues_10000000samples.csv'

    #build_facet_list_with_phi(matrix1, as_lakes_3_pvalues, as_lakes_2_pvalues, 0.01, phi_pos=False)
    #build_facet_list_with_phi(matrix1, ex_lakes_3_pvalues, ex_lakes_2_pvalues, 0.0001, phi_pos=False)

    #build_facet_list(matrix1, as_lakes_3_pvalues, as_lakes_2_pvalues, 0.01)

    #prun('facet_list.txt')

    #facetlist = to_facet_list()

    #st = build_simplex_tree(facetlist, 2)

    #g = skeleton_to_graph(st)

    #plt.imshow(nx.to_numpy_matrix(g))

    #plt.show()

    #g = read_pairwise_p_values_phi(as_lakes_2_pvalues, 0.01, phi_pos=False)

    #plt.imshow(nx.to_numpy_matrix(g))

    #plt.show()

    #g = read_pairwise_p_values(as_lakes_2_pvalues, 0.01)

    #plt.imshow(nx.to_numpy_matrix(g))

    #plt.show()



    #exit()

    # Figures en fonction de alpha (Nombre de liens, nombre de 2-simplexes, nombre de composantes, nombres de Betti)


    ## Nombre de 2-simplexes avec les deux méthodes

    for alpha in np.logspace(-8, -1.3):

        plt.scatter(alpha, len(two_simplex_from_csv(ex10k_lakes_3_pvalues, alpha)), c='b')
        plt.scatter(alpha, len(two_simplex_from_csv(ex10m_lakes_3_pvalues, alpha)), c='k', marker='+')
        #plt.scatter(alpha, len(two_simplex_from_csv(as3d_lakes_3_pvalues, alpha)), c='k', marker='+')
        #plt.scatter(alpha, len(two_simplex_from_csv(as_lakes_3_pvalues, alpha)), c='b')
        plt.scatter(alpha, len(two_simplex_from_csv(ex_lakes_3_pvalues, alpha)), marker='x', c='r')
        plt.scatter(alpha, len(two_simplex_from_csv(as7d_lakes_3_pvalues, alpha)), c='g', marker='*')

    plt.scatter(alpha, len(two_simplex_from_csv(as7d_lakes_3_pvalues, alpha)), c='g', marker='*', label='Aympt 7d')
    plt.scatter(alpha, len(two_simplex_from_csv(ex10k_lakes_3_pvalues, alpha)), c='b', label='10k')
    plt.scatter(alpha, len(two_simplex_from_csv(ex10m_lakes_3_pvalues, alpha)), c='k', marker='+', label='10m')
    plt.scatter(alpha, len(two_simplex_from_csv(ex_lakes_3_pvalues, alpha)), c='r', marker='x', label='1m')
    #plt.scatter(alpha, len(two_simplex_from_csv(as_lakes_3_pvalues, alpha)), c='b', label='Asymptotic 1 degree')
    #plt.scatter(alpha, len(two_simplex_from_csv(as3d_lakes_3_pvalues, alpha)), c='k', marker='+', label='Asymptotic 3 degrees')
    #plt.scatter(alpha, len(two_simplex_from_csv(ex_lakes_3_pvalues, alpha)), c='r', marker='x', label='Exact 3 degrees')
    plt.legend(loc=0)
    plt.xlabel('alpha')
    plt.ylabel('Nb 2-simplices')
    plt.title('Nombre de 2-simplexes pour les deux méthodes selon alpha')
    plt.show()


    ## Nombre de 1-simplexes avec les deux méthodes

    #for alpha in np.logspace(-20, -1.3):

    #    plt.scatter(alpha, len(one_simplex_from_csv(as_lakes_2_pvalues, alpha)), c='b')
    #    plt.scatter(alpha, len(one_simplex_from_csv(ex_lakes_2_pvalues, alpha)), c='r', marker='x')

    #plt.scatter(alpha, len(one_simplex_from_csv(as_lakes_2_pvalues, alpha)), c='b', label='Asymptotic')
    #plt.scatter(alpha, len(one_simplex_from_csv(ex_lakes_2_pvalues, alpha)), c='r', marker='x', label='Exact')
    #plt.xlabel('alpha')
    #plt.ylabel('Nb 1-simplices')
    #plt.title('Nombre de 1-simplexes pour les deux méthodes selon alpha')
    #plt.legend(loc=0)
    #plt.show()

    ### Nombre de composantes pour les deux méthodes selon Alpha

    #for alpha in np.logspace(-20, -1.3):
    #    build_facet_list(matrix1, as_lakes_3_pvalues, as_lakes_2_pvalues, alpha)
    #    prun('facet_list.txt')
    #    facetlist = to_facet_list()
    #    st = build_simplex_tree(facetlist, 2)

    #    bettis = compute_betti(facetlist, 3)

    #    plt.scatter(alpha, bettis[0], c='b')

    #    build_facet_list(matrix1, ex_lakes_3_pvalues, ex_lakes_2_pvalues, alpha)
    #    prun('facet_list.txt')
    #    facetlist = to_facet_list()
    #    st = build_simplex_tree(facetlist, 2)

    #    bettis1 = compute_betti(facetlist, 3)

    #    plt.scatter(alpha, bettis1[0], c='r', marker='x')

    #plt.scatter(alpha, bettis[0], c='b', label='Asymptotic')
    #plt.scatter(alpha, bettis1[0], c='r', marker='x', label='Exact')
    #plt.xlabel('alpha')
    #plt.ylabel('Nb of Betti0')
    #plt.title('Nombre de composantes les deux méthodes selon alpha')
    #plt.legend(loc=0)
    #plt.show()

    ## Betti 1

    #for alpha in np.logspace(-20, -1.3):
    #    build_facet_list(matrix1, as_lakes_3_pvalues, as_lakes_2_pvalues, alpha)
    #    prun('facet_list.txt')
    #    facetlist = to_facet_list()
    #    st = build_simplex_tree(facetlist, 2)

    #    bettis = compute_betti(facetlist, 3)

    #    plt.scatter(alpha, bettis[1], c='b')

    #    build_facet_list(matrix1, ex_lakes_3_pvalues, ex_lakes_2_pvalues, alpha)
    #    prun('facet_list.txt')
    #    facetlist = to_facet_list()
    #    st = build_simplex_tree(facetlist, 2)

    #    bettis1 = compute_betti(facetlist, 3)

    #    plt.scatter(alpha, bettis1[1], c='r', marker='x')

    #plt.scatter(alpha, bettis[1], c='b', label='Asymptotic')
    #plt.scatter(alpha, bettis1[1], c='r', marker='x', label='Exact')
    #plt.xlabel('alpha')
    #plt.ylabel('Nb of Betti1')
    #plt.title('Betti_1 pour les deux méthodes selon alpha')
    #plt.legend(loc=0)
    #plt.show()

    ## Betti 2
    #for alpha in np.logspace(-20, -1.3):
    #    build_facet_list(matrix1, as_lakes_3_pvalues, as_lakes_2_pvalues, alpha)
    #    prun('facet_list.txt')
    #    facetlist = to_facet_list()
    #    st = build_simplex_tree(facetlist, 2)

    #    bettis = compute_betti(facetlist, 3)

    #    plt.scatter(alpha, bettis[2], c='b')

    #    build_facet_list(matrix1, ex_lakes_3_pvalues, ex_lakes_2_pvalues, alpha)
    #    prun('facet_list.txt')
    #    facetlist = to_facet_list()
    #    st = build_simplex_tree(facetlist, 2)

    #    bettis1 = compute_betti(facetlist, 3)

    #    plt.scatter(alpha, bettis1[2], c='r', marker='x')

    #plt.scatter(alpha, bettis[2], c='b', label='Asymptotic')
    #plt.scatter(alpha, bettis1[2], c='r', marker='x', label='Exact')
    #plt.xlabel('alpha')
    #plt.title('Betti_2 pour les deux méthodes selon alpha')
    #plt.ylabel('Nb of Betti2')
    #plt.legend(loc=0)
    #plt.show()

    ## Nombre de 1-simplexes si nous considérons les 2-simplexes vs juste en construisant le réseau.

    #for alpha in np.logspace(-20, -1.3):
    #    build_facet_list(matrix1, as_lakes_3_pvalues, as_lakes_2_pvalues, alpha)
    #    prun('facet_list.txt')
    #    facetlist = to_facet_list()
    #    st = build_simplex_tree(facetlist, 2)

    #    nb_simplices = compute_nb_simplices(st, 1, andlower=False)
    #    plt.scatter(alpha, nb_simplices, c='b')

    #    plt.scatter(alpha, len(one_simplex_from_csv(as_lakes_2_pvalues, alpha)), c='k', marker='x')


    #plt.scatter(alpha, nb_simplices, c='b', label='Complexe simplicial')
    #plt.scatter(alpha, len(one_simplex_from_csv(as_lakes_2_pvalues, alpha)), c='k', marker='x', label='liens_csv')
    #plt.xlabel('alpha')
    #plt.ylabel('nb liens')
    #plt.title('Nombre de liens')
    #plt.legend(loc=0)
    #plt.show()


    ## Nombre de composantes si on considère le réseau VS le complexe simplicial

    #for alpha in np.logspace(-20, -1.3):
    #    build_network_from_one_simplice_file(matrix1, as_lakes_2_pvalues, alpha)
    #    prun('facet_list.txt')
    #    facetlist = to_facet_list()
    #    st = build_simplex_tree(facetlist, 2)

    #    bettis = compute_betti(facetlist, 3)

    #    plt.scatter(alpha, bettis[0], c='b')

    #    build_facet_list(matrix1, ex_lakes_3_pvalues, ex_lakes_2_pvalues, alpha)
    #    prun('facet_list.txt')
    #    facetlist = to_facet_list()
    #    st = build_simplex_tree(facetlist, 2)

    #    bettis1 = compute_betti(facetlist, 3)

    #    plt.scatter(alpha, bettis1[0], c='r', marker='x')

    #plt.scatter(alpha, bettis[0], c='b', label='Complexe simplicial')
    #plt.scatter(alpha,  bettis1[0], c='r', marker='x', label='réseau')
    #plt.xlabel('alpha')
    #plt.ylabel('nb de composantes')
    #plt.title('Nombre de composantes')
    #plt.legend(loc=0)
    #plt.show()

    ## Pour un alpha donné, Nb noeuds lien 2-simplexes Betti distribution des degrés communautés, filtrations sur PHI

    #exit()

    #for alpha in [0.1]:
    #    g = read_pairwise_p_values(virus_path, alpha)
    #    plt.scatter(alpha, nx.number_connected_components(g))

    #plt.xlabel('alpha')
    #plt.ylabel('nb de composantes')
    #plt.title('Nombre de composantes en fonction de alpha')
    #plt.legend(loc=0)
    #plt.show()




    alpha = 0.001

    #g = read_pairwise_p_values(virus_path, alpha)

    #g = read_pairwise_p_values_phi(virus_path, alpha)

    g = read_pairwise_p_values_phi_matrix(virus_path, matrice, alpha)

    plt.imshow(nx.to_numpy_matrix(g))

    np.save('Virus_pvalue_001', nx.to_numpy_matrix(g))

    plt.show()

    print('Number of nodes : ', g.number_of_nodes())
    print('Number of links : ', g.number_of_edges())
    print('Mean degree : ', meandegree(g))
    #print('Number of triangles : ', countTriangle(g))
    print('Number of connected components : ', nx.number_connected_components(g))
    for connected_comp in nx.connected_components(g):
        print(len(connected_comp))
        print(connected_comp)

    degree_sequence = sorted([d for n, d in g.degree()], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

    plt.show()


    exit()

    build_facet_list(matrice, None, virus_path, alpha)
    prun('facet_list.txt')
    facetlist = to_facet_list()
    st = build_simplex_tree(facetlist, 2)

    nb_simplices = compute_nb_simplices(st, 3)


