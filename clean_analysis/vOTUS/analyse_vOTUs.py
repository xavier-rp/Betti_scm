#from clean_analysis.exact_significative_interactions import *
#from clean_analysis.bird.compare_edge_list import *
from analyse_betti import compute_betti, compute_nb_simplices, build_simplex_tree
import matplotlib.pyplot as plt
import numpy as np
import csv
from tqdm import tqdm
from utilities.prune import *

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

def to_facet_list():

    facetlist = []
    with open('facet_list.txt', 'r') as file:
        for l in file:
            facetlist.append([int(x) for x in l.strip().split()])

    return facetlist


if __name__ == '__main__':

    matrix1 = np.load('/home/xavier/Documents/Projet/Betti_scm/clean_analysis/vOTUS_occ.npy').T

    # Chemins vers les p-values pour les liens et les 2-simplexes des lacs asymptotique puis exact
    as_lakes_2_pvalues = r'/home/xavier/Documents/Projet/Betti_scm/clean_analysis/vOTUS/vOTUS_asymptotic_pvalues.csv'
    as_lakes_3_pvalues = r'/home/xavier/Documents/Projet/Betti_scm/clean_analysis/vOTUS/vOTUS_asymptotic_cube_pvalues.csv'

    ex_lakes_2_pvalues = r'/home/xavier/Documents/Projet/Betti_scm/clean_analysis/vOTUS/vOTUS_exact_pvalues.csv'
    ex_lakes_3_pvalues = r'/home/xavier/Documents/Projet/Betti_scm/clean_analysis/vOTUS/vOTUS_exact_cube_pvalues.csv'

    # Figures en fonction de alpha (Nombre de liens, nombre de 2-simplexes, nombre de composantes, nombres de Betti)


    # Nombre de 2-simplexes avec les deux méthodes

    for alpha in np.logspace(-8, -1.3):

        plt.scatter(alpha, len(two_simplex_from_csv(as_lakes_3_pvalues, alpha)), c='b')
        plt.scatter(alpha, len(two_simplex_from_csv(ex_lakes_3_pvalues, alpha)), marker='x', c='r')

    plt.scatter(alpha, len(two_simplex_from_csv(as_lakes_3_pvalues, alpha)), c='b', label='Asymptotic')
    plt.scatter(alpha, len(two_simplex_from_csv(ex_lakes_3_pvalues, alpha)), c='r', marker='x', label='Exact')
    plt.legend(loc=0)
    plt.xlabel('alpha')
    plt.ylabel('Nb 2-simplices')
    plt.title('Nombre de 2-simplexes pour les deux méthodes selon alpha')
    plt.show()


    # Nombre de 1-simplexes avec les deux méthodes

    for alpha in np.logspace(-20, -1.3):

        plt.scatter(alpha, len(one_simplex_from_csv(as_lakes_2_pvalues, alpha)), c='b')
        plt.scatter(alpha, len(one_simplex_from_csv(ex_lakes_2_pvalues, alpha)), c='r', marker='x')

    plt.scatter(alpha, len(one_simplex_from_csv(as_lakes_2_pvalues, alpha)), c='b', label='Asymptotic')
    plt.scatter(alpha, len(one_simplex_from_csv(ex_lakes_2_pvalues, alpha)), c='r', marker='x', label='Exact')
    plt.xlabel('alpha')
    plt.ylabel('Nb 1-simplices')
    plt.title('Nombre de 1-simplexes pour les deux méthodes selon alpha')
    plt.legend(loc=0)
    plt.show()

    ## Nombre de composantes pour les deux méthodes selon Alpha

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
    #plt.ylabel('Nb of Betti2')
    #plt.legend(loc=0)
    #plt.show()

    # Nombre de 1-simplexes si nous considérons les 2-simplexes vs juste en construisant le réseau.

    for alpha in np.logspace(-20, -1.3):
        build_facet_list(matrix1, as_lakes_3_pvalues, as_lakes_2_pvalues, alpha)
        prun('facet_list.txt')
        facetlist = to_facet_list()
        st = build_simplex_tree(facetlist, 2)

        nb_simplices = compute_nb_simplices(st, 3)

        plt.scatter(alpha, nb_simplices[1], c='b')

        plt.scatter(alpha, len(one_simplex_from_csv(as_lakes_2_pvalues, alpha)), c='k', marker='x')


    plt.scatter(alpha, nb_simplices[1], c='b', label='Complexe simplicial')
    plt.scatter(alpha, len(one_simplex_from_csv(as_lakes_2_pvalues, alpha)), c='k', marker='x', label='liens_csv')
    plt.xlabel('alpha')
    plt.ylabel('nb liens')
    plt.title('Nombre de liens')
    plt.legend(loc=0)
    plt.show()


    # Noeuds isolés au départ :

    for alpha in np.logspace(-20, -1.3):
        build_network_from_one_simplice_file(matrix1, as_lakes_2_pvalues, alpha)
        prun('facet_list.txt')
        facetlist = to_facet_list()
        st = build_simplex_tree(facetlist, 2)

        bettis = compute_betti(facetlist, 3)

        nb_simplices = compute_nb_simplices(st, 3)

        plt.scatter(alpha, nb_simplices[0], c='b')

        build_facet_list(matrix1, ex_lakes_3_pvalues, ex_lakes_2_pvalues, alpha)
        prun('facet_list.txt')
        facetlist = to_facet_list()
        st = build_simplex_tree(facetlist, 2)

        bettis1 = compute_betti(facetlist, 3)

        plt.scatter(alpha, bettis1[0], c='r', marker='x')

        plt.scatter(alpha, len(one_simplex_from_csv(as_lakes_2_pvalues, alpha)), c='k', marker='x')

    plt.scatter(alpha, nb_simplices[1], c='b', label='Complexe simplicial')
    plt.scatter(alpha, len(one_simplex_from_csv(as_lakes_2_pvalues, alpha)), c='k', marker='x', label='liens_csv')
    plt.xlabel('alpha')
    plt.ylabel('nb liens')
    plt.title('Nombre de liens')
    plt.legend(loc=0)
    plt.show()






    # Pour un alpha donné, Nb noeuds lien 2-simplexes Betti distribution des degrés communautés, filtrations sur PHI


