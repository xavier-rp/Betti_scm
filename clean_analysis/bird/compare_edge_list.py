import networkx as nx
import numpy as np
import pandas as pd
from loglin_model import *
from Exact_chi_square_1_deg import *


def compare_edge_list(asymptotic_file, exact_file):
    G = nx.read_edgelist(asymptotic_file)

    g = nx.read_edgelist(exact_file)

    print('Finding edges that are in exact but not in asymptotic')
    for edge in g.edges:

        if edge not in G.edges:
            print(edge)

    print('Done')


def compare_two_simplices_list(asymptotic_file, exact_file):


    asymp = pd.read_csv(asymptotic_file)
    exact = pd.read_csv(exact_file)

    cols = [0,1,2]

    asymp_idx = asymp[asymp.columns[cols]]
    exact_idx = exact[exact.columns[cols]]


    print('Two-simplices in exact that are not in asympt')


    out = pd.concat([exact_idx, asymp_idx, asymp_idx]).drop_duplicates(keep=False)
    print(out)
    print('_____________________________________________')

    print('Two-simplices in asympt that are not in exact')

    out2 = pd.concat([asymp_idx, exact_idx, exact_idx]).drop_duplicates(keep=False)

    print(out2)
    print('_____________________________________________')

    i1 = exact_idx.set_index(exact_idx.columns.tolist()).index
    i2 = asymp_idx.set_index(asymp_idx.columns.tolist()).index

    print('Exact is subset of asymptotic : ', i1.isin(i2).all())

    return

def get_cont_cube(u_idx, v_idx, w_idx, matrix):
    # Computes the 2X2X2 contingency table for the occurrence matrix

    row_u_present = matrix[u_idx, :]
    row_v_present = matrix[v_idx, :]
    row_w_present = matrix[w_idx, :]

    row_u_not = 1 - row_u_present
    row_v_not = 1 - row_v_present
    row_w_not = 1 - row_w_present

    #All present :
    table000 =np.sum(row_u_present*row_v_present*row_w_present)

    # v absent
    table010 = np.sum(row_u_present*row_v_not*row_w_present)

    # u absent
    table100 = np.sum(row_u_not*row_v_present*row_w_present)

    # u absent, v absent
    table110 = np.sum(row_u_not*row_v_not*row_w_present)

    # w absent
    table001 = np.sum(row_u_present*row_v_present*row_w_not)

    # v absent, w absent
    table011 = np.sum(row_u_present*row_v_not*row_w_not)

    # u absent, w absent
    table101 = np.sum(row_u_not*row_v_present*row_w_not)

    # all absent
    table111 = np.sum(row_u_not*row_v_not*row_w_not)

    return np.array([[[table000, table010], [table100, table110]], [[table001, table011], [table101, table111]]], dtype=np.float64)





if __name__ == '__main__':
    #matrix1 = np.load(r'D:\Users\Xavier\Documents\Analysis_master\Analysis\clean_analysis\vOTUS_occ.npy').T
    #matrix1 = matrix1.astype(np.int64)

    #row0 = matrix1[0]
    #row3 = matrix1[3]
    #row7 = matrix1[7]
    #nb_samples = 1000000

    #cont_cube = get_cont_cube(0, 3, 7, matrix1)

    #table_str = str(int(cont_cube[0, 0, 0])) + '_' + str(int(cont_cube[0, 0, 1])) + '_' + str(
    #    int(cont_cube[0, 1, 0])) + '_' + str(int(cont_cube[0, 1, 1])) + '_' + str(int(cont_cube[1, 0, 0])) + '_' + str(
    #    int(cont_cube[1, 0, 1])) + '_' + str(int(cont_cube[1, 1, 0])) + '_' + str(int(cont_cube[1, 1, 1]))




    #table_id = table_str
    #table = np.random.rand(2, 2, 2)
    #table_id_list = str.split(table_id, '_')
    #table[0, 0, 0] = int(table_id_list[0])
    #table[0, 0, 1] = int(table_id_list[1])
    #table[0, 1, 0] = int(table_id_list[2])
    #table[0, 1, 1] = int(table_id_list[3])
    #table[1, 0, 0] = int(table_id_list[4])
    #table[1, 0, 1] = int(table_id_list[5])
    #table[1, 1, 0] = int(table_id_list[6])
    #table[1, 1, 1] = int(table_id_list[7])

    #N = np.sum(table)
    #expected_original = iterative_proportional_fitting_AB_AC_BC_no_zeros(table)

    #if expected_original is not None:
    #    problist = mle_multinomial_from_table(expected_original)
    #    sample = multinomial_problist_cont_cube(N, problist, nb_samples)
    #    expec = np.tile(expected_original, (nb_samples, 1, 1)).reshape(nb_samples, 2, 2, 2)
    #    chisq = chisq_formula_vector_for_cubes(sample, expec)
    #    s, p = sampled_chisq_test(table, expected_original, chisq)
    #    print(s, p)

    #exit()



    compare_two_simplices_list(r'D:\Users\Xavier\Documents\Analysis_master\Analysis\clean_analysis\vOTUS\vOTUS_asymptotic_two_simplices_01.csv', r'D:\Users\Xavier\Documents\Analysis_master\Analysis\clean_analysis\vOTUS\vOTUS_exact_two_simplices_01.csv')




    asymptotic_file_path = r"D:\Users\Xavier\Documents\Analysis_master\Analysis\clean_analysis\vOTUS\vOTUS_asymptotic_edge_list_01.txt"
    exact_file_path = r"D:\Users\Xavier\Documents\Analysis_master\Analysis\clean_analysis\vOTUS\vOTUS_exact_edge_list_01.txt"

    compare_edge_list(asymptotic_file_path, exact_file_path)

