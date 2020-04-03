from another_sign_int import *
from base import *
from factorgraph import *
from loglin_model import *
from metropolis_sampler import *
from script_sampler import *
from synth_data_analysis import *
from sympy.solvers import solve
from sympy import Symbol
from matplotlib import rc

def problist_to_table(prob_dist, sample_size):

    dimension = len(prob_dist) - 1
    table = np.random.rand(dimension)
    reshape = np.log(dimension)/np.log(2)
    table = np.reshape(table, np.repeat(2, reshape))

    for key in list(prob_dist.keys()):

        if key != 'T':

            table[key] = prob_dist[key]

    #table = np.roll(table, (1, 1), (0, 1))

    return table * sample_size

def distance_to_original(bam, original):

    sampled_table = get_cont_cube(1, 2, 0, bam)



    return np.sum(np.abs((sampled_table - original)))

    #return np.nan_to_num(np.sum((sampled_table - original/N_original*N)**2/(original/N_original*N)))

def mle_multinomial_from_table(cont_table):
    n = np.sum(cont_table)
    p_list = []
    for element in cont_table.flatten():
        p_list.append(element/n)

    return p_list

def multinomial_problist_cont_cube(nb_trials, prob_list, s=1):
    return np.random.multinomial(nb_trials, prob_list, s).reshape(s, 2, 2, 2)

def sampled_chisq_test(cont_table, expected_table, sampled_array):
    if float(0) in expected_table:
        test_stat = 0
        pval = 1
    else:
        test_stat = np.sum((cont_table - expected_table) ** 2 / expected_table)
        cdf = np.sum((sampled_array < test_stat) * 1) / len(sampled_array)
        pval = 1 - cdf
    return test_stat, pval

def chisq_formula_vector_for_cubes(cont_tables, expected):
    # Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
    # via MLE or iterative proportional fitting.

    return np.nan_to_num(np.sum(np.sum(np.sum((cont_tables - expected) ** 2 / expected, axis = 1), axis = 1), axis=1))

def chisq_test_here(cont_tab, expected, df=1):
    #Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
    #via MLE or iterative proportional fitting.
    if np.any(expected == 0):
        print('HERE')
        return 0, 1
    #df = 7
    test_stat = np.sum((cont_tab-expected)**2/expected)
    p_val = chi2.sf(test_stat, df)

    return test_stat, p_val

if __name__ == '__main__':


    #factorgraph = FactorGraph([[0, 1, 2]], N=400, alpha=0.0001)
    #probdist = Prob_dist(factorgraph)

    with open('factorgraph_three-dep_triangle_to_simplex.pkl', 'wb') as output:
        pickle.dump(factorgraph, output, pickle.HIGHEST_PROTOCOL)

    #for i in np.arange(0, 1000, 1):
    #    factorgraph = FactorGraph([[0, 1, 2]], N=400, alpha=0.001)
    #    state = np.random.randint(2, size=3)
    #    print(state)

    #    energy_obj = Energy(state, factorgraph)
    #    proposer = BitFlipProposer(factorgraph, energy_obj, state)
    #    sampler = Sampler(proposer, temperature=1, initial_burn=5, sample_burn=5)
    #    sampler.sample(400)
    #    print(sampler.results['nb_rejected'])
    #    print(sampler.results['nb_success'])

    #    bipartite = build_bipartite(sampler.results['sample'])
    #    np.save(
    #        '/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/Simplicial_complex_3dep/data_001/bipartite_' + str(
    #            i), bipartite)

    #exit()

    list_of_pval_list = []
    pval_list = []
    distance_list = []
    link_count = []
    for i in np.arange(0, 1000, 1):

        if i % 1000 == 0 and i != 0:
            list_of_pval_list.append(copy.deepcopy(pval_list))
            pval_list = []

        bam = np.load(
            '/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/Simplicial_complex_3dep/data_001/bipartite_' + str(
                i) + '.npy')
        # print(bam.shape)

        cont_table = get_cont_cube(1, 2, 0, bam)

        exp = iterative_proportional_fitting_AB_AC_BC_no_zeros(cont_table)

        pval = chisq_test_here(cont_table, exp, df=1)[1]


        analyser = Analyser(bam,
                            '/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/three_dep_triangle_to_simplex_clean/factorgraph_three-ind')
        if analyser.analyse_asymptotic_for_triangle(0.01) == 3:
            print('3 links to 2-simplex', cont_table, pval)
        link_count.append(analyser.analyse_asymptotic_for_triangle(0.05))


        pval_list.append(pval)

    list_of_pval_list.append(pval_list)


    print(len(np.where(np.array(link_count) == 0)[0]), len(np.where(np.array(link_count) == 1)[0]),
          len(np.where(np.array(link_count) == 2)[0]), len(np.where(np.array(link_count) == 3)[0]))

    list_of_pval_list.append(copy.deepcopy(pval_list))
    list_of_compte_list = []
    # print(list_of_pval_list)
    # exit()
    print(np.amax(np.array(list_of_pval_list)))
    plt.plot(np.arange(0, 0.101, 0.001), np.arange(0, 0.101, 0.001), ls='--', color='#00a1ffff', label=r'$y = \alpha$')
    for pval_list in list_of_pval_list:
        # print(pval_list)
        comptelist = []
        for alpha in np.arange(0, 0.1001, 0.001):
            compte = 0
            for pval in pval_list:
                if pval > alpha:
                    compte += 1
                    # print(pval)
            comptelist.append(compte)
        # print(comptelist)
        list_of_compte_list.append(np.array(comptelist) / 1000)
        # plt.plot(np.arange(0, 0.101, 0.001), np.array(comptelist)/1000)

    plt.plot(np.arange(0, 0.101, 0.001), np.mean(np.array(list_of_compte_list), axis=0), color='#ff7f00ff',
             linewidth='2', label='Proportion de rejet')
    plt.fill_between(np.arange(0, 0.101, 0.001),
                     np.mean(np.array(list_of_compte_list), axis=0) + np.std(np.array(list_of_compte_list), axis=0),
                     np.mean(np.array(list_of_compte_list), axis=0) - np.std(np.array(list_of_compte_list), axis=0),
                     color='#ff7f00ff', alpha=0.2)
    plt.legend(loc=0)
    plt.ylabel(r"Proportion d\textquotesingle erreur de type $1$")
    plt.xlabel(r'$\alpha$')
    plt.show()

    pval_list = []
    for i in np.arange(0, 10000, 1):

        bam = np.load(
            '/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/three_dep_triangle_to_simplex_clean/data_100_2/bipartite_' + str(
                i) + '.npy')

        analyser = Analyser(bam,
                            '/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/three_dep_triangle_to_simplex_clean/factorgraph_three-ind')
        if analyser.analyse_asymptotic_for_triangle(0.01) == 3:
            cont_table = get_cont_cube(1, 2, 0, bam)

            exp = iterative_proportional_fitting_AB_AC_BC_no_zeros(cont_table)

            pval = chisq_test_here(cont_table, exp, df=1)[1]

            pval_list.append(pval)

    print(pval_list)

    # exit()

    # plt.plot(np.arange(0, 0.101, 0.001), np.arange(0, 0.101, 0.001), ls='--', color='#00a1ffff', label=r'$y = \alpha$')
    comptelist = []
    for alpha in np.arange(0.001, 0.1001, 0.001):
        compte = 0
        for pval in pval_list:
            if pval < alpha:
                compte += 1
                print(pval)
        comptelist.append(compte)
    print(comptelist)
    comptelist = np.array(comptelist) / 5763
    # plt.plot(np.arange(0, 0.101, 0.001), np.array(comptelist)/1000)
    rc('text', usetex=True)
    rc('font', size=16)
    plt.plot(np.arange(0.001, 0.101, 0.001), comptelist, color='#ff7f00ff',
             linewidth='2', label='Proportion de rejet')

    plt.legend(loc=0)
    plt.ylabel(r"Proportion d\textquotesingle erreur de type $2$")
    plt.xlabel(r'$\alpha$')
    plt.show()
    exit()

    exit()
    comptelist = []
    for alpha in np.arange(0, 0.1001, 0.001):
        compte = 0
        for pval in pval_list:
            if pval < alpha:
                compte += 1
                print(pval)
        comptelist.append(compte)
    print(comptelist)
    plt.plot(np.arange(0, 0.101, 0.001), np.arange(0, 0.101, 0.001))
    plt.plot(np.arange(0, 0.101, 0.001), np.array(comptelist) / 1000)

    plt.show()
    compte = 0
    for pval in pval_list:
        if pval < 0.01:
            compte += 1
            print(pval)
    print('Compte under 0.01 : ', compte)

    distance_list.sort()
    print(distance_list)

    plt.figure(1)
    n, b, p = plt.hist(distance_list, bins=99)
    plt.xlabel('distance')
    plt.ylabel('count')
    # plt.xlim(0, 1)
    plt.show()

    # pval_list.sort()
    # print(pval_list)
    # print(min(pval_list))
    # print('Compte = ', compte)

    np.save('pval_list_1000', np.array(pval_list))
    print(pval_list)
    # exit()
    # pval_list = np.load('pval_list_100.npy')
    plt.figure(1)
    n, b, p = plt.hist(pval_list, bins=np.arange(0, 1, 0.01))
    plt.xlabel('p-value')
    plt.ylabel('count')
    plt.xlim(0, 1)
    plt.show()

    sum = np.array([[0.0, 0.0], [0.0, 0.0]])
    for i in range(1000):
        bam = np.load(
            '/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/Two_co_clean/data_1000/bipartite_' + str(
                i) + '.npy')

        cont_table = get_cont_table(0, 1, bam)

        print(cont_table)

        sum += cont_table

    print(sum / (1000 * 100))
