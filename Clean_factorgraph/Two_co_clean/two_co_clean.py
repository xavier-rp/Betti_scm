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

    table = np.roll(table, (1, 1), (0, 1))

    return table * sample_size

def distance_to_original(bam, original):

    sampled_table = get_cont_table(0, 1, bam)
    N = np.sum(sampled_table)
    sampled_table = sampled_table

    N_original = np.sum(original)

    print(N_original)

    return np.nan_to_num(np.sum(np.abs((sampled_table - original))))

    #return np.nan_to_num(np.sum((sampled_table - original/N_original*N)**2/(original/N_original*N)))

if __name__ == '__main__':

    """
    For this one, we're going to generate a two_factor, but we adjust the probabilities, so that both are independent
    """

    ind_25 = np.load('dep_48_2_2_48.npy')
    rc('text', usetex=True)
    rc('font', size=16)

    plt.plot(np.arange(0, 2 * len(ind_25), 2), ind_25, marker='x', markeredgecolor='black', markerfacecolor='black',
             markersize='7', color='#00a1ffff', linewidth='3', label='N = 100')

    plt.legend(loc=0)
    plt.xlabel('Distance $L_1$')
    plt.ylabel('Taux de succ\`es')
    # plt.grid()
    plt.xlim([0, 96])
    #plt.ylim([0.5, 1.05])
    plt.show()

    factorgraph = FactorGraph([[0, 1]])
    #print(factorgraph.factor_list)
    probdist = Prob_dist(factorgraph)
    print(probdist.prob_dist)

    table_test = problist_to_table(probdist.prob_dist, 1000)

    print(table_test)
    #exit()
    a = 49 / 100
    b = 1 / 100
    c = 1 / 100
    d = 49 / 100

    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    w = Symbol('w')
    print(solve(
        [(a - 1) * x + a * y + a * z + a * w, b * x + (b - 1) * y + b * z + b * w, c * x + c * y + (c - 1) * z + c * w,
         d * x + d * y + d * z + (d - 1) * w]))

    table_test = problist_to_table(probdist.prob_dist, 100)
    table_test = np.array([[48, 2], [2, 48]])
    #exit()

    #mle = mle_2x2_ind(table_test)
    #print(np.sum((table_test - mle)**2/(mle)))
    #print(chisq_test(table_test, mle_2x2_ind(table_test)))

    #exit()

    #with open('factorgraph_two_disco.pkl', 'wb') as output:
    #    pickle.dump(factorgraph, output, pickle.HIGHEST_PROTOCOL)

    #for i in np.arange(0, 10000, 1):

    #    state = np.random.randint(2, size=2)
    #    print(state)

    #    energy_obj = Energy(state, factorgraph)
    #    proposer = BitFlipProposer(factorgraph, energy_obj, state)
    #    sampler = Sampler(proposer, temperature=1, initial_burn=5, sample_burn=5)
    #    sampler.sample(100)
    #    print(sampler.results['nb_rejected'])
    #    print(sampler.results['nb_success'] )

    #    bipartite = build_bipartite(sampler.results['sample'])
    #    np.save('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/Two_co_clean/data_test_100/bipartite_' + str(i), bipartite)

    #exit()

    list_of_pval_list = []
    pval_list = []
    distance_list = []

    for i in np.arange(0, 10000, 1):

        if i % 1000 == 0 and i != 0:
            list_of_pval_list.append(copy.deepcopy(pval_list))
            pval_list = []

        bam = np.load('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/Two_co_clean/data_test_100/bipartite_' +str(i) + '.npy')
        #print(bam.shape)
        distance_list.append(distance_to_original(bam, table_test))

        cont_table = get_cont_table(0, 1, bam)

        print(cont_table)

        analyser = Analyser(bam, '8_poster_asympt')

        pval = analyser.analyse_asymptotic(0.01)

        #if i == 2:
        #    cont_table = get_cont_table(0, 1, bam)
        #    expected = mle_2x2_ind(cont_table)
        #    chis = chisq_test(cont_table, expected)
        #    print('Here 13', cont_table, expected, pval, chis)


        #if pval < 0.01:
        #    cont_table = get_cont_table(0, 1, bam)
        #    expected = mle_2x2_ind(cont_table)
        #    chis = chisq_test(cont_table, expected)
        #    print('Here', cont_table, expected, pval, chis)

        pval_list.append(pval)
    list_of_pval_list.append(pval_list)
    distance_list.sort()
    print(distance_list)
    rc('text', usetex=True)
    rc('font', size=16)
    plt.figure(1)
    n, b, p = plt.hist(distance_list, bins=np.arange(0, 50,1)-0.5, color='#00a1ffff')
    plt.xlabel('Distance $L_1$')
    plt.ylabel('Nombre de tables')
    plt.xlim(0, 50)
    plt.show()

    list_of_pval_list.append(copy.deepcopy(pval_list))
    list_of_compte_list = []
    print(list_of_pval_list)
    # exit()

    plt.plot(np.arange(0, 0.101, 0.001), np.arange(0, 0.101, 0.001), ls='--', color='#00a1ffff', label=r'$y = \alpha$')
    for pval_list in list_of_pval_list:
        print(pval_list)
        comptelist = []
        for alpha in np.arange(0, 0.1001, 0.001):
            compte = 0
            for pval in pval_list:
                if pval < alpha:
                    compte += 1
                    # print(pval)
            comptelist.append(compte)
        print(comptelist)
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
    #plt.xlim(0, 1)
    plt.show()

    #pval_list.sort()
    #print(pval_list)
    #print(min(pval_list))
    #print('Compte = ', compte)

    np.save('pval_list_1000', np.array(pval_list))
    print(pval_list)
    #exit()
    #pval_list = np.load('pval_list_100.npy')
    plt.figure(1)
    n, b, p = plt.hist(pval_list, bins=np.arange(0, 1, 0.01))
    plt.xlabel('p-value')
    plt.ylabel('count')
    plt.xlim(0, 1)
    plt.show()

    sum = np.array([[0.0, 0.0], [0.0, 0.0]])
    for i in range(1000):

        bam = np.load('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/Two_co_clean/data_1000/bipartite_' +str(i) + '.npy')

        cont_table = get_cont_table(0, 1, bam)

        print(cont_table)

        sum += cont_table

    print(sum/(1000*100))

