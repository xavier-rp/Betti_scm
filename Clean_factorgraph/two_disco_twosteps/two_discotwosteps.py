from another_sign_int import *
from base import *
from factorgraph import *
from loglin_model import *
from metropolis_sampler import *
from script_sampler import *
from synth_data_analysis import *


def problist_to_table(prob_dist, sample_size):

    dimension = len(prob_dist) - 1
    table = np.random.rand(dimension)
    reshape = np.log(dimension)/np.log(2)
    table = np.reshape(table, np.repeat(2, reshape))

    for key in list(prob_dist.keys()):

        if key != 'T':

            table[key] = prob_dist[key]

    return table * sample_size

if __name__ == '__main__':

    """
    For this one, we're going to generate a two_factor, but we adjust the probabilities, so that both are independent
    """

    factorgraph = FactorGraph([[0, 1]])
    print(factorgraph.factor_list)
    probdist = Prob_dist(factorgraph)
    print(probdist.prob_dist)

    table_test = problist_to_table(probdist.prob_dist, 100)

    print(table_test)

    print(chisq_test(table_test, mle_2x2_ind(table_test)))

    with open('factorgraph_two_disco.pkl', 'wb') as output:
        pickle.dump(factorgraph, output, pickle.HIGHEST_PROTOCOL)

    for i in np.arange(0, 1000, 1):

        state = np.random.randint(2, size=2)
        print(state)

        energy_obj = Energy(state, factorgraph)
        proposer = BitFlipProposer(factorgraph, energy_obj, state)
        sampler = Sampler(proposer, temperature=1, initial_burn=10, sample_burn=11)
        sampler.sample(50)

        bipartite = build_bipartite(sampler.results['sample'])
        np.save('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/two_disco_twosteps/data_100/bipartite_' + str(i), bipartite)

    #exit()
    #for i in np.arange(0, 1000, 1):
    #    state = np.random.randint(2, size=2)
    #    print(state)

    #    energy_obj = Energy(state, factorgraph)
    #    proposer = BitFlipProposer(factorgraph, energy_obj, state)
    #    sampler = Sampler(proposer, temperature=1, initial_burn=100, sample_burn=51)
    #    sampler.sample(100)

    #    bipartite = build_bipartite(sampler.results['sample'])
    #    np.save('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/two_disco/data_100/bipartite_' + str(i),
    #            bipartite)

    #exit()

    #for i in np.arange(0, 1000, 1):
    #    state = np.random.randint(2, size=2)
    #    print(state)

    #    energy_obj = Energy(state, factorgraph)
    #    proposer = BitFlipProposer(factorgraph, energy_obj, state)
    #    sampler = Sampler(proposer, temperature=1, initial_burn=100, sample_burn=100)
    #    sampler.sample(10)

    #    bipartite = build_bipartite(sampler.results['sample'])
    #    np.save('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/two_disco/data_10/bipartite_' + str(i),
    #            bipartite)

    #exit()

    pval_list = []

    for i in np.arange(0, 1000, 1):

        bam = np.load('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/two_disco_twosteps/data_100/bipartite_' +str(i) + '.npy')

        analyser = Analyser(bam, '8_poster_asympt')

        pval = analyser.analyse_asymptotic(0.01)

        if i == 2:
            cont_table = get_cont_table(0, 1, bam)
            expected = mle_2x2_ind(cont_table)
            chis = chisq_test(cont_table, expected)
            print('Here 13', cont_table, expected, pval, chis)


        if pval < 0.5:
            cont_table = get_cont_table(0, 1, bam)
            expected = mle_2x2_ind(cont_table)
            chis = chisq_test(cont_table, expected)
            print('Here', cont_table, expected, pval, chis)

        pval_list.append(pval)
    compte = 0
    for pval in pval_list:
        if pval < 0.01:
            compte += 1
            print(pval)

    pval_list.sort()
    print(pval_list)
    print(min(pval_list))
    print('Compte = ', compte)

    np.save('pval_list_100', np.array(pval_list))

    pval_list = np.load('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/two_disco_twosteps/pval_list_100.npy')
    plt.figure(1)
    n, b, p = plt.hist(pval_list, bins=np.arange(0, 1.01, 0.01))
    plt.xlabel('p-value')
    plt.ylabel('count')
    plt.xlim(0, 1)
    plt.show()

    sum = np.array([[0.0, 0.0], [0.0, 0.0]])
    pval_trafiquee = []
    for i in range(1000):
        bam = np.load('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/two_disco_twosteps/data_100/bipartite_' +str(i) + '.npy')
        cont_table = get_cont_table(0, 1, bam)

        #if i%4 == 0:
        #    aggrandi_std = np.array([[-10.0, -10.0], [10.0, 10.0]])
        #elif i%4 == 1:
        #    aggrandi_std = np.array([[10.0, 10.0], [-10.0, -10.0]])
        #elif i % 4 == 2:
        #    aggrandi_std = np.array([[-10.0, 10.0], [10.0, -10.0]])
        #else:
        #    aggrandi_std = -1*np.array([[10.0, -10.0], [-10.0, 10.0]])

        aggrandi_std = 0

        cont_table = cont_table + aggrandi_std
        #print(np.min(cont_table))
        x, p = chisq_test(cont_table, mle_2x2_ind(cont_table))
        if np.min(cont_table) <= 16:
            print(cont_table)
        pval_trafiquee.append(p)

        sum += (cont_table + aggrandi_std)

    print(sum/(1000*100))

    plt.figure(1)
    n, b, p = plt.hist(pval_trafiquee, bins=np.arange(0, 1.01, 0.01))
    plt.xlabel('p-value')
    plt.ylabel('count')
    plt.xlim(0, 1)
    plt.show()

    std = np.array([[0.0, 0.0], [0.0, 0.0]])
    aggrandi_std = np.array([[-1.0, 1], [-1.0, 1.0]])
    for i in range(1000):
        bam = np.load(
            '/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/two_disco_twosteps/data_100/bipartite_' + str(
                i) + '.npy')

        cont_table = get_cont_table(0, 1, bam)

        #if i%4 == 0:
        #    aggrandi_std = np.array([[-10.0, -10.0], [10.0, 10.0]])
        #elif i%4 == 1:
        #    aggrandi_std = np.array([[10.0, 10.0], [-10.0, -10.0]])
        #elif i % 4 == 2:
        #    aggrandi_std = np.array([[-10.0, 10.0], [10.0, -10.0]])
        #else:
        #    aggrandi_std = -1*np.array([[10.0, -10.0], [-10.0, 10.0]])

        #if i % 2 == 0:
        #    aggrandi_std = np.array([[4.0, -4], [-4.0, 4.0]])
        #else:
        #    aggrandi_std = -1 * np.array([[4.0, -4], [-4.0, 4.0]])

        aggrandi_std = 0

        std += ((cont_table + aggrandi_std) - sum / 1000) ** 2

    print(np.sqrt(std / 1000))



