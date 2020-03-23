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

    factorgraph = FactorGraph([[0, 1]])
    print(factorgraph.factor_list)

    probdist = Prob_dist(factorgraph)
    print(probdist.prob_dist)
    table_test = problist_to_table(probdist.prob_dist, 100)

    print(table_test)

    print(chisq_test(table_test, mle_2x2_ind(table_test)))

    with open('factorgraph_two_disco.pkl', 'wb') as output:
        pickle.dump(factorgraph, output, pickle.HIGHEST_PROTOCOL)

    #for i in np.arange(0, 1000, 1):

    #    state = np.random.randint(2, size=2)
    #    print(state)

    #    energy_obj = Energy(state, factorgraph)
    #    proposer = BitFlipProposer(factorgraph, energy_obj, state)
    #    sampler = Sampler(proposer, temperature=1, initial_burn=2000, sample_burn=500)
    #    sampler.sample(1000)

    #    bipartite = build_bipartite(sampler.results['sample'])
    #    np.save('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/two_disco/data/bipartite_' + str(i), bipartite)

    for i in np.arange(0, 100, 1):
        state = np.random.randint(2, size=2)
        print(state)

        energy_obj = Energy(state, factorgraph)
        proposer = BitFlipProposer(factorgraph, energy_obj, state)
        sampler = Sampler(proposer, temperature=1, initial_burn=50, sample_burn=51)
        sampler.sample(10)

        bipartite = build_bipartite(sampler.results['sample'])

        print('Number success : ', sampler.results['nb_success'])
        print('Number rejected : ', sampler.results['nb_rejected'])
        np.save('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/two_co/Testspace/bipartite_' + str(i),
                bipartite)

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

    for i in np.arange(0, 100, 1):

        bam = np.load('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/two_co/Testspace/bipartite_' +str(i) + '.npy')

        analyser = Analyser(bam, '8_poster_asympt')

        pval = analyser.analyse_exact(0.01)

        #if i == 13:
        #    cont_table = get_cont_table(0, 1, bam)
        #    expected = mle_2x2_ind(cont_table)
        #    chis = chisq_test(cont_table, expected)
        #    print('Here 13', cont_table, expected, pval, chis)


        if  pval > 0.01 :
            cont_table = get_cont_table(0, 1, bam)
            expected = mle_2x2_ind(cont_table)
            chis = chisq_test(cont_table, expected)
            print('Here', cont_table, expected, pval, chis)

        pval_list.append(pval)
    compte = 0
    for pval in pval_list:
        if pval < 0.01 :
            compte += 1
            print(pval)
    print('Compte = ', compte)
    print('Min = ', min(pval_list))

    print(pval_list)

    np.save('pval_list_asympt', np.array(pval_list))

