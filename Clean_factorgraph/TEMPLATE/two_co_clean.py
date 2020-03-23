from another_sign_int import *
from base import *
from factorgraph import *
from loglin_model import *
from metropolis_sampler import *
from script_sampler import *
from synth_data_analysis import *
from sympy.solvers import solve
from sympy import Symbol

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

    return np.nan_to_num(np.sum((sampled_table - original/N_original*N)**2/(original/N_original*N)))

if __name__ == '__main__':

    """
    For this one, we're going to generate a two_factor, but we adjust the probabilities, so that both are independent
    """

    #[[12. 25.]
    # [25. 38.]]

    factorgraph = FactorGraph([[0, 1]])
    print(factorgraph.factor_list)
    probdist = Prob_dist(factorgraph)
    print(probdist.prob_dist)

    #a = 12/100
    #b = 25/100
    #c = 25/100
    #d = 38/100

    #x = Symbol('x')
    #y = Symbol('y')
    #z = Symbol('z')
    #w = Symbol('w')
    #print(solve([(a-1)*x + a*y + a*z + a*w, b*x + (b-1)*y + b*z + b*w, c*x + c*y + (c-1)*z + c*w, d*x + d*y + d*z + (d-1)*w]))

    #table_test = problist_to_table(probdist.prob_dist, 1000)

    #print(table_test)
    #mle = mle_2x2_ind(table_test)
    #table_test = np.array([[38.7455619,  23.50037122], [23.50037122, 14.25369566]])
    #print(np.sum((table_test - mle)**2/(mle)))
    #print(chisq_test(table_test, mle_2x2_ind(table_test)))

    #exit()

    #exit()

    #with open('factorgraph_two_disco.pkl', 'wb') as output:
    #    pickle.dump(factorgraph, output, pickle.HIGHEST_PROTOCOL)

    #for i in np.arange(0, 1000, 1):

    #    state = np.random.randint(2, size=2)
    #    print(state)

    #    energy_obj = Energy(state, factorgraph)
    #    proposer = BitFlipProposer(factorgraph, energy_obj, state)
    #    sampler = Sampler(proposer, temperature=1, initial_burn=14, sample_burn=15)
    #    sampler.sample(100)
    #    print(sampler.results['nb_rejected'])
    #    print(sampler.results['nb_success'] )

    #    bipartite = build_bipartite(sampler.results['sample'])
    #    np.save('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/Two_co_clean/data_100/bipartite_' + str(i), bipartite)

    #exit()
    #for i in np.arange(0, 1000, 1):
    #    state = np.random.randint(2, size=2)
    #    print(state)

    #    energy_obj = Energy(state, factorgraph)
    #    proposer = BitFlipProposer(factorgraph, energy_obj, state)
    #    sampler = Sampler(proposer, temperature=1, initial_burn=25, sample_burn=25)
    #    sampler.sample(1000)

    #    bipartite = build_bipartite(sampler.results['sample'])
    #    np.save('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/two_disco_pval_fiftypercent/data_1000/bipartite_' + str(i),
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
    distance_list = []

    for i in np.arange(0, 1000, 1):

        bam = np.load('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/Two_co_clean/data_100/bipartite_' +str(i) + '.npy')

        #distance_list.append(distance_to_original(bam, table_test))

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
    compte = 0
    for pval in pval_list:
        if pval < 0.01:
            compte += 1
            print(pval)

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

    np.save('pval_list_100', np.array(pval_list))
    print(pval_list)
    #exit()
    #pval_list = np.load('pval_list_100.npy')
    plt.figure(1)
    n, b, p = plt.hist(pval_list, bins=np.arange(0, 1, 0.01))
    print(n)
    print(b)
    print(p)
    plt.xlabel('p-value')
    plt.ylabel('count')
    plt.xlim(0, 1)
    plt.show()

    sum = np.array([[0.0, 0.0], [0.0, 0.0]])
    for i in range(1000):

        bam = np.load('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/Two_co_clean/data_100/bipartite_' +str(i) + '.npy')

        cont_table = get_cont_table(0, 1, bam)

        print(cont_table)

        sum += cont_table

    print(sum/(1000*100))

