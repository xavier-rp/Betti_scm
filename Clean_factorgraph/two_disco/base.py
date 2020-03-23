"""
Base class for the implementation of a metropolis scheme for
the generation of of co-occurrence matrices. This code is based
on the architecture of Zoo2019
"""

import networkx as nx
import numpy as np
import itertools
#import gudhi
from copy import deepcopy

class FactorGraph():
    """Original graph that encodes all the interactions."""

    def __init__(self, facet_list=[]):
        """__init__
        :param facet_list: (list of lists) Each list in the list is a combination of integers
                            representing the nodes of a facet in a simplicial complex.
        """

        self.facet_list = facet_list
        self._get_node_list()
        self.set_factors()
        #self._get_simplicial_complex()

    #def _get_simplicial_complex(self):

    #    st = gudhi.SimplexTree()
    #    for facet in self.factor_list:
    #        st.insert(facet)

    #    self.st = st

    def _get_skeleton(self, j=1):

        skeleton_facet_list = []

        for facet in self.facet_list:

            if len(facet) > j + 1 :

                for lower_simplex in itertools.combinations(facet, j + 1):

                    skeleton_facet_list.append(list(lower_simplex))

            else :
                skeleton_facet_list.append(facet)

        return skeleton_facet_list

    def _get_st_skeleton(self, j=1):

        return self.st.get_skeleton(j)


    def _get_node_list(self):

        node_set = set()

        for facet in self.facet_list:

            for node in facet:

                node_set.add(node)

        self.node_list = list(node_set)
        self.node_list.sort()


    def set_factors(self):

        weight_list = []

        factor_list = []

        for facet in self.facet_list:

            if len(facet) == 1:

                weight_list.append(-1)

                factor_list.append(self.onefactor_state)

            elif len(facet) == 2:

                weight_list.append(-1)

                #factor_list.append(self.twofactor_difference_squared)

                #factor_list.append(self.twofactor_sum_squared)

                #factor_list.append(self.twofactor_sum_squared_sym)

                factor_list.append(self.twofactor_table_entry_pos)

                #if np.random.rand(1)[0] > 0.5:
                #    factor_list.append(self.twofactor_table_entry)
                #else:
                #    factor_list.append(self.twofactor_table_entry_pos)

            elif len(facet) == 3:

                weight_list.append(-1)

                #factor_list.append(self.threefactor_pairsum_plus)
                #factor_list.append(self.threefactor_sym)
                #factor_list.append(self.threefactor_pairsum_sym_plus)

                #factor_list.append(self.threefactor_all_terms_sym)
                #factor_list.append(self.threefactor_all_terms_sym_with_diff_weights)
                factor_list.append(self.threefactor_table_entry)

            else :

                print('Interactions with more than three nodes are not yet coded.')

        self.factor_list = factor_list
        self.weight_list = weight_list

    def set_weight_list(self):

        #TODO

        return


        #TODO : It seems that in most case, adding : /(1 + sum(node_states)) at the end a a factor flattens the
        #probability distribution and should be encouraged I guess.
    def threefactor_pairsum_plus(self, node_states, weight):
        x1 = node_states[0]
        x2 = node_states[1]
        x3 = node_states[2]
        return weight * ((x1 + x2) * (x1 + x3) * (x2 + x3) + x1 + x2 + x3)/(1 + sum(node_states))

        #return weight * ((x1 + x2) * (x1 + x3) * (x2 + x3) + x1 + x2 + x3)

    def threefactor_pairsum_sym_plus(self, node_states, weight):

        x1 = node_states[0]
        x2 = node_states[1]
        x3 = node_states[2]
        return weight * ((x1 + x2) * (x1 + x3) * (x2 + x3) + x1 + x2 + x3 + ((1 - x1) + (1 - x2)) * ((1 - x1) + (1 - x3)) * ((1 - x2) + (1 - x3)) + (1-x1) + (1 - x2) + (1 - x3)) / 3

    def threefactor_sym(self, node_states, weight):
        x1 = node_states[0]
        x2 = node_states[1]
        x3 = node_states[2]
        return weight * ((x1*x2*x3) + (1-x1)*(1-x2)*(1-x3))

        #return weight * ((x1 + x2) * (x1 + x3) * (x2 + x3) + x1 + x2 + x3)

    def threefactor_all_terms_sym(self, node_states, weight):
        x1 = node_states[0]
        x2 = node_states[1]
        x3 = node_states[2]
        return weight * ((x1*x2*x3) + (1-x1)*(1-x2)*(1-x3) + x1*x2 + x1*x3 + x2*x3 + (1-x1)*(1-x2) + (1-x1)*(1-x3) + (1-x2)*(1-x3) + x1 + x2 + x3 + (1-x1) + (1-x2) + (1-x3))

    def threefactor_all_terms_sym_with_diff_weights(self, node_states, weight, a=2, b=1, c=1.2, d=1, e=1, f=1, g=1.7):
        x1 = node_states[0]
        x2 = node_states[1]
        x3 = node_states[2]
        return weight * (a*(x1*x2*x3) + (1-x1)*(1-x2)*(1-x3) + b*x1*x2 + c*x1*x3 + d*x2*x3 + (1-x1)*(1-x2) + (1-x1)*(1-x3) + (1-x2)*(1-x3) + e*x1 + f*x2 + g*x3 + (1-x1) + (1-x2) + (1-x3))

    def threefactor_table_entry(self, node_states, weight, a=2, b=1.3, c=1.2, d=1, e=0.7, f=0.9, g=0.8, h=0.3):
        x1 = node_states[0]
        x2 = node_states[1]
        x3 = node_states[2]
        return weight * (a*(x1*x2*x3) + b*x1*x2*(1-x3) + c*x1*(1-x2)*x3 + d*(1-x1)*x2*x3 + e*x1*(1-x2)*(1-x3)
                         + f*(1-x1)*x2*(1-x3) + g*(1-x1)*(1-x2)*x3 + h*(1-x1)*(1-x2)*(1-x3))


    def twofactor_difference_squared(self, node_states, weight):

        return weight * (node_states[0] - node_states[1]) ** 2

    def twofactor_sum_squared(self, node_states, weight):

        return weight * (node_states[0] + node_states[1])**2

    def twofactor_sum_squared_sym(self, node_states, weight):

        return weight *((node_states[0] + node_states[1])**2 + ((1-node_states[0]) + (1-node_states[1]))**2)

    def twofactor_table_entry(self, node_states, weight, a=1, b=1, c=1, d=1):

        x1 = node_states[0]
        x2 = node_states[1]

        return weight * (a*x1*x2 + b*(1-x1)*x2 + c*x1*(1-x2) + d*(1-x1)*(1-x2))

    def twofactor_table_entry_pos(self, node_states, weight, a=1, b=1, c=1, d=1):

        x1 = node_states[0]
        x2 = node_states[1]

        return weight * (a*x1*x2 + b*(1-x1)*x2 + c*x1*(1-x2) + d*(1-x1)*(1-x2))


    def onefactor_state(self, node_states, weight):

        return weight * node_states[0]

class Prob_dist():

    def __init__(self, factorgraph, temperature=1):

        self.temperature = temperature

        if factorgraph is not None :
            self.fg = factorgraph
            self._get_Z()
            self._get_prob_dist()

        else:
            pass


    def _get_Z(self):

        self.energy_per_state = {}

        self.Z = 0

        for state in itertools.product(range(2), repeat=len(self.fg.node_list)):

            state_energy = Energy(state, self.fg).get_total_energy()

            self.energy_per_state[state] = state_energy

            self.Z += np.exp(-(1/self.temperature)*state_energy)

    def _get_prob_dist(self):

        prob_dist = {}

        for state in itertools.product(range(2), repeat=len(self.fg.node_list)):

            prob_dist[state] = np.exp(-(1/self.temperature)*self.energy_per_state[state])/self.Z

        prob_dist['T'] = self.temperature

        self.prob_dist = prob_dist



class Energy():
    """Base class that returns the total energy of the state or the local energy of a node at a certain time"""
    def __init__(self, current_state, factorgraph):
        """__init__

        :param current_state: Array of 0 and 1 denoting the current presence and absence state of all nodes
        :param factor graph: FactorGraph object encoding all relations between the nodes
        """
        self.current_state = current_state
        self.factorgraph = factorgraph
        self.total_energy = self.get_total_energy()

    def get_total_energy(self):

        energy = 0

        facet_idx = 0

        for facet in self.factorgraph.facet_list:

            node_states = []

            for node_idx in facet:

                node_states.append(self.current_state[node_idx])

            energy += self.factorgraph.factor_list[facet_idx](node_states, self.factorgraph.weight_list[facet_idx])

            facet_idx += 1

        return energy

    def get_local_energy(self, targeted_node):

        energy = 0

        involved_facets_idx = []

        i = 0

        for facet in self.factorgraph.facet_list:

            if targeted_node in facet:

                involved_facets_idx.append(i)

            i += 1


        for facet_idx in involved_facets_idx :

            node_states = []

            for node_idx in self.factorgraph.facet_list[facet_idx]:

                node_states.append(self.current_state[node_idx])

            energy += self.factorgraph.factor_list[facet_idx](node_states, self.factorgraph.weight_list[facet_idx])

        return energy

class Proposer():
    """Propose new states for the system and give the energy variation"""
    def __init__(self):
        return

    def __call__(self):
        raise NotImplementedError("__call__ must be overwritten")

    def rejected(self):
        raise NotImplementedError("rejected must be overwritten")

    def get_state(self):
        raise NotImplementedError("get_state must be overwritten")


class BitFlipProposer(Proposer):
    """Propose a new perturbed_graph"""
    def __init__(self, fg, local_transition_energy, state):
        """__init__

        :param g: initial perturbed_graph
        :param local_transition_energy: LocalTransitionEnergy object
        :param prior_energy: PriorEnergy object
        :param p: probability for the geometric series
        """
        super(BitFlipProposer, self).__init__()
        self.state = state
        self.factorgraph = fg
        self.lte = local_transition_energy
        self.total_energy = self.lte.get_total_energy()


    def __call__(self):
        """__call__ propose a new perturbed graph. Returns the energy
        difference

        :returns likelihood_var, bias_var: tuple of energy variation for
        the proposal

        """
        return self._propose_bit_flip()


    def copy(self):
        """copy__ returns a copy of the object

        :returns pgp: PerturbedGraphProposer object with identitical set of
        parameters
        """
        state = self.state
        g = deepcopy(self.g)
        lte = self.lte
        return BitFlipProposer(g, lte,state)


    def _propose_bit_flip(self):
        """_propose_change_point propose a new change point

        :returns likelihood_var, bias_var: tuple of energy variation for
        the proposal
        """

        self.targeted_node = np.random.randint(0, len(self.state))

        current_local = self.get_local_energy(self.targeted_node)

        self.state[self.targeted_node] = abs(self.state[self.targeted_node] - 1)

        flipped_local = self.get_local_energy(self.targeted_node)

        energy_diff = flipped_local - current_local

        return self.total_energy + energy_diff

    def get_local_energy(self, targeted_node):


        energy = 0

        involved_facets_idx = []

        i = 0

        for facet in self.factorgraph.facet_list:

            if targeted_node in facet:

                involved_facets_idx.append(i)

            i += 1


        for facet_idx in involved_facets_idx :

            node_states = []

            for node_idx in self.factorgraph.facet_list[facet_idx]:

                node_states.append(self.state[node_idx])

            energy += self.factorgraph.factor_list[facet_idx](node_states, self.factorgraph.weight_list[facet_idx])

        return energy

    def rejected(self):
        """rejected reverse applied changes"""

        self.state[self.targeted_node] = abs(self.state[self.targeted_node] - 1)

    def get_state(self):
        """get_state returns the change point and removed edges set"""
        return deepcopy(self.state)

if __name__ == '__main__':

    st = gudhi.SimplexTree()
    if st.insert([0, 1]):
        print("[0, 1] inserted")
    if st.insert([0, 1, 2]):
        print("[0, 1, 2] inserted")
    if st.find([0, 1]):
        print("[0, 1] found")
    #result_str = 'num_vertices=' + repr(st.num_vertices())
    #print(result_str)
    #result_str = 'num_simplices=' + repr(st.num_simplices())
    #print(result_str)
    #print("skeleton(2) =")
    #for sk_value in st.get_skeleton(2):
    #    print(sk_value)

    print(st.get_skeleton(1))

    exit()


    current = [1, 1, 1]

    FG = FactorGraph([[0, 1, 2]])

    print(Prob_dist(FG).Z)
    print(Prob_dist(FG).prob_dist)

    lte = Energy(current, FG)

    print(lte.get_local_energy(0))

    print(lte.get_local_energy(1))

    print(lte.get_total_energy()/(1 + sum(current)))

    print(FG.factor_list)

    exit()






    current = [0, 0, 0]

    #FG = FactorGraph([[0,1], [1, 2]])

    current = [1, 0]

    FG = FactorGraph([[0, 1]])
    print(Prob_dist(FG).Z)
    print(Prob_dist(FG).prob_dist)

    lte = Energy(current, FG)

    print(lte.get_local_energy(0))

    print(lte.get_local_energy(1))

    print(lte.get_local_energy(2))

    print(lte.get_total_energy())

    print(FG.factor_list)



