"""
Base class for the implementation of a metropolis scheme for
the generation of of co-occurrence matrices. This code is based
on the architecture of Zoo2019
"""

import networkx as nx
import numpy as np
from copy import deepcopy

"""
Je dois commencer en générant un graph de manière aléatoire. Ensuite, je dois attribuer une interaction particulière
à chacun des liens (la forme de l'intéraction et son poids genre wij = 9 et f = (xi-xj)^2 ou wij = -2 et f = (2xi+xj)^2.


TODO Définir d'autres facteurs, des méthodes d'attribution de poids
        implémenter une manière de désigner des simplexes de dimension 2 et plus ainsi qu'un facteur

Avec ce réseau et ces fonction d'énergie, je dois initilialiser un état sur chacun des noeuds donc une série de 0 et de 1
et calculer l'énergie.

Ensuite je propose un changement aléatoire (0 vers 1 ou 1 vers 0) et je recalcule l'énergie, puis le sampler va s'occuper
du reste
"""

class FactorGraph(nx.Graph):
    """Original graph that encodes all the interactions."""

    def __init__(self, G=nx.Graph):
        """__init__
        :param G: Networkx graph
        """

        super(FactorGraph, self).__init__(G)
        self.set_factors()
        self.weights = []


    def set_factors(self):

        weight = -1

        for node in self.nodes:

            edges_of_node = self.edges(node)

            if len(edges_of_node) > 0:

                factor_list = []

                for edge in edges_of_node:

                    factor_list.append((list(edge), weight, self.twofactor_difference_squared))

                self.nodes[node]['list'] = factor_list

            else :

                self.nodes[node]['list'] = [([node], weight, self.onefactor_state)]

    def twofactor_difference_squared(self, node1_state, node2_state, weight):

        return weight * (node1_state - node2_state) ** 2

    def onefactor_state(self, node1_state, weight):

        return weight * node1_state


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

        network_copy = deepcopy(self.factorgraph)

        energy = 0

        for node_info in self.factorgraph.nodes(data='list'):

            for interaction in node_info[1]:

                state_list = []
                if len(interaction[0]) > 1:
                    if interaction[0] in network_copy.edges:
                        for interacting_node in interaction[0]:

                            state_list.append(self.current_state[interacting_node])


                        energy += interaction[-1](*state_list, interaction[1])
                else:
                    energy += interaction[-1](self.current_state[interaction[0][0]], interaction[1])

            network_copy.remove_node(node_info[0])


        return energy

    def get_local_energy(self, targeted_node):

        info = self.factorgraph.nodes[targeted_node]['list']
        energy = 0

        for interaction in info:

            state_list = []
            if len(interaction[0]) > 1:
                for interacting_node in interaction[0]:
                    state_list.append(self.current_state[interacting_node])

                energy += interaction[-1](*state_list, interaction[1])
            else:
                energy += interaction[-1](self.current_state[interaction[0][0]], interaction[1])

        return energy




    def __call__(self, target_node_idx):
        """__call__ returns the local energy of a node in the perturbed graph
        at a certain time index

        Note that the local energy for a node i at a time index "j" corresponds
        to -log(P(X_i(t_{j+1}|X_i(t_j)).

        Therefore, for a timeseries len(X) = N, there are (N-1) time index
        for the local energy

        :param node: a node of the graph
        :param index: time index
        :param g1: perturbed_graph instance
        """

        local_energy = 0
        for interaction in self.factorgraph.node[target_node_idx]['list']:
            energy += interaction[-1](interaction[0], interaction[1])
        raise NotImplementedError("__call__ must be overwritten")

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

        energy_diff = current_local - flipped_local

        return self.total_energy + energy_diff

    def get_local_energy(self, targeted_node):

        info = self.factorgraph.nodes[targeted_node]['list']
        local_energy = 0

        for interaction in info:

            state_list = []
            if len(interaction[0]) > 1:
                for interacting_node in interaction[0]:
                    state_list.append(self.state[interacting_node])

                local_energy += interaction[-1](*state_list, interaction[1])
            else:
                local_energy += interaction[-1](self.state[interaction[0][0]], interaction[1])

        return local_energy

    def rejected(self):
        """rejected reverse applied changes"""

        self.state[self.targeted_node] = abs(self.state[self.targeted_node] - 1)

    def get_state(self):
        """get_state returns the change point and removed edges set"""
        return deepcopy(self.state)

if __name__ == '__main__':

    current = [1, 0, 1]

    G = nx.generators.random_graphs.erdos_renyi_graph(3,1,seed=0)

    FG = FactorGraph(G)

    FG.set_factors()

    lte = Energy(current, FG)

    print(lte.local_energy(0))

    print(lte.local_energy(1))

    print(lte.local_energy(2))

    print(lte.total_energy())

    print(FG.nodes)

    print(FG.nodes(data='list'))

