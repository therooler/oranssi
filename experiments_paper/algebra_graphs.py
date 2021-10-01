import networkx as nx
import numpy as np
import itertools as it
from oranssi.pauli import Pauli, PauliMonomial
from oranssi.opt_tools import AlgebraSU4
import matplotlib.pyplot as plt

def SU4_to_SU8_algebra():
    algebra = AlgebraSU4(3)
    algebra.get_all_directions_and_qubits()
    algebra = list(algebra.full_paulis.keys())
    paulis = [PauliMonomial([Pauli(loc, pauli) for pauli, loc in
                             zip(pauli[0], pauli[1:])]) for pauli in algebra]
    paulis_A = [p for p in paulis if all(loc in [0, 1] for loc in p.locations)]
    paulis_A = [p * PauliMonomial([Pauli(loc,0) for loc in range(3)]) for p in paulis_A]
    paulis_B = [p for p in paulis if all(loc in [1, 2] for loc in p.locations)]
    paulis_B = [p * PauliMonomial([Pauli(loc,0) for loc in range(3)]) for p in paulis_B]

    return paulis_A, paulis_B

def dynamic_algebra(graph):
    identity = PauliMonomial([Pauli(loc, 0) for loc in range(3)])
    connections_added = 0
    for comb in it.product(graph.nodes, graph.nodes):
        for p in comb:
            if not p in graph.nodes:
                graph.add_node(comb[0])
        p_a_comm_b = comb[0].commutator(comb[1])
        p_a_comm_b.coeff *= -0.5
        if not p_a_comm_b in graph.nodes and not np.isclose(p_a_comm_b.coeff, 0.0):
            graph.add_node(p_a_comm_b)
        if p_a_comm_b == identity:
            if not graph.has_edge(comb[0], identity):
                graph.add_edge(comb[0], identity)
                connections_added += 1
            if not graph.has_edge(comb[1], identity):
                graph.add_edge(comb[1], identity)
                connections_added+=1
        elif not np.isclose(p_a_comm_b.coeff, 0.0):
            if not graph.has_edge(comb[0], p_a_comm_b):
                graph.add_edge(comb[0], p_a_comm_b)
                connections_added+=1
            if not graph.has_edge(comb[1], p_a_comm_b):
                graph.add_edge(comb[1], p_a_comm_b)
                connections_added+=1
    print(f'New connections added: {connections_added}')
    return connections_added

def dynamic_algebra_directed(graph):
    max_depth = max(nx.get_node_attributes(graph,'depth').values())
    print(max_depth)
    identity = PauliMonomial([Pauli(loc, 0) for loc in range(3)])
    connections_added = 0
    nodes_depth = [x for x,y in graph.nodes(data=True)]
    for comb in it.product(nodes_depth,nodes_depth):
        for p in comb:
            if not p in graph.nodes:
                graph.add_node(comb[0], depth=max_depth+1)
        p_a_comm_b = comb[0].commutator(comb[1])
        p_a_comm_b.coeff *= -0.5
        if not p_a_comm_b in graph.nodes and not np.isclose(p_a_comm_b.coeff, 0.0):
            graph.add_node(p_a_comm_b, depth=max_depth+1)
            if p_a_comm_b == identity:
                if not graph.has_edge(comb[0], identity):
                    graph.add_edge(comb[0], identity)
                    connections_added += 1
                if not graph.has_edge(comb[1], identity):
                    graph.add_edge(comb[1], identity)
                    connections_added+=1
            elif not np.isclose(p_a_comm_b.coeff, 0.0):
                if not graph.has_edge(comb[0], p_a_comm_b):
                    graph.add_edge(comb[0], p_a_comm_b)
                    connections_added+=1
                if not graph.has_edge(comb[1], p_a_comm_b):
                    graph.add_edge(comb[1], p_a_comm_b)
                    connections_added+=1
    print(f'New connections added: {connections_added}')
    return connections_added

def generate_product_algebra_graph(algebra):
    G = nx.Graph()
    print('Generating product algebra graph...')
    for pauli in algebra:
        G.add_node(pauli)
    while dynamic_algebra(G):
        pass
    print('Dynamic algebra closed succesfully.')
    return G

def generate_product_algebra_digraph(algebra):
    G = nx.DiGraph()
    print('Generating product algebra graph...')
    for pauli in algebra:
        G.add_node(pauli, depth=0)
    while dynamic_algebra_directed(G):
        pass
    print('Dynamic algebra closed succesfully.')
    return G

def plot_product_algebra_graph(graph, show=True):
    labels = [r"$" + n.__tex__() + "$" for n in graph.nodes]

    nx.draw(graph, with_labels=True, labels=dict(zip(graph.nodes, labels)), pos=nx.circular_layout(graph))
    if show:
        plt.show()
def main():
    paulis_A, paulis_B = SU4_to_SU8_algebra()
    # graph = generate_product_algebra_graph(paulis_A+paulis_B)
    graph = generate_product_algebra_digraph(paulis_A+paulis_B)

    plot_product_algebra_graph(graph)



if __name__ == '__main__':
    main()
