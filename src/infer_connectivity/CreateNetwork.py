# Author: Sonica Saraf 2017
# Creates network of neurons according to "Emergent spike patterns in neuronal populations" by L. Chariker and L. Young
# Modularized for use as importable functions

from __future__ import division
from scipy import stats
import networkx as nx
import random
import numpy as np


def GenerateEdgeSet(N_E, N_I, m_EE, m_EI, m_IE, m_II, r):
    """
    Generate edges for neural network based on truncated normal distributions.

    Parameters
    ----------
    N_E : int
        Number of excitatory neurons
    N_I : int
        Number of inhibitory neurons
    m_EE : float
        Mean connections from excitatory to excitatory
    m_EI : float
        Mean connections from excitatory to inhibitory
    m_IE : float
        Mean connections from inhibitory to excitatory
    m_II : float
        Mean connections from inhibitory to inhibitory
    r : float
        Variability parameter

    Returns
    -------
    list
        List of edge tuples (source, target)
    """
    Edges = []
    wAmount = stats.truncnorm(-1, 1).rvs(N_E)
    wAmount = [int(round(m_EE + r * i)) for i in wAmount]
    xAmount = stats.truncnorm(-1, 1).rvs(N_E)
    xAmount = [int(round(m_EI + (r / 2) * i)) for i in xAmount]
    yAmount = stats.truncnorm(-1, 1).rvs(N_I)
    yAmount = [int(round(m_IE + 2.5 * r * i)) for i in yAmount]
    zAmount = stats.truncnorm(-1, 1).rvs(N_I)
    zAmount = [int(round(m_II + (r / 2) * i)) for i in zAmount]

    # Excitatory to excitatory and inhibitory to excitatory connections
    for exc in range(N_E):
        # Excitatory to excitatory (avoid self-connections)
        a = list(range(exc))
        a.extend(range(exc + 1, N_E))
        if wAmount[exc] <= len(a):
            excList = random.sample(a, wAmount[exc])
            excToExc = [(e, exc) for e in excList]
            Edges.extend(excToExc)

        # Inhibitory to excitatory
        if xAmount[exc] <= N_I:
            inhibList = random.sample(range(N_E, N_E + N_I), xAmount[exc])
            inhToExc = [(inh, exc) for inh in inhibList]
            Edges.extend(inhToExc)

    # Inhibitory connections
    for inh in range(N_E, N_E + N_I):
        # Excitatory to inhibitory
        if yAmount[inh - N_E] <= N_E:
            excList = random.sample(range(N_E), yAmount[inh - N_E])
            excToInh = [(exc, inh) for exc in excList]
            Edges.extend(excToInh)

        # Inhibitory to inhibitory (avoid self-connections)
        a = list(range(N_E, inh))
        a.extend(range(inh + 1, N_E + N_I))
        if zAmount[inh - N_E] <= len(a):
            inhibList = random.sample(a, zAmount[inh - N_E])
            inhToInh = [(i, inh) for i in inhibList]
            Edges.extend(inhToInh)

    return Edges


def CreateGraph(N_E, N_I, EdgeSet):
    """
    Create a NetworkX directed graph from edge set with neuron attributes.

    Parameters
    ----------
    N_E : int
        Number of excitatory neurons
    N_I : int
        Number of inhibitory neurons
    EdgeSet : list
        List of edge tuples (source, target)

    Returns
    -------
    networkx.DiGraph
        Directed graph with neuron nodes and synaptic edges
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(N_E + N_I))

    # Initialize node attributes
    voltDict = {k: random.random() for k in range(N_E + N_I)}
    gEDict = {k: random.random() / 769 for k in range(N_E + N_I)}
    gIDict = {k: random.random() / 769 for k in range(N_E + N_I)}

    nx.set_node_attributes(G, voltDict, 'voltage')
    nx.set_node_attributes(G, gEDict, 'g_E')
    nx.set_node_attributes(G, gIDict, 'g_I')
    nx.set_node_attributes(G, 0, 'refractory_left')
    nx.set_node_attributes(G, 0, 'spike_count')

    G.add_edges_from(EdgeSet)
    return G


def save_connectivity_matrix(G, filepath):
    """
    Save the connectivity matrix of a graph as a numpy array.

    Parameters
    ----------
    G : networkx.DiGraph
        The network graph
    filepath : str
        Path to save the matrix (without .npy extension)
    """
    connectivity_matrix = nx.adjacency_matrix(G).todense()
    np.save(f"{filepath}.npy", connectivity_matrix)


def get_network_stats(G, N_E):
    """
    Calculate and return network statistics.

    Parameters
    ----------
    G : networkx.DiGraph
        The network graph
    N_E : int
        Number of excitatory neurons

    Returns
    -------
    dict
        Dictionary containing network statistics
    """
    stats = {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'excitatory_nodes': N_E,
        'inhibitory_nodes': G.number_of_nodes() - N_E,
    }

    in_degrees = [G.in_degree(n) for n in G.nodes()]
    out_degrees = [G.out_degree(n) for n in G.nodes()]

    stats.update({
        'avg_in_degree': np.mean(in_degrees),
        'std_in_degree': np.std(in_degrees),
        'avg_out_degree': np.mean(out_degrees),
        'std_out_degree': np.std(out_degrees),
        'avg_degree': G.number_of_edges() * 2 / G.number_of_nodes()
    })

    return stats