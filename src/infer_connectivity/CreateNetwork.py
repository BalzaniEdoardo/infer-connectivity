#Author: Sonica Saraf 2017
#Creates network of neurons according to "Emergent spike patterns in neuronal populations" by L. Chariker and L. Young

from __future__ import division
from scipy import stats
import networkx as nx
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import os

def GenerateEdgeSet(N_E, N_I, m_EE, m_EI, m_IE, m_II, r):
  Edges = []
  wAmount = stats.truncnorm(-1, 1).rvs(N_E)
  wAmount = [ int(round(m_EE+r*i)) for i in wAmount ]
  xAmount = stats.truncnorm(-1, 1).rvs(N_E)
  xAmount = [ int(round(m_EI+(r/2)*i)) for i in xAmount ]
  yAmount = stats.truncnorm(-1, 1).rvs(N_I)
  yAmount = [ int(round(m_IE+2.5*r*i)) for i in yAmount ]
  zAmount = stats.truncnorm(-1, 1).rvs(N_I)
  zAmount = [ int(round(m_II+(r/2)*i)) for i in zAmount ]

  for exc in range(N_E):
    a = list(range(exc))
    a.extend(range(exc+1, N_E))
    excList = random.sample(a, wAmount[exc])
    excToExc = map(lambda e:(e, exc), excList)
    Edges.extend(excToExc)
    inhibList = random.sample(range(N_E, N_E + N_I), xAmount[exc])
    inhToExc = map(lambda inh:(inh, exc), inhibList)
    Edges.extend(inhToExc)
  for inh in range(N_E, N_E + N_I):
    excList = random.sample(range(N_E), yAmount[inh-N_E])
    excToInh = map(lambda exc:(exc, inh), excList)
    Edges.extend(excToInh)
    a = list(range(N_E, inh))
    a.extend(range(inh+1, N_E + N_I))
    inhibList = random.sample(a, zAmount[inh-N_E])
    inhToInh = map(lambda i:(i, inh), inhibList)
    Edges.extend(inhToInh)
  return Edges

def CreateGraph(N_E, N_I, EdgeSet):
  G = nx.DiGraph()
  G.add_nodes_from(range(N_E + N_I))
  voltDict = {k: random.random() for k in range(N_E+N_I)}
  gEDict = {k: random.random()/769 for k in range(N_E + N_I)}
  gIDict = {k: random.random()/769 for k in range(N_E + N_I)}
  nx.set_node_attributes(G, voltDict, 'voltage')
  nx.set_node_attributes(G, gEDict, 'g_E')
  nx.set_node_attributes(G, gIDict, 'g_I')
  nx.set_node_attributes(G, 0, 'refractory_left')
  nx.set_node_attributes(G, 0, 'spike_count')
  G.add_edges_from(EdgeSet)
  return G

if __name__ == "__main__":
  input = input("Enter N_E, N_I, m_EE, m_EI, m_IE, m_II, r separated by comma and spaces: ")
  inputs = input.split(', ')
  args = [float(i) for i in inputs]
  N_E = int(args[0])
  N_I = int(args[1])
  EdgeSet = GenerateEdgeSet(N_E, N_I, args[2], args[3], args[4], args[5], args[6])
  G = CreateGraph(N_E, N_I, EdgeSet)
 # os.remove('graph0.graphml')
  nx.write_graphml(G, 'graph0.graphml')
 # plt.figure(figsize= (16, 16))
 # pos = nx.circular_layout(G)
 # nx.draw_networkx_nodes(G, pos, nodelist=xrange(N_E), node_color = "r")
 # nx.draw_networkx_nodes(G, pos, nodelist=xrange(N_E, N_E + N_I), node_color = "b")
#  nx.draw_networkx_edges(G, pos, arrows = False)
#  nx.draw_networkx_labels(G, pos) 
 # plt.show()
