# Author: Sonica Saraf 2017
# Simulates neuronal firing similar to "Emergent spike patterns in neuronal populations" by L. Chariker and L. Young,
# and "Malleability of gamma rhythms enhances population-level correlation" by S. Saraf and L. Young
# Minimal modularization - keeping original code structure intact

from __future__ import division
import networkx as nx
import random
import csv
import os
import pickle

# Global variables - keeping exactly as original
V_E = 14 / 3
V_I = -2 / 3
V_R = 0
V_T = 1
Tau_leak = 20
S_amb = .003
lambda_amb = .433
Espikes = [0 for i in range(400)]
Ispikes = [0 for i in range(400)]
Drivespikes = [0 for i in range(400)]
Ambspikes = [0 for i in range(400)]
NodeEConductance = []
NodeIConductance = []
NodeVoltage = []


def trackEConductance(G, i, time):
    node = G.nodes[i]  # Only change: G.node -> G.nodes for newer networkx
    NodeEConductance.append([str(node['g_E']), str(time)])


def trackIConductance(G, i, time):
    node = G.nodes[i]  # Only change: G.node -> G.nodes for newer networkx
    NodeIConductance.append([str(node['g_I']), str(time)])


def trackVoltage(G, i, time):
    node = G.nodes[i]  # Only change: G.node -> G.nodes for newer networkx
    NodeVoltage.append([str(node['voltage']), str(time)])


def avgVoltage(voltagesE, voltagesI, steps, time, NE, NI):
    avgE = voltagesE / (NE * steps)
    avgI = voltagesI / (NI * steps)
    with open('avgVoltage1.csv', 'a') as f:
        writer = csv.writer(f)
        row = [str(avgE), str(avgI), str(time)]
        writer.writerow(row)
    f.close()


def Simulate(G, spikes, spikestime, spikescounter, refrac, EEdel, IEdel, EIdel, IIdel, Edrive, Idrive, SEE, SIE, SEI,
             SII, TauE, TauI, NE, NI, sec):
    if os.path.exists('EConductance1.csv'):
        os.remove('EConductance1.csv')
    if os.path.exists('IConductance1.csv'):
        os.remove('IConductance1.csv')
    if os.path.exists('Voltage1.csv'):
        os.remove('Voltage1.csv')
    if os.path.exists('avgVoltage1.csv'):
        os.remove('avgVoltage1.csv')
    UpdatedGraph, spikes, spikestime = step(G, 0, spikes, spikestime, spikescounter, refrac, EEdel, IEdel, EIdel, IIdel,
                                            Edrive, Idrive, SEE, SIE, SEI, SII, TauE, TauI, NE, NI)
    count = 1.0
    voltagesE = 0.0
    voltagesI = 0.0
    while count < sec * 10000:
        UpdatedGraph, spikes, spikestime = step(UpdatedGraph, count, spikes, spikestime, spikescounter, refrac, EEdel,
                                                IEdel, EIdel, IIdel, Edrive, Idrive, SEE, SIE, SEI, SII, TauE, TauI, NE,
                                                NI)
        if count % 500 == 0:
            if count != 20000:
                spikesExc1 = len(sum(spikes[:NE], []))
                spikesInh = len(sum(spikes[NE:NE + NI], []))
                print(str((spikesExc1) / (((count / 10000) - 2) * NE)) + "," + str(
                    (spikesInh) / (((count / 10000) - 2) * NI)))
        count += 1.0
    return spikes


def step(G, count, spikes, spikestime, spikescounter, refrac, EEdel, IEdel, EIdel, IIdel, Edrive, Idrive, SEE, SIE, SEI,
         SII, TauE, TauI, NE, NI):
    for i in range(NE + NI):
        node = G.nodes[i]
        delG_E = -1 * node['g_E'] / TauE
        delG_I = -1 * node['g_I'] / TauI

        # ambient drive component
        if random.random() < lambda_amb * 0.1:
            delG_E += S_amb * (1 / TauE)
            Ambspikes[i] += 1

        # S_EE S_IE kicks
        if i < NE:
            if random.random() < Edrive * 0.1:
                delG_E += SEE * (1 / TauE)
                Drivespikes[i] += 1
        else:
            if random.random() < Idrive * 0.1:
                delG_E += SIE * (1 / TauE)
                Drivespikes[i] += 1

        receivedSpikes = [[time, E] for [time, E] in spikescounter[i] if time == count]
        for [time, E] in receivedSpikes:
            if random.random() < .5:
                if E:
                    if i < NE:
                        node['g_E'] += SEE * (1 / TauE) * 0.1
                    else:
                        node['g_E'] += SIE * (1 / TauE) * 0.1
                    Espikes[i] += 1
                else:
                    if i < NE:
                        node['g_I'] += SEI * (1 / TauI) * 0.1
                    else:
                        node['g_I'] += SII * (1 / TauI) * 0.1
                    Ispikes[i] += 1
            spikescounter[i].remove([time, E])

        node['g_E'] += 0.1 * delG_E
        node['g_I'] += 0.1 * delG_I

        # node is in refractory period
        if node['refractory_left'] > 0:
            node['refractory_left'] = node['refractory_left'] - 1
        # node is not in refractory period
        else:
            # update V as well
            V = node['voltage']
            delV = -1 / Tau_leak * V * 0.1 - (V - V_E) * node['g_E'] - (V - V_I) * node['g_I']
            node['voltage'] += delV
            # if node spiked
            if node['voltage'] >= 1.0:
                if count >= 20000:
                    spikes[i].append(count / 10.0)
                for s in G.successors(i):
                    isExcitatory = 1 if i < NE else 0
                    spikescounter[s].append([count + spikestime[i][s], isExcitatory])
                node['voltage'] = 0
                node['refractory_left'] = refrac
    return G, spikes, spikestime


def run_simulation_with_params(graph_file='graph0.graphml', seed=102194,
                               refrac=20, EEdela=10, EEdel=23, IEdel=10, EIdel=10, IIdel=10,
                               Edrive=.288, Idrive=.972, SEE=.0255, SIE=.01, SEI=.084, SII=.0275,
                               TauE=2, TauI=4, cE=1.1, cI=None, NE=300, NI=100, sec=3,
                               output_prefix="simulation"):
    """
    Run simulation with given parameters. This is the main function that wraps
    the original main block code.
    """
    # Reset global arrays - size them for the actual network size
    global Espikes, Ispikes, Drivespikes, Ambspikes, NodeEConductance, NodeIConductance, NodeVoltage
    total_neurons = NE + NI
    Espikes = [0 for i in range(total_neurons)]
    Ispikes = [0 for i in range(total_neurons)]
    Drivespikes = [0 for i in range(total_neurons)]
    Ambspikes = [0 for i in range(total_neurons)]
    NodeEConductance = []
    NodeIConductance = []
    NodeVoltage = []

    # Original main block code - keeping exactly the same
    random.seed(seed)

    if cI is None:
        cI = .5 * (cE + 1)

    Edrive = cE * Edrive
    Idrive = cI * Idrive

    spikes = [[] for i in range(NE + NI)]
    spikestime = [[10 for j in range(NE + NI)] for i in range(NE + NI)]
    for i in range(NE + NI):
        if i < NE:
            spikestime[i] = [random.randint(EEdela, EEdel) if x < NE else 10 for x in range(NE + NI)]
    spikescounter = [[] for j in range(NE + NI)]

    print("\n")
    G = nx.read_graphml(graph_file, int)
    spikes = Simulate(G, spikes, spikestime, spikescounter, refrac, EEdel, IEdel, EIdel, IIdel, Edrive, Idrive, SEE,
                      SIE, SEI, SII, TauE, TauI, NE, NI, sec)

    # Save results with custom prefix
    f = open(f'{output_prefix}_spikes.pckl', 'wb')
    pickle.dump(spikes, f)
    f.close()
    f1 = open(f'{output_prefix}_Espikes.pckl', 'wb')
    pickle.dump(Espikes, f1)
    f1.close()
    f2 = open(f'{output_prefix}_Ispikes.pckl', 'wb')
    pickle.dump(Ispikes, f2)
    f2.close()
    f3 = open(f'{output_prefix}_Drivespikes.pckl', 'wb')
    pickle.dump(Drivespikes, f3)
    f3.close()
    f7 = open(f'{output_prefix}_Ambspikes.pckl', 'wb')
    pickle.dump(Ambspikes, f7)
    f7.close()
    f4 = open(f'{output_prefix}_NodeEConductance.pckl', 'wb')
    pickle.dump(NodeEConductance, f4)
    f4.close()
    f5 = open(f'{output_prefix}_NodeIConductance.pckl', 'wb')
    pickle.dump(NodeIConductance, f5)
    f5.close()
    f6 = open(f'{output_prefix}_NodeVoltage.pckl', 'wb')
    pickle.dump(NodeVoltage, f6)
    f6.close()

    return spikes


# Keep original main block for backward compatibility
if __name__ == "__main__":
    # Original main code - unchanged
    random.seed(102194)
    refrac = 20
    EEdela = 10
    EEdel = 23
    IEdel = 10
    EIdel = 10
    IIdel = 10
    Edrive = .288
    Idrive = .972
    SEE = .0255  # 0.0255
    SIE = .01  # 0.0095
    SEI = .084  # 0.054
    SII = .0275  # 0.0275
    TauE = 2
    TauI = 4  # 4
    cE = 1.1  # 2.25
    cI = .5 * (cE + 1)
    Edrive = cE * Edrive
    Idrive = cI * Idrive
    NE = 300
    NI = 100
    sec = 3
    spikes = [[] for i in range(NE + NI)]
    spikestime = [[10 for j in range(NE + NI)] for i in range(NE + NI)]
    for i in range(NE + NI):
        if i < NE:
            spikestime[i] = [random.randint(EEdela, EEdel) if x < NE else 10 for x in range(NE + NI)]
    spikescounter = [[] for j in range(NE + NI)]

    print("\n")
    G = nx.read_graphml('graph0.graphml', int)
    spikes = Simulate(G, spikes, spikestime, spikescounter, refrac, EEdel, IEdel, EIdel, IIdel, Edrive, Idrive, SEE,
                      SIE, SEI, SII, TauE, TauI, NE, NI, sec)
    f = open('spikes1.pckl', 'wb')
    pickle.dump(spikes, f)
    f.close()
    f1 = open('Espikes1.pckl', 'wb')
    pickle.dump(Espikes, f1)
    f1.close()
    f2 = open('Ispikes1.pckl', 'wb')
    pickle.dump(Ispikes, f2)
    f2.close()
    f3 = open('Drivespikes1.pckl', 'wb')
    pickle.dump(Drivespikes, f3)
    f3.close()
    f7 = open('Ambspikes1.pckl', 'wb')
    pickle.dump(Ambspikes, f7)
    f7.close()
    f4 = open('NodeEConductance1.pckl', 'wb')
    pickle.dump(NodeEConductance, f4)
    f4.close()
    f5 = open('NodeIConductance1.pckl', 'wb')
    pickle.dump(NodeIConductance, f5)
    f5.close()
    f6 = open('NodeVoltage1.pckl', 'wb')
    pickle.dump(NodeVoltage, f6)
    f6.close()