#!/usr/bin/env python3
"""
Command line interface for neural network generation.
Uses the modular functions from infer_connectivity package.
"""

import click
import random
import numpy as np
import networkx as nx
from infer_connectivity import GenerateEdgeSet, CreateGraph, save_connectivity_matrix, get_network_stats
import json
import pathlib


@click.command()
@click.option('--n_e', default=300, help='Number of excitatory neurons')
@click.option('--n_i', default=100, help='Number of inhibitory neurons')
@click.option('--m_ee', default=20.0, help='Mean connections from excitatory to excitatory')
@click.option('--m_ei', default=20.0, help='Mean connections from excitatory to inhibitory')
@click.option('--m_ie', default=20.0, help='Mean connections from inhibitory to excitatory')
@click.option('--m_ii', default=20.0, help='Mean connections from inhibitory to inhibitory')
@click.option('--r', default=5.0, help='Variability parameter')
@click.option('--output', default='graph0.graphml', help='Output GraphML file name')
@click.option('--save-matrix', help='Save connectivity matrix as numpy array (provide filename without extension)')
@click.option('--seed', default=None, type=int, help='Random seed for reproducibility')
@click.option('--show-stats/--no-show-stats', default=True, help='Show network statistics')
def create_network(n_e, n_i, m_ee, m_ei, m_ie, m_ii, r, output, save_matrix, seed, show_stats):
    """
    Create a neural network with specified parameters.

    This script generates a directed graph representing connections between
    excitatory and inhibitory neurons based on the paper "Emergent spike patterns
    in neuronal populations" by L. Chariker and L. Young.

    Examples:
        # Basic usage with defaults
        python network_cli.py

        # Custom parameters
        python network_cli.py --n_e 500 --n_i 150 --r 10 --output my_network.graphml

        # Save connectivity matrix
        python network_cli.py --save-matrix my_network_matrix

        # With specific seed for reproducibility
        python network_cli.py --seed 42 --save-matrix connectivity --output network.graphml
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    click.echo(f"Creating network with {n_e} excitatory and {n_i} inhibitory neurons...")

    # Generate the network
    try:
        EdgeSet = GenerateEdgeSet(n_e, n_i, m_ee, m_ei, m_ie, m_ii, r)
        G = CreateGraph(n_e, n_i, EdgeSet)
    except Exception as e:
        click.echo(f"Error generating network: {e}", err=True)
        return

    # Save GraphML file
    try:
        nx.write_graphml(G, output)
        params = dict(n_e=n_e, n_i=n_i, m_ee=m_ee, m_ei=m_ei, m_ie=m_ie, m_ii=m_ii, r=r)
        path_param = pathlib.Path(output).stem + "_params.json"
        with open(path_param, "w") as f:
            json.dump(params, f)
        click.echo(f"✓ Network saved as GraphML: {output}")
    except Exception as e:
        click.echo(f"Error saving GraphML: {e}", err=True)
        return

    # Save connectivity matrix if requested
    if save_matrix:
        try:
            save_connectivity_matrix(G, save_matrix)
            click.echo(f"✓ Connectivity matrix saved as: {save_matrix}.npy")
        except Exception as e:
            click.echo(f"Error saving connectivity matrix: {e}", err=True)

    # Show network statistics
    if show_stats:
        try:
            stats = get_network_stats(G, n_e)
            click.echo("\n" + "=" * 50)
            click.echo("NETWORK STATISTICS")
            click.echo("=" * 50)
            click.echo(f"Total nodes: {stats['total_nodes']}")
            click.echo(f"  - Excitatory: {stats['excitatory_nodes']}")
            click.echo(f"  - Inhibitory: {stats['inhibitory_nodes']}")
            click.echo(f"Total edges: {stats['total_edges']}")
            click.echo(f"Average degree: {stats['avg_degree']:.2f}")
            click.echo(f"Average in-degree: {stats['avg_in_degree']:.2f} ± {stats['std_in_degree']:.2f}")
            click.echo(f"Average out-degree: {stats['avg_out_degree']:.2f} ± {stats['std_out_degree']:.2f}")
            click.echo("=" * 50)
        except Exception as e:
            click.echo(f"Error calculating statistics: {e}", err=True)


if __name__ == "__main__":
    create_network()