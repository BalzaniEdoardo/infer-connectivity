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


@click.group()
def cli():
    """Neural Network Generation Tools"""
    pass


@cli.command()
@click.argument('matrix_file')
def analyze_matrix(matrix_file):
    """Analyze a saved connectivity matrix."""
    try:
        matrix = np.load(matrix_file)
        click.echo(f"Matrix shape: {matrix.shape}")
        click.echo(f"Total connections: {np.sum(matrix)}")
        click.echo(f"Connection density: {np.sum(matrix) / (matrix.shape[0] * matrix.shape[1]):.4f}")
        click.echo(f"Mean degree: {np.mean(np.sum(matrix, axis=1)):.2f}")
    except Exception as e:
        click.echo(f"Error analyzing matrix: {e}", err=True)


@cli.command()
@click.argument('graphml_file')
def analyze_graph(graphml_file):
    """Analyze a saved GraphML network."""
    try:
        G = nx.read_graphml(graphml_file, node_type=int)
        # Assume first 300 nodes are excitatory (you might want to make this configurable)
        n_e = 300
        stats = get_network_stats(G, n_e)

        click.echo(f"Graph from: {graphml_file}")
        click.echo(f"Nodes: {stats['total_nodes']} (E: {stats['excitatory_nodes']}, I: {stats['inhibitory_nodes']})")
        click.echo(f"Edges: {stats['total_edges']}")
        click.echo(f"Average degree: {stats['avg_degree']:.2f}")

    except Exception as e:
        click.echo(f"Error analyzing graph: {e}", err=True)


# Add the original create_network command to the group
cli.add_command(create_network)

if __name__ == "__main__":
    cli()