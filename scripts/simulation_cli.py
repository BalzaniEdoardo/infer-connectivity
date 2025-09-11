#!/usr/bin/env python3
"""
Simple CLI wrapper for the original neural simulation code.
Minimal changes - just adds Click interface to existing functions.
"""

import click
import os
from infer_connectivity.Simulatenew2 import run_simulation_with_params


@click.command()
@click.option('--graph-file', default='graph0.graphml', help='Input GraphML network file')
@click.option('--output-prefix', default='simulation', help='Prefix for output files')
@click.option('--seed', default=102194, type=int, help='Random seed')
@click.option('--duration', default=3.0, help='Simulation duration in seconds')
@click.option('--n-e', default=300, help='Number of excitatory neurons')
@click.option('--n-i', default=100, help='Number of inhibitory neurons')
@click.option('--refrac', default=20, help='Refractory period')
@click.option('--ee-delay-min', default=10, help='Min EE delay')
@click.option('--ee-delay-max', default=23, help='Max EE delay')
@click.option('--ie-delay', default=10, help='IE delay')
@click.option('--ei-delay', default=10, help='EI delay')
@click.option('--ii-delay', default=10, help='II delay')
@click.option('--e-drive', default=0.288, help='Excitatory drive')
@click.option('--i-drive', default=0.972, help='Inhibitory drive')
@click.option('--s-ee', default=0.0255, help='EE synaptic strength')
@click.option('--s-ie', default=0.01, help='IE synaptic strength')
@click.option('--s-ei', default=0.084, help='EI synaptic strength')
@click.option('--s-ii', default=0.0275, help='II synaptic strength')
@click.option('--tau-e', default=2.0, help='Excitatory time constant')
@click.option('--tau-i', default=4.0, help='Inhibitory time constant')
@click.option('--c-e', default=1.1, help='Excitatory drive multiplier')
@click.option('--c-i', default=None, type=float, help='Inhibitory drive multiplier')
def simulate(graph_file, output_prefix, seed, duration, n_e, n_i, refrac,
             ee_delay_min, ee_delay_max, ie_delay, ei_delay, ii_delay,
             e_drive, i_drive, s_ee, s_ie, s_ei, s_ii, tau_e, tau_i, c_e, c_i):
    """
    Run neural network simulation.

    Examples:
        python simulation_cli.py
        python simulation_cli.py --duration 5 --s-ee 0.03
        python simulation_cli.py --graph-file my_network.graphml --output-prefix my_sim
    """

    # Check if network file exists
    if not os.path.exists(graph_file):
        click.echo(f"Error: Network file '{graph_file}' not found!", err=True)
        return

    # Load network to get actual size
    import networkx as nx
    G = nx.read_graphml(graph_file, node_type=int)
    actual_nodes = G.number_of_nodes()

    # Warn if network size doesn't match parameters
    if actual_nodes != n_e + n_i:
        click.echo(f"Warning: Network has {actual_nodes} nodes but parameters specify {n_e + n_i}")
        click.echo(f"Adjusting simulation parameters to match network size...")

        # Estimate E/I split based on typical ratio or user parameters
        total_requested = n_e + n_i
        e_ratio = n_e / total_requested
        n_e = int(actual_nodes * e_ratio)
        n_i = actual_nodes - n_e

        click.echo(f"Using: {n_e} excitatory, {n_i} inhibitory neurons")

    click.echo(f"Running simulation with {graph_file}...")
    click.echo(f"Duration: {duration}s, Seed: {seed}")
    click.echo(f"Network: {n_e}E + {n_i}I = {n_e + n_i} total neurons")

    # Call the original simulation function
    spikes = run_simulation_with_params(
        graph_file=graph_file,
        seed=seed,
        refrac=refrac,
        EEdela=ee_delay_min,
        EEdel=ee_delay_max,
        IEdel=ie_delay,
        EIdel=ei_delay,
        IIdel=ii_delay,
        Edrive=e_drive,
        Idrive=i_drive,
        SEE=s_ee,
        SIE=s_ie,
        SEI=s_ei,
        SII=s_ii,
        TauE=tau_e,
        TauI=tau_i,
        cE=c_e,
        cI=c_i,
        NE=n_e,
        NI=n_i,
        sec=duration,
        output_prefix=output_prefix
    )

    # Simple result summary
    total_exc_spikes = len(sum(spikes[:n_e], []))
    total_inh_spikes = len(sum(spikes[n_e:n_e + n_i], []))

    click.echo(f"Simulation complete!")
    click.echo(f"Total excitatory spikes: {total_exc_spikes}")
    click.echo(f"Total inhibitory spikes: {total_inh_spikes}")
    click.echo(f"Results saved with prefix: {output_prefix}")


if __name__ == "__main__":
    simulate()