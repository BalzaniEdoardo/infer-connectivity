import os
import pathlib
import json
import warnings
import re
from pathlib import Path

import nemos as nmo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from matplotlib.colors import ListedColormap, to_rgb
from nemos.regularizer import GroupLasso

from infer_connectivity import load_model
from infer_connectivity.roc_utils import compute_filters, compute_roc_curve
from infer_connectivity.utils import extract_coef_single_neu_glm

# Set environment variables for memory management
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Color palette for graded visualizations
GRADED_COLOR_LIST = [
    "navy", "blue", "royalblue", "cornflowerblue", "skyblue", "lightblue",
    "aquamarine", "mediumseagreen", "limegreen", "yellowgreen", "gold",
    "orange", "darkorange", "tomato", "orangered", "red", "crimson",
    "deeppink", "magenta",
]

# ============================================================================
# Configuration and Setup
# ============================================================================
base_dir = Path("/Users/ebalzani/Code/infer-connectivity/infer-connectivity/")
graph_file = base_dir / "simulations/sonica-sept-25-2025/graph0-sonica-sept-26-2026.graphml"
fit_id = "sonica-oct-8-2025-400-seconds"
output_directory = base_dir / "scripts" / "output" / fit_id
figures_directory = base_dir / "figures" / fit_id
config_directory = base_dir / "scripts" / "configs" / fit_id
simulation_directory = base_dir / "simulations" / fit_id
figures_directory.mkdir(exist_ok=True, parents=True)
# Load configuration
conf = None
for conf_file in config_directory.iterdir():
    if conf_file.suffix == ".json":
        conf = conf_file
        break

if conf:
    with open(conf, "r") as fh:
        conf_dict = json.load(fh)
else:
    conf_dict = {}

# Extract neuron populations
all_inhib_neu = np.array(conf_dict["inhibitory_neuron_id"])

# Load connectivity matrix
connectivity_path = simulation_directory / pathlib.Path(conf_dict["connectivity_path"]).name
full_conn = np.load(connectivity_path)
full_conn += np.eye(full_conn.shape[0], dtype=int)  # Add self-connections

# Extract parameters with defaults for backward compatibility
if not all(key in conf_dict for key in ["binsize", "history_window", "n_basis_funcs"]):
    warnings.warn(
        "Parameters were hard-coded. Please verify defaults are correct.",
        UserWarning
    )

binsize = conf_dict.get("binsize", 0.0003)
history_window = conf_dict.get("history_window", 0.014)
window_size = int(history_window / binsize)
n_basis_funcs = conf_dict.get("n_basis_funcs", 4)
basis_cls_name = conf_dict["basis_cls_name"]

# Initialize basis functions
basis_cls = getattr(nmo.basis, basis_cls_name)
basis = basis_cls(n_basis_funcs, window_size=window_size)

# ============================================================================
# Data Processing
# ============================================================================
steps = [1, 2, 5, 10, 25]

# Initialize storage dictionaries
coef_dict = {}
neu_ids = {}
reg_strengths = {}
filters = {}
true_conn = {}
fpr, tpr, roc_auc = {}, {}, {}
fpr_inh, tpr_inh, roc_auc_inh = {}, {}, {}
fpr_exc, tpr_exc, roc_auc_exc = {}, {}, {}

# Process each subsampling step
for step in steps:
    print(f"Analyzing step {step}")

    # Define pattern for loading files
    if step != 1:
        pattern = re.compile(fr"best_.*neuron_(\d+).*_{step}\.npz$")
    else:
        pattern = re.compile(r"best_.*neuron_(\d+).*e\.npz$")

    # Extract coefficients
    neu_ids[step], coef_dict[step], reg_strengths[step] = extract_coef_single_neu_glm(
        output_directory, pattern=pattern
    )

    # Compute filters and connectivity
    filters[step] = compute_filters(coef_dict[step], basis)
    true_conn[step] = full_conn[neu_ids[step], :][:, neu_ids[step]]

    # Overall ROC
    fpr[step], tpr[step], roc_auc[step], _, _, _ = compute_roc_curve(
        true_conn[step], filters[step]
    )

    # Inhibitory presynaptic neurons
    inhib_neu = np.intersect1d(all_inhib_neu, neu_ids[step])
    conn_presynaptic_inhib = full_conn[inhib_neu, :][:, neu_ids[step]]
    idx_inhib_neu = np.searchsorted(neu_ids[step], inhib_neu)
    fpr_inh[step], tpr_inh[step], roc_auc_inh[step], _, _, _ = compute_roc_curve(
        conn_presynaptic_inhib, filters[step][idx_inhib_neu]
    )

    # Excitatory presynaptic neurons
    excit_neu = np.setdiff1d(neu_ids[step], inhib_neu)
    conn_presynaptic_excit = full_conn[excit_neu, :][:, neu_ids[step]]
    idx_excit_neu = np.searchsorted(neu_ids[step], excit_neu)
    fpr_exc[step], tpr_exc[step], roc_auc_exc[step], _, _, _ = compute_roc_curve(
        conn_presynaptic_excit, filters[step][idx_excit_neu]
    )


# ============================================================================
# Analysis of Connection Types (E→E, E→I, I→E, I→I)
# ============================================================================
def analyze_connection_types(step, neu_ids_step, all_inhib_neu, full_conn, filters_step):
    """
    Analyze ROC curves for all four connection types.

    Parameters
    ----------
    step : int
        Subsampling step
    neu_ids_step : array
        Neuron IDs for this step
    all_inhib_neu : array
        All inhibitory neuron IDs
    full_conn : array
        Full connectivity matrix
    filters_step : array
        Computed filters for this step

    Returns
    -------
    dict : Dictionary containing FPR, TPR, AUC, and counts for each connection type
    """
    # Identify neuron types
    inhib_neu = np.intersect1d(all_inhib_neu, neu_ids_step)
    excit_neu = np.setdiff1d(neu_ids_step, inhib_neu)

    # Get indices in the subsampled network
    idx_inhib = np.searchsorted(neu_ids_step, inhib_neu)
    idx_excit = np.searchsorted(neu_ids_step, excit_neu)

    results = {}

    # E→E connections
    conn_EE = full_conn[np.ix_(excit_neu, excit_neu)]
    filters_EE = filters_step[np.ix_(idx_excit, idx_excit)]
    fpr_EE, tpr_EE, auc_EE, _, _, _ = compute_roc_curve(conn_EE, filters_EE)
    results['E→E'] = {
        'fpr': fpr_EE, 'tpr': tpr_EE, 'auc': auc_EE,
        'n_connections': np.sum(conn_EE),
        'n_possible': conn_EE.size
    }

    # E→I connections
    conn_EI = full_conn[np.ix_(excit_neu, inhib_neu)]
    filters_EI = filters_step[np.ix_(idx_excit, idx_inhib)]
    fpr_EI, tpr_EI, auc_EI, _, _, _ = compute_roc_curve(conn_EI, filters_EI)
    results['E→I'] = {
        'fpr': fpr_EI, 'tpr': tpr_EI, 'auc': auc_EI,
        'n_connections': np.sum(conn_EI),
        'n_possible': conn_EI.size
    }

    # I→E connections
    conn_IE = full_conn[np.ix_(inhib_neu, excit_neu)]
    filters_IE = filters_step[np.ix_(idx_inhib, idx_excit)]
    fpr_IE, tpr_IE, auc_IE, _, _, _ = compute_roc_curve(conn_IE, filters_IE)
    results['I→E'] = {
        'fpr': fpr_IE, 'tpr': tpr_IE, 'auc': auc_IE,
        'n_connections': np.sum(conn_IE),
        'n_possible': conn_IE.size
    }

    # I→I connections
    conn_II = full_conn[np.ix_(inhib_neu, inhib_neu)]
    filters_II = filters_step[np.ix_(idx_inhib, idx_inhib)]
    fpr_II, tpr_II, auc_II, _, _, _ = compute_roc_curve(conn_II, filters_II)
    results['I→I'] = {
        'fpr': fpr_II, 'tpr': tpr_II, 'auc': auc_II,
        'n_connections': np.sum(conn_II),
        'n_possible': conn_II.size
    }

    return results


# Compute connection type results for all steps
conn_type_results = {}
for step in steps:
    conn_type_results[step] = analyze_connection_types(
        step, neu_ids[step], all_inhib_neu, full_conn, filters[step]
    )


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_roc_comparison(steps, fpr, tpr, roc_auc, fpr_exc, tpr_exc, roc_auc_exc,
                        fpr_inh, tpr_inh, roc_auc_inh):
    """Plot ROC curves comparing overall, excitatory, and inhibitory neurons."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Use graded color list, selecting colors evenly spaced
    colors = [GRADED_COLOR_LIST[i] for i in np.arange(len(steps))]

    titles = ['Overall', 'Excitatory Presynaptic', 'Inhibitory Presynaptic']
    data_sets = [
        (fpr, tpr, roc_auc),
        (fpr_exc, tpr_exc, roc_auc_exc),
        (fpr_inh, tpr_inh, roc_auc_inh)
    ]

    for ax, title, (fpr_dict, tpr_dict, auc_dict) in zip(axes, titles, data_sets):
        for step, color in zip(steps, colors):
            pct = 100.0 / step
            ax.plot(fpr_dict[step], tpr_dict[step],
                    label=f'{pct:.1f}% neurons (AUC={auc_dict[step]:.3f})',
                    color=color, linewidth=2)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Chance')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    return fig


def plot_connection_type_roc(conn_type_results, steps):
    """Plot ROC curves for all four connection types across sampling steps."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    conn_types = ['E→E', 'E→I', 'I→E', 'I→I']

    # Use graded color list, selecting colors evenly spaced
    colors = [GRADED_COLOR_LIST[i] for i in np.arange(len(steps))]

    for ax, conn_type in zip(axes, conn_types):
        for step, color in zip(steps, colors):
            pct = 100.0 / step
            results = conn_type_results[step][conn_type]
            ax.plot(results['fpr'], results['tpr'],
                    label=f'{pct:.1f}% neurons (AUC={results["auc"]:.3f})',
                    color=color, linewidth=2)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Chance')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'{conn_type} Connections', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    return fig


def plot_auc_summary(conn_type_results, steps, roc_auc, roc_auc_exc, roc_auc_inh):
    """Create bar plot summarizing AUC across all analyses."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Prepare data
    auc_data = []
    for step in steps:
        pct = 100.0 / step
        auc_data.append({
            'Step': step,
            'Percent': pct,
            'Overall': roc_auc[step],
            'E→E': conn_type_results[step]['E→E']['auc'],
            'E→I': conn_type_results[step]['E→I']['auc'],
            'I→E': conn_type_results[step]['I→E']['auc'],
            'I→I': conn_type_results[step]['I→I']['auc'],
        })

    df = pd.DataFrame(auc_data)

    # Plot
    x = np.arange(len(steps))
    width = 0.12
    analyses = ['Overall', 'E→E', 'E→I', 'I→E', 'I→I']
    colors_palette = plt.cm.Set3(np.linspace(0, 1, len(analyses)))

    for i, analysis in enumerate(analyses):
        data = df[analysis].values
        ax.bar(x + i * width, data, width, label=analysis, color=colors_palette[i])

    ax.set_xlabel('Subsampling Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROC AUC', fontsize=12, fontweight='bold')
    ax.set_title('ROC AUC Comparison Across Analyses', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * len(analyses) / 2 - width/2)
    ax.set_xticklabels([f'{100.0 / s:.1f}%' for s in steps])
    ax.legend(loc='best', ncol=2)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0.5, 1.0])

    plt.tight_layout()
    return fig, df


def plot_connectivity_statistics(conn_type_results, steps):
    """Plot connection density and sample sizes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Connection probabilities
    conn_types = ['E→E', 'E→I', 'I→E', 'I→I']
    colors_palette = plt.cm.Set2(np.linspace(0, 1, len(conn_types)))

    for conn_type, color in zip(conn_types, colors_palette):
        probs = [conn_type_results[step][conn_type]['n_connections'] /
                 conn_type_results[step][conn_type]['n_possible']
                 for step in steps]
        axes[0].plot(steps, probs, marker='o', linewidth=2,
                     markersize=8, label=conn_type, color=color)

    axes[0].set_xlabel('Subsampling Step', fontsize=12)
    axes[0].set_ylabel('Connection Probability', fontsize=12)
    axes[0].set_title('Connection Density by Type', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')

    # Sample sizes
    for conn_type, color in zip(conn_types, colors_palette):
        counts = [conn_type_results[step][conn_type]['n_connections']
                  for step in steps]
        axes[1].plot(steps, counts, marker='s', linewidth=2,
                     markersize=8, label=conn_type, color=color)

    axes[1].set_xlabel('Subsampling Step', fontsize=12)
    axes[1].set_ylabel('Number of True Connections', fontsize=12)
    axes[1].set_title('Sample Size by Connection Type', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')

    plt.tight_layout()
    return fig


# ============================================================================
# Generate All Plots
# ============================================================================

# Plot 1: Overall comparison
fig1 = plot_roc_comparison(steps, fpr, tpr, roc_auc,
                           fpr_exc, tpr_exc, roc_auc_exc,
                           fpr_inh, tpr_inh, roc_auc_inh)
plt.savefig(figures_directory / 'roc_comparison_overall.png', dpi=300, bbox_inches='tight')

# Plot 2: Connection type ROC curves
fig2 = plot_connection_type_roc(conn_type_results, steps)
plt.savefig(figures_directory / 'roc_connection_types.png', dpi=300, bbox_inches='tight')

# Plot 3: AUC summary
fig3, auc_summary_df = plot_auc_summary(conn_type_results, steps,
                                        roc_auc, roc_auc_exc, roc_auc_inh)
plt.savefig(figures_directory / 'auc_summary.png', dpi=300, bbox_inches='tight')

# Plot 4: Connectivity statistics
fig4 = plot_connectivity_statistics(conn_type_results, steps)
plt.savefig(figures_directory / 'connectivity_statistics.png', dpi=300, bbox_inches='tight')

# Save summary statistics
auc_summary_df.to_csv(figures_directory / 'auc_summary.csv', index=False)

print("\n" + "=" * 70)
print("Analysis Complete!")
print("=" * 70)
print(f"\nResults saved to: {figures_directory}")
print("\nGenerated files:")
print("  - roc_comparison_overall.png")
print("  - roc_connection_types.png")
print("  - auc_summary.png")
print("  - connectivity_statistics.png")
print("  - auc_summary.csv")
print("\n" + "=" * 70)

plt.show()