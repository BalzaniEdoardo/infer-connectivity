import os
import nemos as nmo

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from matplotlib.colors import ListedColormap, to_rgb
from nemos.regularizer import GroupLasso

from infer_connectivity.CreateNetwork import save_connectivity_matrix
import networkx as nx

from sklearn.metrics import auc, average_precision_score, f1_score, roc_curve
from pathlib import Path
GRADED_COLOR_LIST = [
    "navy",
    "blue",
    "royalblue",
    "cornflowerblue",
    "skyblue",
    "lightblue",
    "aquamarine",
    "mediumseagreen",
    "limegreen",
    "yellowgreen",
    "gold",
    "orange",
    "darkorange",
    "tomato",
    "orangered",
    "red",
    "crimson",
    "deeppink",
    "magenta",
]


def compute_roc_curve(true_conn, filters):
    abs_argmax = np.argmax(np.abs(filters), axis=1)
    peak_filt = np.array(
        np.take_along_axis(filters, abs_argmax[:, np.newaxis], axis=1).squeeze(axis=1)
    )
    scores = np.abs(peak_filt)
    fpr, tpr, roc_thresh = roc_curve(true_conn, scores)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(true_conn, scores)

    f1s = []
    for t in roc_thresh:
        preds = (scores >= t).astype(int)
        f1s.append(f1_score(true_conn, preds))
    best_t = roc_thresh[np.argmax(f1s)]
    pred_conn = (scores >= best_t).astype(int)

    return fpr, tpr, roc_auc, ap, pred_conn

def compute_filters(weights, basis, window_size):
    # weights = pop_model.coef_
    kernels = basis.evaluate_on_grid(window_size)[1]
    resp_filters = np.einsum("jki,tk->ijt", weights, kernels)
    return resp_filters


base_dir = Path("/Users/ebalzani/Code/infer-connectivity/infer-connectivity/")
graph_file = base_dir / "simulations/sonica-sept-25-2025/graph0-sonica-sept-26-2026.graphml"

# get true connectivity
conn_matrix_file = graph_file.parent / (graph_file.stem + "_connectivity.npy")
graph= nx.read_graphml(graph_file, int)
if not conn_matrix_file.exists():
    save_connectivity_matrix(graph, conn_matrix_file)
true_conn = np.load(conn_matrix_file)
true_conn += np.eye(true_conn.shape[0], dtype=int)

# get pop cooef
with open("/Users/ebalzani/Code/infer-connectivity/infer-connectivity/best_model_aggregate_coeff/sonica-sept-25-2025/pop_coefficients.pckl", "rb") as f:
    pop_coef_dict = pickle.load(f)

binsize = 0.0003
history_window = 0.014
window_size = int(history_window / binsize)

basis = nmo.basis.RaisedCosineLogConv(4, window_size=window_size)

results_roc = {}
for (reg, obs, ei), coeffs in pop_coef_dict.items():
    print(reg, obs, ei)
    filters = compute_filters(coeffs.reshape(400, -1, 400), basis, window_size)
    fpr, tpr, roc_auc, ap, pred_conn = compute_roc_curve(true_conn.reshape(filters.shape[0]**2,-1), filters.reshape(filters.shape[0]**2,-1))
    results_roc[reg, obs, ei] =  fpr, tpr, roc_auc, ap, pred_conn


# AUROC and precision plot
fig, axs = plt.subplots(1, 1, figsize=(10, 8))
for k, ((reg, obs, ei), (fpr, tpr, roc_auc, ap, pred_conn)) in enumerate(results_roc.items()):
    # ROC curves'
    ei_label = "-ei" if ei else ""
    axs[0].plot(
        fpr, tpr, label=f"{reg}-{obs}{ei_label}(AUC = {roc_auc:.2f})", c=GRADED_COLOR_LIST[k]
    )
axs[0].plot([0, 1], [0, 1], "k--")
axs[0].set_xlabel("False Positives")
axs[0].set_ylabel("True Positives")
axs[0].legend()
axs[0].set_title("ROC Curves")

plt.show()