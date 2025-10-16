import os
import pathlib
import json
import warnings

import nemos as nmo
import re

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from matplotlib.colors import ListedColormap, to_rgb
from nemos.regularizer import GroupLasso

from infer_connectivity import load_model
from infer_connectivity.roc_utils import compute_filters, compute_roc_curve
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

base_dir = Path("/Users/ebalzani/Code/infer-connectivity/infer-connectivity/")
graph_file = base_dir / "simulations/sonica-sept-25-2025/graph0-sonica-sept-26-2026.graphml"
fit_id = "sonica-oct-8-2025-400-seconds"
output_directory = base_dir / "scripts" / "output" / fit_id
config_directory = base_dir / "scripts" / "configs" / fit_id
simulation_directory = base_dir / "simulations" / fit_id


# get conf
conf = None
for conf in config_directory.iterdir():
    if conf.suffix == ".json":
        break

def extract_coef_single_neu_glm(output_dir: str | pathlib.Path, pattern=None):
    output_dir = pathlib.Path(output_dir)
    if pattern is None:
        pattern = re.compile(r'best_.*neuron_(\d+).*\.npz')

    neu_ids = []
    for fh in output_dir.iterdir():
        found = re.search(pattern, fh.name)
        if found is not None:
            neu_ids.append(int(found.group(1)))
    neu_ids = sorted(neu_ids)

    coef = None
    for fh in output_dir.iterdir():
        found = re.search(pattern, fh.name)
        if found is not None:
            neu = int(found.group(1))
            model = load_model(fh)
            if coef is None:
                coef = np.zeros((*model.coef_.shape, len(neu_ids)))
            coef[..., neu_ids.index(neu)] = model.coef_
    return np.array(neu_ids), coef


# Parameters for processing
if conf:
    with open(conf, "r") as fh:
        conf_dict = json.load(fh)
else:
    conf_dict = {}

# Note that this used to be hardcoded.
if not all(key in conf_dict for key in ["binsize", "history_window", "n_basis_funcs"]):
    warnings.warn("Parameters where hard-coded, please make sure that the defaults are correct.", UserWarning)


# default used until Oct 16 2025, after that should be in the config
binsize = conf_dict.get("binsize", 0.0003)
history_window = conf_dict.get("history_window", 0.014)
window_size = int(history_window / binsize)
n_basis_funcs = conf_dict.get("n_basis_funcs", 4)
# this was always in config
basis_cls_name = conf_dict["basis_cls_name"]


basis_cls = getattr(nmo.basis, basis_cls_name)
basis = basis_cls(n_basis_funcs, window_size=window_size)
neu_id, coef_pop = extract_coef_single_neu_glm(output_directory)
filters = compute_filters(coef_pop, basis)

# extract connectivity
connectivity_path = simulation_directory / pathlib.Path(conf_dict["connectivity_path"]).name
true_conn = np.load(connectivity_path)
true_conn += np.eye(true_conn.shape[0], dtype=int)


fpr, tpr, roc_auc, ap, pred_conn, best_f1 = compute_roc_curve(true_conn, filters)



# # AUROC and precision plot
fig, axs = plt.subplots(1, 1, figsize=(10, 8))
# ROC curves'
axs.plot(
    fpr, tpr, label=f"Lasso-Bernoulli (AUC = {roc_auc:.2f})"
)
axs.plot([0, 1], [0, 1], "k--")
axs.set_xlabel("False Positives")
axs.set_ylabel("True Positives")
axs.legend()
axs.set_title("ROC Curves")

plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 3 ,1)
plt.title("True connectivity")
plt.pcolormesh(true_conn)
plt.subplot(1, 3 ,2)
plt.title("Predicted connectivity")
plt.pcolormesh(pred_conn.reshape(len(neu_id), -1))
plt.subplot(1, 3 ,3)
delta = (true_conn - pred_conn.reshape(len(neu_id), -1)).astype(float)
delta[delta == 0] = np.nan
plt.title("Delta")
plt.pcolormesh(delta, cmap="Spectral")
plt.tight_layout()

plt.show()