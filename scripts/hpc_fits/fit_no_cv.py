import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'  # Use only 75% of GPU memory


import json
import logging
import pathlib
import pickle
import sys

import jax
import nemos as nmo
import numpy as np
import pynapple as nap
from sklearn.model_selection import GridSearchCV
from infer_connectivity import GLMEI, projection_ei
from infer_connectivity.roc_utils import compute_filters, compute_roc_curve


# Basic setup that prints to terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # This ensures output goes to terminal
    ]
)

logger = logging.getLogger(__name__)
logger.info("Check logger...")

jax.config.update("jax_enable_x64", True)

try:
    conf_path = pathlib.Path(sys.argv[1])
    dataset_path = pathlib.Path(sys.argv[2])
    output_dir = pathlib.Path(sys.argv[3])
except IndexError:
    conf_path = pathlib.Path("../configs/sonica-sept-25-2025/Lasso_Bernoulli_RaisedCosineLogConv_0_True.json")
    dataset_path = pathlib.Path("../../simulations/sonica-oct-8-2025-400-seconds/spikes_400_neur_400_sec.pckl")
    output_dir = pathlib.Path("../outputs")

logging.log(level=logging.INFO, msg=f"Fitting dataset: '{dataset_path}'.")

# Load configs
with open(conf_path, "r") as f:
    conf_dict = json.load(f)

logging.log(level=logging.INFO, msg="Loaded config dictionary.")

observation_model = conf_dict["observation_model"]
regularizer = conf_dict["regularizer"]
basis_cls_name = conf_dict["basis_cls_name"]
neuron_fit = conf_dict["neuron_id"]
connectivity_path = conf_dict["connectivity_path"]

logging.log(level=logging.INFO, msg="Extracted configs.")

# Load simulations
spikes = pickle.load(open(dataset_path, "rb"))
spikes_tsgroup = nap.TsGroup(
    {n: nap.Ts(np.array(spikes[n]) / 1000) for n in range(len(spikes))}
)

logging.log(level=logging.INFO, msg="Loaded simulated spikes into pynapple.")

# Parameters for processing
binsize = 0.0003
history_window = 0.014
window_size = int(history_window / binsize)
n_basis_funcs = 4


# Fit Hyperparameters
solver_name = "LBFGS" if "Lasso" not in regularizer else None
solver_kwargs = {"tol": 10**-12}
param_grid = {
    "regularizer_strength": (
        np.geomspace(10**-8, 10**-3, 15) if regularizer != "UnRegularized" else [None]
    )
}

logging.log(level=logging.INFO, msg="Set up fit hyperparameters.")


# Count & transform & fit
counts = spikes_tsgroup.count(binsize)
logging.log(level=logging.INFO, msg="Counted spikes.")

basis_cls = getattr(nmo.basis, basis_cls_name)
basis = basis_cls(n_basis_funcs, window_size)

logging.log(level=logging.INFO, msg="Defined basis.")

X = basis.compute_features(counts)

# # cut to 5% data
use_n = int(X.shape[0] * 0.05)
X = X[:use_n]
counts = counts[:use_n]

logging.log(level=logging.INFO, msg="Computed design matrix.")

if conf_dict["enforce_ei"]:
    model_cls = GLMEI
else:
    model_cls = nmo.glm.GLM

model = model_cls(
    observation_model=observation_model,
    regularizer=regularizer,
    solver_name=solver_name,
    solver_kwargs=solver_kwargs,
)

# set up mask for group lasso
if regularizer == "GroupLasso":
    logging.log(level=logging.INFO, msg="Preparing mask for group lasso.")
    mask = np.eye(len(spikes_tsgroup), dtype=float)
    mask = np.repeat(mask, basis.n_basis_funcs, axis=1)
    assert mask.shape[1] == X.shape[1], "Mask and X shape are not matching"
    model.regularizer.mask = mask
    logging.log(level=logging.INFO, msg="Mask setup successfully.")

if conf_dict["enforce_ei"]:
    inhibitory_neu_id = jax.numpy.asarray(conf_dict["inhibitory_neuron_id"])
    inhib_mask = jax.numpy.zeros(counts.shape[1], dtype=bool)
    inhib_mask = inhib_mask.at[inhibitory_neu_id].set(True)
    inhib_mask = jax.numpy.repeat(inhib_mask, basis.n_basis_funcs)
    enforce_ei_proj = lambda x, hyperparams=None: projection_ei(x, inhib_mask, hyperparams=hyperparams)
    if isinstance(model, GLMEI):
        model._solver_name = "ProjectedGradient"
    model.solver_kwargs.update({"projection": enforce_ei_proj})

logging.log(level=logging.INFO, msg="Extract true connectivity")
true_conn = np.load(connectivity_path)
true_conn += np.eye(true_conn.shape[0], dtype=int)

logging.log(level=logging.INFO, msg="Start model fitting...")

# Storage for results
results = {
    'regularizer_strengths': [],
    'roc_auc_scores': [],
    'ap_scores': [],
    'best_f1_scores': [],
    'model_coefs': [],
    'filters': [],
    'pred_conns': []
}

best_roc_auc = -np.inf
best_model = None
best_reg_strength = None

for reg_str in param_grid["regularizer_strength"]:
    logging.log(level=logging.INFO, msg=f"Fitting with regularizer_strength={reg_str}")

    current_model = model.__sklearn_clone__()
    current_model.set_params(regularizer_strength=reg_str)
    current_model.fit(X, counts[:, neuron_fit])

    # Compute response filters, shape [input, 1, time]
    filters = compute_filters(current_model.coef_, basis, window_size)

    # Compute ROC metrics
    fpr, tpr, roc_auc, ap, pred_conn, best_f1 = compute_roc_curve(true_conn[:, neuron_fit], filters)

    # Store results
    results['regularizer_strengths'].append(reg_str)
    results['roc_auc_scores'].append(roc_auc)
    results['ap_scores'].append(ap)
    results['best_f1_scores'].append(best_f1)
    results['model_coefs'].append(current_model.coef_.copy())
    results['filters'].append(filters.copy())
    results['pred_conns'].append(pred_conn.copy())

    logging.log(level=logging.INFO,
                msg=f"reg_strength={reg_str:.2e}, ROC_AUC={roc_auc:.4f}, AP={ap:.4f}, best_F1={best_f1:.4f}")

    # Track best model
    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        best_model = current_model
        best_reg_strength = reg_str

logging.log(level=logging.INFO, msg="Model fit complete.")
logging.log(level=logging.INFO,
            msg=f"Best regularizer strength: {best_reg_strength:.2e} with ROC_AUC: {best_roc_auc:.4f}")

# Save all results
results_path = (
        output_dir
        / f"reg_sweep_results_dataset_{dataset_path.stem}_neuron_{neuron_fit}_config_{conf_path.stem}.npz"
)

np.savez(
    results_path,
    regularizer_strengths=np.array(results['regularizer_strengths']),
    roc_auc_scores=np.array(results['roc_auc_scores']),
    ap_scores=np.array(results['ap_scores']),
    best_f1_scores=np.array(results['best_f1_scores']),
    model_coefs=np.array(results['model_coefs']),
    filters=np.array(results['filters']),
    pred_conns=np.array(results['pred_conns']),
    best_reg_strength=best_reg_strength,
    best_roc_auc=best_roc_auc,
    neuron_id=neuron_fit
)

logging.log(level=logging.INFO, msg=f"Saved regularization sweep results to {results_path}")

# Save best model separately
best_model_path = (
        output_dir
        / f"best_model_dataset_{dataset_path.stem}_neuron_{neuron_fit}_config_{conf_path.stem}.npz"
)
best_model_copy = best_model.__sklearn_clone__()
best_model_copy.set_params(**best_model.get_params())
best_model_copy.coef_ = best_model.coef_.copy()
best_model_copy.intercept_ = best_model.intercept_.copy() if hasattr(best_model, 'intercept_') else None
if hasattr(best_model_copy, 'solver_kwargs'):
    best_model_copy.solver_kwargs.pop("projection", None)
best_model_copy.save_params(best_model_path)

logging.log(level=logging.INFO, msg=f"Saved best model to {best_model_path}")

# Save metadata
metadata = {
    "config": conf_dict,
    "dataset_path": dataset_path.as_posix(),
    "binsize": binsize,
    "history_window": history_window,
    "n_basis_funcs": n_basis_funcs,
    "neuron_fit": neuron_fit,
    "best_reg_strength": float(best_reg_strength) if best_reg_strength is not None else None,
    "best_roc_auc": float(best_roc_auc),
}

metadata_path = (
        output_dir
        / f"metadata_dataset_{dataset_path.stem}_neuron_{neuron_fit}_config_{conf_path.stem}.json"
)

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

logging.log(level=logging.INFO, msg=f"Saved metadata to {metadata_path}")