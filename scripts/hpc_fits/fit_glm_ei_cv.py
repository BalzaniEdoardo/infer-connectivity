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
import itertools


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
    dataset_path = pathlib.Path("../../simulations/sonica-sept-25-2025/spikes60s_bg.pckl")
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

logging.log(level=logging.INFO, msg="Extracted configs.")

# Load simulations
spikes = pickle.load(open(dataset_path, "rb"))
spikes_tsgroup = nap.TsGroup(
    {n: nap.Ts(np.array(spikes[n]) / 1000) for n in range(len(spikes))}
)

logging.log(level=logging.INFO, msg="Loaded simulated spikes into pynapple.")

# Parameters for processing
binsize = conf_dict["binsize"]
history_window = conf_dict["history_window"]
window_size = int(history_window / binsize)
n_basis_funcs = conf_dict["n_basis_funcs"]


# Fit Hyperparameters
solver_name = "LBFGS" if "Lasso" not in regularizer else None
solver_kwargs = {"tol": 10**-12}


logging.log(level=logging.INFO, msg="Set up fit hyperparameters.")


# Count & transform & fit
counts = spikes_tsgroup.count(binsize)
logging.log(level=logging.INFO, msg="Counted spikes.")

basis_cls = getattr(nmo.basis, basis_cls_name)
basis = basis_cls(n_basis_funcs, window_size)

logging.log(level=logging.INFO, msg="Defined basis.")

X = basis.compute_features(counts)

# # cut to 5% data
# use_n = int(X.shape[0] * 0.05)
# X = X[:use_n]
# counts = counts[:use_n]

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
if regularizer in ["GroupLasso", "GroupLassoMultiRegularization"]:
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


logging.log(level=logging.INFO, msg="Start model fitting...")
n_coeff = X.shape[1]

inhibitory_neu_id = jax.numpy.asarray(conf_dict["inhibitory_neuron_id"], dtype=int)
excitatory_neu_id = jax.numpy.asarray(list(set(range(counts.shape[1])).difference(inhibitory_neu_id.tolist())), dtype=int)
one_dim_param = jax.numpy.geomspace(10**-8, 10**-3, 8)

reg_str = jax.numpy.ones((one_dim_param.shape[0] ** 2, counts.shape[1]), dtype=float)
for i, (reg1, reg2) in enumerate(itertools.product(one_dim_param, one_dim_param)):
    reg_str = reg_str.at[i, inhibitory_neu_id].set(reg1)
    reg_str = reg_str.at[i, excitatory_neu_id].set(reg2)


if regularizer in ["GroupLassoMultiRegularization"]:
    pass
else:
    reg_str = jax.numpy.repeat(reg_str, basis.n_basis_funcs, axis=1)


param_grid = {
    "regularizer_strength": (
         list(reg_str) if regularizer != "UnRegularized" else [None]
    )
}

cls = GridSearchCV(model, param_grid, cv=5)
cls.fit(X, counts[:, neuron_fit])


logging.log(level=logging.INFO, msg="Model fit complete.")

# Extract numpy-compatible CV results
cv_data = {
    "best_score": cls.best_score_,
    "best_index": cls.best_index_,
    "mean_test_score": cls.cv_results_["mean_test_score"],
    "std_test_score": cls.cv_results_["std_test_score"],
    "mean_fit_time": cls.cv_results_["mean_fit_time"],
    "std_fit_time": cls.cv_results_["std_fit_time"],
    "rank_test_score": cls.cv_results_["rank_test_score"],
}

# Add parameter values if they're numeric
if "param_regularizer_strength" in cls.cv_results_:
    cv_data["param_regularizer_strength"] = np.array(
        cls.cv_results_["param_regularizer_strength"]
    )


metadata = {
    "config": conf_dict,
    "dataset_path": dataset_path.as_posix(),
    "binsize": binsize,
    "history_window": history_window,
    "n_basis_funcs": n_basis_funcs,
}


cv_path = (
    output_dir
    / f"cv_results_dataset_{dataset_path.stem}_neuron_{neuron_fit}_config_{conf_path.stem}.npz"
)
np.savez(cv_path, **cv_data)

metadata_path = (
    output_dir
    / f"metadata_{dataset_path.stem}_neuron_{neuron_fit}_config_{conf_path.stem}.json"
)

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

model_path = (
    output_dir
    / f"best_model_{dataset_path.stem}_neuron_{neuron_fit}_config_{conf_path.stem}.npz"
)
best_estimator = cls.best_estimator_
best_estimator.solver_kwargs.pop("projection", None)
best_estimator.save_params(model_path)

logging.log(level=logging.INFO, msg=f"Saved CV results and best model to {output_dir}")
