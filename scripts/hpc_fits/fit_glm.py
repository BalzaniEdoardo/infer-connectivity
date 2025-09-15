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

jax.config.update("jax_enable_x64", True)

conf_path = pathlib.Path(sys.argv[1])
dataset_path = pathlib.Path(sys.argv[2])
output_dir = pathlib.Path(sys.argv[3])

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
binsize = 0.0003
history_window = 0.014
window_size = int(history_window / binsize)
n_basis_funcs = 4


# Fit Hyperparameters
solver_name = "LBFGS" if "Lasso" not in regularizer else None
solver_kwargs = {"tol": 10**-12}
param_grid = {
    "regularizer_strength": (
        np.geomspace(10**-8, 10**-3, 8) if regularizer != "UnRegularized" else None
    )
}

logging.log(level=logging.INFO, msg="Set up fit hyperparameters.")


# Count & transform & fit
counts = spikes_tsgroup.count(binsize)
logging.log(level=logging.INFO, msg="Counted spikes.")

basis_cls = getattr(nmo, basis_cls_name)
basis = basis_cls(n_basis_funcs, window_size)

logging.log(level=logging.INFO, msg="Defined basis.")

X = basis.compute_features(counts)

logging.log(level=logging.INFO, msg="Computed design matrix.")

model = nmo.glm.GLM(
    observation_model=observation_model,
    regularizer=regularizer,
    solver_name=solver_name,
    solver_kwargs=solver_kwargs,
)

logging.log(level=logging.INFO, msg="Start model fitting...")
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
    "dataset_path": dataset_path,
    "neuron_fit": neuron_fit,
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
cls.best_estimator_.save(model_path)

logging.log(level=logging.INFO, msg=f"Saved CV results and best model to {output_dir}")
