import nemos as nmo
import infer_connectivity as ic
import pathlib
import itertools
import numpy as np
import pickle


regularizers = ["Ridge", "Lasso", "GroupLasso", "UnRegularized"]
observation_model = ["Bernoulli", "Poisson"]
basis_class_name = ["RaisedCosineLogConv"]
neuron_id = range(400)
n_basis_funcs = 4
enforce_ei = [False, True]

sim_dirname = "ei-cv-sonica-sept-25-2025"
pars = itertools.product(regularizers, observation_model, basis_class_name, neuron_id, enforce_ei)


base_dir = pathlib.Path("/mnt/ceph/users/ebalzani/synaptic_connectivity")
conf_dir = base_dir / "configs" / sim_dirname
dataset_path = base_dir / "simulations" / sim_dirname / "spikes60s_bg.pckl"
output_dir = base_dir/ "outputs" / sim_dirname

coeff_save_dir = base_dir/ "best_model_aggregate_coeff" / sim_dirname
coeff_save_dir.mkdir(exist_ok=True, parents=True)
coeff_save_path = coeff_save_dir / "pop_coefficients.npz"

pop_models = {}
for reg, obs, ei in itertools.product(regularizers, observation_model, enforce_ei):
    pop_models[reg, obs, ei] = np.zeros((len(neuron_id), len(neuron_id)*n_basis_funcs))

for reg, obs, bas, neu, ei in pars:
    if enforce_ei:
        conf_path = base_dir / f"{reg}_{obs}_{bas}_{neu}_{ei}.json"
        model_path = (
                output_dir
                / f"best_model_{dataset_path.stem}_neuron_{neu}_config_{conf_path.stem}.npz"
        )
    else:
        conf_path = base_dir / f"{reg}_{obs}_{bas}_{neu}.json"

        model_path = (
                output_dir
                / f"best_model_{dataset_path.stem}_neuron_{neu}_config_{conf_path.stem}.npz"
        )
    model = ic.load_model(model_path)
    pop_models[reg, obs, ei][neu] = model.coef_


np.savez(coeff_save_path, pop_models)




