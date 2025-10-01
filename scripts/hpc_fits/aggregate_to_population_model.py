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

sim_dirname = "sonica-sept-25-2025"
pars = itertools.product(regularizers, observation_model, basis_class_name, neuron_id, enforce_ei)


base_dir = pathlib.Path("/mnt/ceph/users/ebalzani/synaptic_connectivity")
conf_dir = base_dir / "configs" / sim_dirname
dataset_path = base_dir / "simulations" / sim_dirname / "spikes60s_bg.pckl"
output_dir = base_dir/ "outputs" / sim_dirname

coeff_save_dir = base_dir/ "best_model_aggregate_coeff" / sim_dirname
coeff_save_dir.mkdir(exist_ok=True, parents=True)
coeff_save_path = coeff_save_dir / "pop_coefficients.pckl"

pop_models = {}
for reg, obs, ei in itertools.product(regularizers, observation_model, enforce_ei):
    pop_models[reg, obs, ei] = np.zeros((len(neuron_id), len(neuron_id)*n_basis_funcs))

for reg, obs, bas, neu, ei in pars:
    print(reg, obs, neu, ei)
    if ei:
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
        print(conf_path)
    model = ic.load_model(model_path)
    pop_models[reg, obs, ei][neu] = model.coef_

with open(coeff_save_path, "wb") as fh:
    pickle.dump(pop_models, fh)




