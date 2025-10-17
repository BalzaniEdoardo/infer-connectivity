"""Configs for GLM fitting."""

from itertools import product
import json
import pathlib

conf_dirname = "sonica-oct-8-2025-400-seconds"
base_dir = pathlib.Path("/mnt/ceph/users/ebalzani/synaptic_connectivity/configs") / conf_dirname
connectivity_path = pathlib.Path("/mnt/ceph/users/ebalzani/synaptic_connectivity/simulations/") / conf_dirname


# Parameters for processing
binsize = 0.0003
history_window = 0.014
n_basis_funcs = 4



print("conf dirname:")
print(conf_dirname)
print("bas dirname:")
print(base_dir)
conn_file_name = None
for fhname in connectivity_path.iterdir():
    if fhname.suffix == ".npy" and fhname.name.startswith("graph"):
        conn_file_name = fhname
if conn_file_name is None:
    raise FileNotFoundError("Unable to locate connectivity matrix.")
else:
    connectivity_path = conn_file_name

if not base_dir.exists():
    print("base_dir NOT FOUND")
    base_dir = pathlib.Path("../configs") / conf_dirname

base_dir.mkdir(exist_ok=True, parents=True)

enforce_ei = [False]
regularizers = ["Lasso"]
observation_model = ["Bernoulli"]
basis_class_name = ["RaisedCosineLogConv"]
neuron_id = range(400)
inhibitory_neu_id = list(range(300, 400))
fit_neurons = []
for step in [2, 5, 10, 25]:
    fit_neurons.append((step, list(range(400)[::step])))

pars = product(regularizers, observation_model, basis_class_name, neuron_id, enforce_ei, fit_neurons)
for reg, obs, bas, neu, ei, fit_list in pars:
    step, neurons = fit_list
    conf_dict = dict(
        observation_model=obs,
        regularizer=reg,
        basis_cls_name=bas,
        neuron_id=neu,
        inhibitory_neuron_id=inhibitory_neu_id,
        enforce_ei = ei,
        connectivity_path=connectivity_path.as_posix(),
        binsize=binsize,
        history_window=history_window,
        n_basis_funcs=n_basis_funcs,
        fit_list=neurons,
    )
    with open(base_dir/f"{reg}_{obs}_{bas}_{neu}_{ei}_{step}.json", "w") as f:
        json.dump(conf_dict, f)

