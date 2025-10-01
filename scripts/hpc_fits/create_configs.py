"""Configs for GLM fitting."""

from itertools import product
import json
import pathlib

conf_dirname = "ei-cv-sonica-sept-25-2025"
base_dir = pathlib.Path("/mnt/ceph/users/ebalzani/synaptic_connectivity/configs") / conf_dirname
if not base_dir.exists():
    print("base_dir NOT FOUND")
    base_dir = pathlib.Path("../configs") / conf_dirname

base_dir.mkdir(exist_ok=True, parents=True)

enforce_ei = [False]
regularizers = ["UnRegularized", "Ridge", "Lasso", "GroupLasso"]
observation_model = ["Bernoulli", "Poisson"]
basis_class_name = ["RaisedCosineLogConv"]
neuron_id = range(400)
inhibitory_neu_id = list(range(300, 400))

pars = product(regularizers, observation_model, basis_class_name, neuron_id, enforce_ei)
for reg, obs, bas, neu, ei in pars:
    conf_dict = dict(
        observation_model=obs,
        regularizer=reg,
        basis_cls_name=bas,
        neuron_id=neu,
        inhibitory_neuron_id=inhibitory_neu_id,
        enforce_ei = ei,
    )
    with open(base_dir/f"{reg}_{obs}_{bas}_{neu}_{ei}.json", "w") as f:
        json.dump(conf_dict, f)
