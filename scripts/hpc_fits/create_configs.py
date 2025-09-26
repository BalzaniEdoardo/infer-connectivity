"""Configs for GLM fitting."""

from itertools import product
import json
import pathlib

conf_dirname = "sonica-sept-25-2026"
base_dir = pathlib.Path("/mnt/ceph/users/ebalzani/synaptic_connectivity/configs") / conf_dirname

conf_path = pathlib.Path("configs")
conf_path.mkdir(exist_ok=True, parents=True)

regularizers = ["UnRegularized", "Ridge", "Lasso", "GroupLasso"]
observation_model = ["Bernoulli", "Poisson"]
basis_class_name = ["RaisedCosineLogConv"]
neuron_id = range(100)

pars = product(regularizers, observation_model, basis_class_name, neuron_id)
for reg, obs, bas, neu in pars:
    conf_dict = dict(
        observation_model=obs,
        regularizer=reg,
        basis_cls_name=bas,
        neuron_id=neu,
    )
    with open(base_dir/f"{reg}_{obs}_{bas}_{neu}.json", "w") as f:
        json.dump(conf_dict, f)
