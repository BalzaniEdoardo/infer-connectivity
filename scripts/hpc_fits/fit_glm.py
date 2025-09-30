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
from copy import deepcopy

class GLMEI(nmo.glm.GLM):
    def __init__(
            self,
            # With python 3.11 Literal[*AVAILABLE_OBSERVATION_MODELS] will be allowed.
            # Replace this manual list after dropping support for 3.10?
            observation_model = "Poisson",
            regularizer = None,
            regularizer_strength = None,
            solver_name = None,
            solver_kwargs = None,
    ):
        super().__init__(
            observation_model = observation_model,
            regularizer = regularizer,
            regularizer_strength = regularizer_strength,
            solver_name = solver_name,
            solver_kwargs = solver_kwargs,
        )
    @property
    def solver_name(self) -> str:
        return self._solver_name

    @solver_name.setter
    def solver_name(self, solver_name):
        self._solver_name = solver_name

    def instantiate_solver(
        self, *args, solver_kwargs = None
    ) -> "GLMEI":
        """
        Instantiate the solver with the provided loss function.

        Instantiate the solver with the provided loss function, and store callable functions
        that initialize the solver state, update the model parameters, and run the optimization
        as attributes.

        This method creates a solver instance from nemos.solvers or the jaxopt library, tailored to
        the specific loss function and regularization approach defined by the Regularizer instance.
        It also handles the proximal operator if required for the optimization method. The returned
        functions are directly usable in optimization loops, simplifying the syntax by pre-setting
        common arguments like regularization strength and other hyperparameters.

        Parameters
        ----------
        *args:
            Positional arguments for the jaxopt `solver.run` method, e.g. the regularizing
            strength for proximal gradient methods.
        solver_kwargs:
            Optional dictionary with the solver kwargs.
            If nothing is provided, it defaults to self.solver_kwargs.

        Returns
        -------
        :
            The instance itself for method chaining.
        """
        # only use penalized loss if not using proximal gradient descent
        # In proximal method you must use the unpenalized loss independently
        # of what regularizer you are using.
        if self.solver_name not in ("ProximalGradient", "ProxSVRG"):
            loss = self.regularizer.penalized_loss(
                self._predict_and_compute_loss, self.regularizer_strength
            )
        else:
            loss = self._predict_and_compute_loss

        if solver_kwargs is None:
            # copy dictionary of kwargs to avoid modifying user settings
            solver_kwargs = deepcopy(self.solver_kwargs)

        # check that the loss is Callable
        nmo.utils.assert_is_callable(loss, "loss")

        # some parsing to make sure solver gets instantiated properly
        if self.solver_name in ("ProximalGradient", "ProxSVRG"):
            if "prox" in self.solver_kwargs:
                raise ValueError(
                    "Proximal operator specification is not permitted. "
                    "The proximal operator is automatically determined based on the selected regularizer. "
                    "Please remove the 'prox' argument from the `solver_kwargs` "
                )

            solver_kwargs.update(prox=self.regularizer.get_proximal_operator())
            # add self.regularizer_strength to args
            args += (self.regularizer_strength,)

        (
            solver_run_kwargs,
            solver_init_state_kwargs,
            solver_update_kwargs,
            solver_init_kwargs,
        ) = self._inspect_solver_kwargs(solver_kwargs)

        # instantiate the solver
        solver = self._get_solver_class(self.solver_name)(
            fun=loss, **solver_init_kwargs
        )

        self._solver_loss_fun = loss

        def solver_run(
            init_params, X=None, y=None
        ):
            return solver.run(init_params, hyperparams_proj=None, X=X, y=y, **solver_run_kwargs)

        def solver_update(params, state, *run_args, **run_kwargs):
            return solver.update(
                params, state, hyperparams_proj=None, *args, *run_args, **solver_update_kwargs, **run_kwargs
            )

        def solver_init_state(params, *run_args, **run_kwargs):
            return solver.init_state(
                params,
                *run_args,
                **run_kwargs,
                **solver_init_state_kwargs,
            )

        self._solver = solver
        self._solver_init_state = solver_init_state
        self._solver_update = solver_update
        self._solver_run = solver_run
        return self


def projection_ei(x, inhib_mask, hyperparams=None):
    del hyperparams
    w, b = x
    wp = jax.tree_util.tree_map(
        lambda z: jax.numpy.where(
            inhib_mask,
            -jax.nn.relu(-w),   # Project to non-positive
            jax.nn.relu(w)      # Project to non-negative
        ),
        w
    )
    return wp, b


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
binsize = 0.0003
history_window = 0.014
window_size = int(history_window / binsize)
n_basis_funcs = 4


# Fit Hyperparameters
solver_name = "LBFGS" if "Lasso" not in regularizer else None
solver_kwargs = {"tol": 10**-12}
param_grid = {
    "regularizer_strength": (
        np.geomspace(10**-8, 10**-3, 8) if regularizer != "UnRegularized" else [None]
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
