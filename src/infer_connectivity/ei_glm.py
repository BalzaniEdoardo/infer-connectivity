from copy import deepcopy

import jax
import nemos as nmo


class GLMEI(nmo.glm.GLM):
    def __init__(
        self,
        # With python 3.11 Literal[*AVAILABLE_OBSERVATION_MODELS] will be allowed.
        # Replace this manual list after dropping support for 3.10?
        observation_model="Poisson",
        regularizer=None,
        regularizer_strength=None,
        solver_name=None,
        solver_kwargs=None,
    ):
        if solver_name is None:
            solver_name = "ProjectedGradient"

        super().__init__(
            observation_model=observation_model,
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )

    @property
    def solver_name(self) -> str:
        return self._solver_name

    @solver_name.setter
    def solver_name(self, solver_name):
        self._solver_name = solver_name

    def instantiate_solver(self, *args, solver_kwargs=None) -> "GLMEI":
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

        def solver_run(init_params, X=None, y=None):
            return solver.run(
                init_params, hyperparams_proj=None, X=X, y=y, **solver_run_kwargs
            )

        def solver_update(params, state, *run_args, **run_kwargs):
            return solver.update(
                params,
                state,
                hyperparams_proj=None,
                *args,
                *run_args,
                **solver_update_kwargs,
                **run_kwargs,
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
            -jax.nn.relu(-w),  # Project to non-positive
            jax.nn.relu(w),  # Project to non-negative
        ),
        w,
    )
    return wp, b
