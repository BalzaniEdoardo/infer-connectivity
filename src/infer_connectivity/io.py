import re
import warnings
from pathlib import Path
from typing import Union

import numpy as np
from nemos.io.io import (
    MODEL_REGISTRY,
    _apply_custom_map,
    _expand_user_keys,
    _get_invalid_mappings,
    _set_fit_params,
    _split_model_params,
    _suggest_keys,
    _unflatten_dict,
)

from .ei_glm import GLMEI

MODEL_REGISTRY.update({"infer_connectivity.ei_glm.GLMEI": GLMEI})


def load_model(filename: Union[str, Path], mapping_dict: dict = None):
    """
    Load a previously saved nemos model from a .npz file.

    This will read the model parameters from the specified file and instantiate
    the model class with those parameters. It allows for custom mapping of
    attribute names to their actual objects using a mapping dictionary.

    Parameters
    ----------
    filename :
        Path to the saved .npz file.
    mapping_dict :
        Optional dictionary to map custom attribute names to their actual objects.

    Returns
    -------
    model :
        An instance of the model class with the loaded parameters.

    Examples
    --------
    >>> import nemos as nmo
    >>> # Create a GLM model with specified parameters
    >>> solver_args = {"stepsize": 0.1, "maxiter": 1000, "tol": 1e-6}
    >>> model = nmo.glm.GLM(
    ...     regularizer="Ridge",
    ...     regularizer_strength=0.1,
    ...     observation_model="Gamma",
    ...     solver_name="BFGS",
    ...     solver_kwargs=solver_args,
    ... )
    >>> for key, value in model.get_params().items():
    ...     print(f"{key}: {value}")
    observation_model__inverse_link_function: <function one_over_x at ...>
    observation_model: GammaObservations(inverse_link_function=one_over_x)
    regularizer: Ridge()
    regularizer_strength: 0.1
    solver_kwargs: {'stepsize': 0.1, 'maxiter': 1000, 'tol': 1e-06}
    solver_name: BFGS
    >>> # Save the model parameters to a file
    >>> model.save_params("model_params.npz")
    >>> # Load the model from the saved file
    >>> model = nmo.load_model("model_params.npz")
    >>> # Model has the same parameters before and after load
    >>> for key, value in model.get_params().items():
    ...     print(f"{key}: {value}")
    observation_model__inverse_link_function: <function one_over_x at ...>
    observation_model: GammaObservations(inverse_link_function=one_over_x)
    regularizer: Ridge()
    regularizer_strength: 0.1
    solver_kwargs: {'stepsize': 0.1, 'maxiter': 1000, 'tol': 1e-06}
    solver_name: BFGS

    >>> # Loading a custom inverse link function
    >>> obs = nmo.observation_models.PoissonObservations(
    ...     inverse_link_function=lambda x: x**2
    ... )
    >>> model = nmo.glm.GLM(observation_model=obs)
    >>> model.save_params("model_params.npz")
    >>> # Provide a mapping for the custom link function when loading.
    >>> mapping_dict = {
    ...     "observation_model__inverse_link_function": lambda x: x**2,
    ... }
    >>> loaded_model = nmo.load_model("model_params.npz", mapping_dict=mapping_dict)
    >>> # Now the loaded model will have the updated solver_name and solver_kwargs
    >>> for key, value in loaded_model.get_params().items():
    ...     print(f"{key}: {value}")
    observation_model__inverse_link_function: <function <lambda> at ...>
    observation_model: PoissonObservations(inverse_link_function=<lambda>)
    regularizer: UnRegularized()
    regularizer_strength: None
    solver_kwargs: {}
    solver_name: GradientDescent
    """
    # load the model from a .npz file
    filename = Path(filename)
    data = np.load(filename, allow_pickle=False)

    invalid_keys = _get_invalid_mappings(mapping_dict)
    if len(invalid_keys) > 0:
        raise ValueError(
            "Invalid map parameter types detected. "
            f"The following parameters are non mappable:\n\t{invalid_keys}\n"
            "Only callables and classes can be mapped."
        )

    flat_map_dict = (
        {}
        if mapping_dict is None
        else {_expand_user_keys(k, data): v for k, v in mapping_dict.items()}
    )

    # check for keys that are not in the parameters
    if mapping_dict is not None:
        not_available = [
            key_user
            for key_expanded, key_user in zip(flat_map_dict.keys(), mapping_dict.keys())
            if key_expanded not in data.keys()
        ]
        available_keys = [re.sub("(__class|__params)", "", key) for key in data.keys()]
        suggested_pairs = _suggest_keys(not_available, available_keys)
        suggestions = "".join(
            [
                (
                    f"\t- '{provided}', did you mean '{suggested}'?\n"
                    if suggested is not None
                    else f"\t- '{provided}'\n"
                )
                for provided, suggested in suggested_pairs
            ]
        )
        if len(not_available) > 0:
            raise ValueError(
                "The following keys in your mapping do not match any parameters in the loaded model:\n\n"
                f"{suggestions}\n"
                "Please double-check your mapping dictionary."
            )

    # Unflatten the dictionary to restore the original structure
    saved_params = _unflatten_dict(data, flat_map_dict)

    # "save_metadata" is used to store versions of Nemos and Jax, not needed for loading
    saved_params.pop("save_metadata")

    # if any value from saved_params is a key in mapping_dict,
    # replace it with the corresponding value from mapping_dict
    saved_params, updated_keys = _apply_custom_map(saved_params)

    if len(updated_keys) > 0:
        warnings.warn(
            f"The following keys have been replaced in the model parameters: {updated_keys}.",
            UserWarning,
        )

    # Extract the model class from the saved attributes
    model_name = str(saved_params.pop("model_class"))
    model_class = MODEL_REGISTRY[model_name]

    config_params, fit_params = _split_model_params(saved_params, model_class)

    # Create the model instance
    try:
        model = model_class(**config_params)
    except Exception:
        raise ValueError(
            f"Failed to instantiate model class '{model_name}' with parameters: {config_params}. "
            f"Use `nmo.inspect_npz('{filename}')` to inspect the saved object."
        )

    # Set the rest of the parameters as attributes if recognized
    _set_fit_params(model, fit_params, filename)

    return model
