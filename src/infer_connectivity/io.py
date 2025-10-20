import inspect
import re
import warnings
from pathlib import Path
from typing import Union

import numpy as np
from nemos.io.io import (
    AVAILABLE_OBSERVATION_MODELS,
    AVAILABLE_REGULARIZERS,
    MODEL_REGISTRY,
    _apply_custom_map,
    _get_invalid_mappings,
    _set_fit_params,
    _split_model_params,
    _suggest_keys,
    _unflatten_dict,
    _unflattened_user_map,
    get_user_keys_from_nested_dict,
    instantiate_observation_model,
    instantiate_regularizer,
)

from .ei_glm import GLMEI

MODEL_REGISTRY.update({"infer_connectivity.ei_glm.GLMEI": GLMEI})


def _expand_user_keys(user_key, flat_keys):
    """Expand user key mapping path to match saved keys."""
    parts = user_key.split("__")

    # flat key (one level only)
    if len(parts) == 1:
        # either it is a class or a param
        if f"{parts[0]}__class" in flat_keys:
            return "__".join([parts[0], "class"])
        return parts[0]

    # interleave params, this assumes that the only nesting allowed
    # is: class__params__class__params... but not dictionaries.
    path = []
    for part in parts[:-1]:
        path.extend([part, "params"])

    flat_key = "__".join(path) + f"__{parts[-1]}__class"
    if flat_key in flat_keys:
        return flat_key
    else:
        path.append(parts[-1])
        flat_key = "__".join(path)
    return flat_key


def load_model(filename: Union[str, Path], mapping_dict: dict = None):
    try:
        return new_load_model(filename, mapping_dict)
    except:
        return deprecated_load_model(filename, mapping_dict)


def new_load_model(filename: Union[str, Path], mapping_dict: dict = None):
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
    inverse_link_function: <function one_over_x at ...>
    observation_model: GammaObservations()
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
    inverse_link_function: <function one_over_x at ...>
    observation_model: GammaObservations()
    regularizer: Ridge()
    regularizer_strength: 0.1
    solver_kwargs: {'stepsize': 0.1, 'maxiter': 1000, 'tol': 1e-06}
    solver_name: BFGS

    >>> # Loading a custom inverse link function
    >>> model = nmo.glm.GLM(inverse_link_function=lambda x: x**2)
    >>> model.save_params("model_params.npz")
    >>> # Provide a mapping for the custom link function when loading.
    >>> mapping_dict = {
    ...     "inverse_link_function": lambda x: x**2,
    ... }
    >>> loaded_model = nmo.load_model("model_params.npz", mapping_dict=mapping_dict)
    >>> # Now the loaded model will have the updated solver_name and solver_kwargs
    >>> for key, value in loaded_model.get_params().items():
    ...     print(f"{key}: {value}")
    inverse_link_function: <function <lambda> at ...>
    observation_model: PoissonObservations()
    regularizer: UnRegularized()
    regularizer_strength: None
    solver_kwargs: {}
    solver_name: GradientDescent
    """
    # load the model from a .npz file
    filename = Path(filename)
    data = np.load(filename, allow_pickle=False)

    # unflatten dictionary
    saved_params = _unflatten_dict(data)
    # "save_metadata" is used to store versions of Nemos and Jax, not needed for loading
    saved_params.pop("save_metadata")
    # unflatten user map
    nested_map_dict, key_not_found = _unflattened_user_map(mapping_dict, saved_params)

    invalid_keys = _get_invalid_mappings(nested_map_dict)
    if len(invalid_keys) > 0:
        raise ValueError(
            "Invalid map parameter types detected. "
            f"The following parameters are non mappable:\n\t{invalid_keys}\n"
            "Only callables and classes can be mapped."
        )

    # backtrack all errors
    if key_not_found:
        available_keys = get_user_keys_from_nested_dict(saved_params)
        requested_keys = get_user_keys_from_nested_dict(mapping_dict)
        not_available = sorted(set(requested_keys).difference(available_keys))
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
        raise ValueError(
            "The following keys in your mapping do not match any parameters in the loaded model:\n\n"
            f"{suggestions}\n"
            "Please double-check your mapping dictionary."
        )
    # if any value from saved_params is a key in mapping_dict,
    # replace it with the corresponding value from mapping_dict
    saved_params, updated_keys = _apply_custom_map(saved_params, nested_map_dict)

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


def _get_invalid_mappings_deprecated(mapping_dict: dict | None) -> list:
    if mapping_dict is None:
        return []
    return [
        k
        for k, v in mapping_dict.items()
        if (not inspect.isclass(v)) and not callable(v)
    ]


def _unwrap_param(value):
    """
    Recursively unwrap a mapped param structure.

    - If it’s a param leaf [orig, [is_mapped, mapped]] → return orig.
    - If it’s a dict → recurse on each value.
    """
    if isinstance(value, dict):
        # Nested dict: unwrap each leaf.
        return {k: _unwrap_param(v) for k, v in value.items()}
    return value[0]


def _is_param(par):
    if not isinstance(par, dict):
        return True
    return "class" not in par


def _safe_instantiate(
    parameter_name: str, class_name: str, **kwargs
) -> "Regularizer | Observations":
    if not isinstance(class_name, str):
        # this should not be hit, if it does the saved params had been modified.
        raise ValueError(
            f"Parameter ``{parameter_name}`` cannot be initialized. "
            "When a parameter specifies a class, the class name must be a string. "
            f"Class name for the loaded parameter is {class_name}."
        )
    class_basename = class_name.split(".")[-1]
    if class_basename in AVAILABLE_REGULARIZERS:
        return instantiate_regularizer(class_name)
    elif any(class_basename.startswith(obs) for obs in AVAILABLE_OBSERVATION_MODELS):
        return instantiate_observation_model(class_name, **kwargs)
    else:
        # Hit when loading a custom class without mapping
        if parameter_name == "observation_model":
            class_type = "observation"
        else:
            class_type = "regularization"
        raise ValueError(
            f"The class '{class_basename}' is not a native NeMoS class.\n"
            f"To load a custom {class_type} class, please provide the following mapping:\n\n"
            f" - nemos.load_model(save_path, mapping_dict={{'{parameter_name}': {class_basename}}})"
        )


def _apply_custom_map_deprecated(
    params: dict, updated_keys: list | None = None
) -> tuple[dict, list]:
    """
    Recursively apply user-defined mappings to a saved parameter structure.

    This function processes the nested parameter dictionary produced by `_unflatten_dict`
    and applies user-specified overrides where allowed. It does the following:

    - For leaf parameters stored as [value, [is_mapped, mapped_value]]:
        * If `is_mapped` is True, the parameter is replaced with `mapped_value`.
          Only callables or classes are allowed; other types raise an error.
        * If `is_mapped` is False, the original saved value is kept.
    - For nested dictionaries of parameters (e.g., solver kwargs):
        * These cannot be overridden because they are not callables or classes.
        * All leaves are recursively unwrapped to extract the original saved values,
          discarding any mapping info.
    - For parameters representing classes:
        * If not mapped, the original class name is checked and instantiated safely using `_safe_instantiate`.
        * If mapped, the mapping must be an actual Python class object (not a string or an instance).
          This invariant is enforced with an internal assertion for developer safety.

    This function also keeps track of which keys were overridden by the user-supplied mapping,
    returning this list alongside the reconstructed parameter dictionary.

    Parameters
    ----------
    params :
        The nested saved parameters to process. Each entry is either:
          - A leaf in the form [value, [is_mapped, mapped_value]], or
          - A nested dict representing classes.
    updated_keys :
        List of keys that have already been updated, used for accumulating changes
        across recursive calls.

    Returns
    -------
    updated_params :
        The new parameter dictionary with mappings applied and wrappers removed.
    updated_keys :
        List of all keys that were actually overridden.

    Raises
    ------
    ValueError
        If a user tries to override a parameter with an unsupported type (non-callable, non-class),
        or provides a mapped class as a string instead of a Python class object. This is triggered
        in `_safe_instantiate`.

    Notes
    -----
    This function enforces strict override safety: only callables and classes may be
    mapped at load time. Directly serializable values and nested dictionaries cannot
    be overridden and must be changed later using `set_params` if needed.

    Internal invariants are checked with `assert` to ensure that only valid class mappings
    reach instantiation. If these assertions fail, it indicates a bug in the input validation
    logic and should never occur in normal use.
    """
    updated_params = {}

    if updated_keys is None:
        updated_keys = []

    for key, val in params.items():
        # handle classes and params separately
        if _is_param(val):
            if isinstance(val, dict):
                # dict cannot be mapped, so store original params
                orig_param = _unwrap_param(val)
                updated_params[key] = orig_param
            else:
                # unpack mapping info and val
                orig_param, (is_mapped, mapped_param) = val
                if is_mapped:
                    updated_params[key] = mapped_param
                    updated_keys.append(key)
                else:
                    updated_params[key] = orig_param

        else:
            # if val is a class, it must be a dict with a "class" key
            class_name, (is_mapped, mapped_class) = val.pop("class")
            if not is_mapped:
                # check for nested callable/classes save instantiate based on the string
                new_params, updated_keys = _apply_custom_map_deprecated(
                    val.pop("params", {}), updated_keys=updated_keys
                )
                updated_params[key] = _safe_instantiate(key, class_name, **new_params)
            else:
                updated_keys.append(key)
                # Should not be hit ever, assertion for developers
                assert inspect.isclass(mapped_class), (
                    f"The parameter '{key}' passed the type check in "
                    f"``nmo.load_model`` but is not callable or class, "
                    "check why this is the case."
                )
                # map callables and nested classes
                new_params, updated_keys = _apply_custom_map_deprecated(
                    val.pop("params", {}), updated_keys=updated_keys
                )
                # try instantiating it with the params
                # this executes code, but we are assuming that the mapped_class is safe
                updated_params[key] = mapped_class(**new_params)

    return updated_params, updated_keys


def _split_model_params_deprecated(params: dict, model_class) -> tuple:
    """Split parameters into config and fit parameters."""
    model_param_names = model_class._get_param_names()
    config_params = {k: v for k, v in params.items() if k in model_param_names}
    fit_params = {k: v for k, v in params.items() if k not in model_param_names}
    return config_params, fit_params


def _set_fit_params_deprecated(model, fit_params: dict, filename: Path):
    """Set fit model attributes, warn if unrecognized."""
    check_str = "\nIf this is confusing, try calling inspect_npz."
    for key, value in fit_params.items():
        if hasattr(model, key):
            setattr(model, key, value)
        else:
            raise ValueError(
                f"Unrecognized attribute '{key}' during model loading.{check_str}"
            )


def _suggest_keys_deprecated(
    unmatched_keys: list[str], valid_keys: list[str], cutoff: float = 0.6
):
    """
    Suggest the closest matching valid key for each unmatched key using fuzzy string matching.

    This function compares each unmatched key to a list of valid keys and returns a suggestion
    if a close match is found based on the similarity score.

    Parameters
    ----------
    unmatched_keys :
        Keys that were provided by the user but not found in the expected set.
    valid_keys :
        The list of valid/expected keys to compare against.
    cutoff :
        The minimum similarity ratio (between 0 and 1) required to consider a match.
        A higher value means stricter matching. Defaults to 0.6.

    Returns
    -------
    :
        A list of (provided_key, suggested_key) pairs. If no match is found,
        `suggested_key` will be `None`.

    Examples
    --------
    >>> _suggest_keys(["observaton_model"], ["observation_model", "regularization"])
    [('observaton_model', 'observation_model')]
    """
    import difflib

    key_paris = []  # format, (user_provided, similar key)
    for unmatched_key in unmatched_keys:
        suggestions = difflib.get_close_matches(
            unmatched_key, valid_keys, n=1, cutoff=cutoff
        )
        key_paris.append((unmatched_key, suggestions[0] if suggestions else None))
    return key_paris


def _unflatten_dict_deprecated(
    flat_dict: dict, flat_map_dict: dict | None = None
) -> dict:
    """
    Unflatten a dictionary with keys representing hierarchy into a nested dictionary.

    Parameters
    ----------
    flat_dict :
        The dictionary to unflatten.

    Returns
    -------
    out :
        A nested dictionary with the original hierarchy restored.
    """
    if flat_map_dict is None:
        flat_map_dict = {}
        add_mapping = False
    else:
        add_mapping = True

    sep = "__"
    nested_dict = {}
    # Process each key-value pair in the flattened dictionary
    for k, v in flat_dict.items():
        keys = k.split(sep)

        if k in flat_map_dict:
            mapping = [True, flat_map_dict[k]]
        else:
            mapping = [False, None]

        dct = nested_dict
        # Traverse or create nested dictionaries
        for key in keys[:-1]:
            if key not in dct:
                dct[key] = {}
            dct = dct[key]
        # Convert numpy string, int, float or nan to their respective types
        if v.dtype.type is np.str_:
            v = str(v)
        elif v.dtype.type is np.int_:
            v = int(v)
        elif issubclass(v.dtype.type, np.floating):
            if v.ndim == 0:
                v = None if np.isnan(v) else float(v)
        dct[keys[-1]] = [v, mapping] if add_mapping else v
    return nested_dict


def deprecated_load_model(filename: Union[str, Path], mapping_dict: dict = None):
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

    invalid_keys = _get_invalid_mappings_deprecated(mapping_dict)
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
        suggested_pairs = _suggest_keys_deprecated(not_available, available_keys)
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
    saved_params = _unflatten_dict_deprecated(data, flat_map_dict)

    # "save_metadata" is used to store versions of Nemos and Jax, not needed for loading
    saved_params.pop("save_metadata")

    # if any value from saved_params is a key in mapping_dict,
    # replace it with the corresponding value from mapping_dict
    # surgery to fix deprecations
    saved_params["inverse_link_function"] = saved_params["observation_model"][
        "params"
    ].pop("inverse_link_function")
    saved_params, updated_keys = _apply_custom_map_deprecated(saved_params)

    if len(updated_keys) > 0:
        warnings.warn(
            f"The following keys have been replaced in the model parameters: {updated_keys}.",
            UserWarning,
        )

    # Extract the model class from the saved attributes
    model_name = str(saved_params.pop("model_class"))
    MODEL_REGISTRY["nemos.glm.GLM"] = MODEL_REGISTRY["nemos.glm.glm.GLM"]
    MODEL_REGISTRY["nemos.glm.PopulationGLM"] = MODEL_REGISTRY[
        "nemos.glm.glm.PopulationGLM"
    ]
    model_class = MODEL_REGISTRY[model_name]

    config_params, fit_params = _split_model_params_deprecated(
        saved_params, model_class
    )

    # Create the model instance
    try:
        model = model_class(**config_params)
    except Exception:
        raise ValueError(
            f"Failed to instantiate model class '{model_name}' with parameters: {config_params}. "
            f"Use `nmo.inspect_npz('{filename}')` to inspect the saved object."
        )

    # Set the rest of the parameters as attributes if recognized
    _set_fit_params_deprecated(model, fit_params, filename)

    return model
