from nemos.regularizer import Ridge, Lasso, GroupLasso
from typing import Union, Tuple
from nemos.typing import DESIGN_INPUT_TYPE
from nemos import tree_utils
import jax.numpy as jnp

class MultiRegularization:
    @staticmethod
    def _validate_regularizer_strength(strength: Union[None, float]):
        if strength is None:
            strength = 1.0
        if hasattr(strength, "astype"):
            strength = strength.astype(float)
        return strength


class RidgeMultiRegularization(MultiRegularization, Ridge):


    @staticmethod
    def _penalization(
        params: Tuple[DESIGN_INPUT_TYPE, jnp.ndarray], regularizer_strength: float | jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute the Ridge penalization for given parameters.

        Parameters
        ----------
        params :
            Model parameters for which to compute the penalization.

        Returns
        -------
        float
            The Ridge penalization value.
        """

        def l2_penalty(coeff: jnp.ndarray, intercept: jnp.ndarray) -> jnp.ndarray:
            return (
                0.5
                * jnp.sum(regularizer_strength * jnp.power(coeff, 2))
                / intercept.shape[0]
            )

        # tree map the computation and sum over leaves
        return tree_utils.pytree_map_and_reduce(
            lambda x: l2_penalty(x, params[1]), sum, params[0]
        )


class LassoMultiRegularization(MultiRegularization, Lasso):

    @staticmethod
    def _penalization(
            params: Tuple[DESIGN_INPUT_TYPE, jnp.ndarray], regularizer_strength: float
    ) -> jnp.ndarray:
        """
        Compute the Lasso penalization for given parameters.

        Parameters
        ----------
        params :
            Model parameters for which to compute the penalization.

        Returns
        -------
        float
            The Lasso penalization value.
        """

        def l1_penalty(coeff: jnp.ndarray, intercept: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(regularizer_strength * jnp.abs(coeff)) / intercept.shape[0]

        # tree map the computation and sum over leaves
        return tree_utils.pytree_map_and_reduce(
            lambda x: l1_penalty(x, params[1]), sum, params[0]
        )

class GroupLassoMultiRegularization(MultiRegularization, GroupLasso):
    def __init__(self, mask=None):
        super().__init__(mask=mask)

    def penalized_loss(self, **kwargs):
        raise NotImplementedError("Not implemented for GroupLasso.")
