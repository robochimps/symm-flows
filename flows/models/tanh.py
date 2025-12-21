from typing import List, Union
from jax import config
from jax import numpy as jnp
from flax import linen as nn
from numpy.typing import NDArray
import numpy as np

config.update("jax_enable_x64", True)

def tanh(x):
    """
    Numerically stable hyperbolic tangent function.

    Parameters
    ----------
    x : array_like
        Input value(s).

    Returns
    -------
    array_like
        tanh(x) elementwise.
    """
    return (jnp.exp(x) - jnp.exp(-x))/(jnp.exp(x) + jnp.exp(-x))


def arctanh(x):
    """
    Numerically stable inverse hyperbolic tangent function.

    Parameters
    ----------
    x : array_like
        Input value(s), must be in (-1, 1).

    Returns
    -------
    array_like
        arctanh(x) elementwise.
    """
    return .5 * jnp.log((1 + x)/ (1 - x))



class Tanh(nn.Module):
    """
    Module for tanh and its inverse (arctanh) as a Flax layer.
    """


    @nn.compact
    def __call__(self, x, inverse: bool = False):
        if inverse:
            return jnp.arctanh(x)
        else:
            return jnp.tanh(x)
        
        

class Tanh2(nn.Module):
    """
    Module for tanh and its inverse (arctanh), using custom functions, as a Flax layer.
    """


    @nn.compact
    def __call__(self, x, inverse: bool = False):
        if inverse:
            return arctanh(x)
        else:
            return tanh(x)

