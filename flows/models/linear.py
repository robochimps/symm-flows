from typing import List, Optional, Union
from jax import config
from jax import numpy as jnp
from flax import linen as nn
from numpy.typing import NDArray
import numpy as np
import jax

config.update("jax_enable_x64", True)

class Identity(nn.Module):
    """Identity transformation x -> x."""

    @nn.compact
    def __call__(self, x, inverse: bool = False):
        return x

class Linear(nn.Module):
    """Trainable linear transformation a * x + b, optional equivariance group."""
    a: Union[List[float], NDArray[np.float64]]
    b: Union[List[float], NDArray[np.float64]]
    opt_a: bool = True
    opt_b: bool = True
    group: Optional[Union[List[float], NDArray[np.float64]]] = None

    @nn.compact
    def __call__(self, x, inverse: bool = False):

        # parameters
        if self.opt_a:
            a = self.param("linear_a", lambda *_: jnp.asarray(self.a), jnp.shape(self.a))
        else:
            a = jnp.asarray(self.a)

        if self.opt_b:
            b = self.param("linear_b", lambda *_: jnp.asarray(self.b), jnp.shape(self.b))
        else:
            b = jnp.asarray(self.b)

        if self.group is None:
            if inverse:
                return x * a + b
            return (x - b) / a

        group = jnp.asarray(self.group)
        xg = jnp.einsum('...i,gki->...gk', x, group)
        
        if inverse:
            xg = a * xg + b
        else:
            xg = (xg - b) / a
        
        return jnp.einsum('...gk,gki->...i', xg, group) / len(group)

## For the output layer, we map into intervals
def _cond_b_infinite(b, interval):
    """
    Return b unchanged (for infinite interval case).

    Parameters
    ----------
    b : float or array_like
        Parameter b.
    interval : array_like
        Interval specification.

    Returns
    -------
    float or array_like
        Unchanged b.
    """
    return b


def _cond_b_open_right(b, interval):
    """
    Map b to (b^2 + left endpoint) for open-right intervals.

    Parameters
    ----------
    b : float or array_like
        Parameter b.
    interval : array_like
        Interval specification.

    Returns
    -------
    float or array_like
        Transformed b.
    """
    return b**2 + interval[0]


def _cond_b_finite(b, interval):
    """
    Map b to finite interval using sigmoid.

    Parameters
    ----------
    b : float or array_like
        Parameter b.
    interval : array_like
        Interval specification (finite).

    Returns
    -------
    float or array_like
        Transformed b in interval.
    """
    left = interval[0]
    len_interval = interval[1] - interval[0]
    effective_b = jax.nn.sigmoid(b) * len_interval
    return left + effective_b


def _cond_a_infinite(a, b, interval):
    """
    Return a unchanged (for infinite interval case).

    Parameters
    ----------
    a : float or array_like
        Parameter a.
    b : float or array_like
        Parameter b (unused).
    interval : array_like
        Interval specification.

    Returns
    -------
    float or array_like
        Unchanged a.
    """
    return a


def _cond_a_open_right(a, b, interval):
    """
    Map a to a^2 for open-right intervals.

    Parameters
    ----------
    a : float or array_like
        Parameter a.
    b : float or array_like
        Parameter b (unused).
    interval : array_like
        Interval specification.

    Returns
    -------
    float or array_like
        Transformed a.
    """
    return a**2


def _cond_a_finite(a, b, interval):
    """
    Map a to finite interval using sigmoid.

    Parameters
    ----------
    a : float or array_like
        Parameter a.
    b : float or array_like
        Parameter b.
    interval : array_like
        Interval specification (finite).

    Returns
    -------
    float or array_like
        Transformed a in interval.
    """
    original_len = 2.0
    new_len_max = interval[1] - b
    effective_a = jax.nn.sigmoid(a)
    return effective_a / original_len * new_len_max


class LinearOnInterval(nn.Module):
    """
    Trainable linear transformation a * x + b.
    In each dimension, maps [-1,1] to the interval specified
    in intervals.
    """

    a: Union[List[float], NDArray[np.float64]]
    b: Union[List[float], NDArray[np.float64]]
    intervals: NDArray[np.float64]
    opt_a: Optional[bool] = True
    opt_b: Optional[bool] = True
    group: Optional[Union[List[float], NDArray[np.float64]]] = None

    def setup(self):

        def _apply_cond_b(index, b, intervals):
            return jax.lax.switch(
                index,
                [_cond_b_infinite, _cond_b_open_right, _cond_b_finite],
                b,
                intervals,
            )

        def _apply_cond_a(index, a, b, intervals):
            return jax.lax.switch(
                index,
                [_cond_a_infinite, _cond_a_open_right, _cond_a_finite],
                a,
                b,
                intervals,
            )

        self.index = jnp.where(
            (self.intervals[:, 0] == -jnp.inf) & (self.intervals[:, 1] == jnp.inf),
            0,
            jnp.where(
                (self.intervals[:, 0] != -jnp.inf) & (self.intervals[:, 1] == jnp.inf),
                1,
                2,
            ),
        )
        
        self.cond_b = lambda b: jax.vmap(_apply_cond_b, (0, 0, 0))(
            self.index, b, self.intervals
        )
        self.cond_a = lambda a, b: jax.vmap(_apply_cond_a, (0, 0, 0, 0))(
            self.index, a, b, self.intervals
        )
        self.b2 = jnp.where((self.index == 1) & (jnp.asarray(self.b) == 0.), jnp.asarray(self.b)+ 1e-3, jnp.asarray(self.b)  )

    @nn.compact
    def __call__(self, x, inverse: bool = False):
        if self.opt_b:
            b = self.param(
                "linear_b", lambda *_: jnp.asarray(self.b2), jnp.shape(self.b2)
            )
        else:
            b = jnp.asarray(self.b)

        if self.opt_a:
            a = self.param(
                "linear_a", lambda *_: jnp.asarray(self.a), jnp.shape(self.a)
            )
        else:
            a = jnp.asarray(self.a)
        
        b = self.cond_b(b)
        a = self.cond_a(a, b)
        if self.group is None:
            if inverse:
                return a * (x + 1) + b
            return (x - b) / a - 1
    
        group = jnp.asarray(self.group)
        xg = jnp.einsum('...i,gki->...gk', x, group)

        if inverse:
            xg = a * (xg + 1) + b
        else:
            xg = (xg - b) / a - 1

        return jnp.einsum('...gk,gki->...i', xg, group) / len(group)


def compute_a_b(a, b, intervals):
    """
    Compute transformed a and b for mapping [-1, 1] to arbitrary intervals.

    Parameters
    ----------
    a : array_like
        Parameter a for each dimension.
    b : array_like
        Parameter b for each dimension.
    intervals : array_like
        Array of intervals, shape (n, 2).

    Returns
    -------
    a_trans : array_like
        Transformed a for each interval.
    b_trans : array_like
        Transformed b for each interval.
    """
    index = jnp.where(
        (intervals[:, 0] == -jnp.inf) & (intervals[:, 1] == jnp.inf),
        0,
        jnp.where(
            (intervals[:, 0] != -jnp.inf) & (intervals[:, 1] == jnp.inf),
            1,
            2,
        ),
    )

    def _apply_cond_b(i, b_i, interval_i):
        return jax.lax.switch(i, [_cond_b_infinite, _cond_b_open_right, _cond_b_finite], b_i, interval_i)

    def _apply_cond_a(i, a_i, b_i, interval_i):
        return jax.lax.switch(i, [_cond_a_infinite, _cond_a_open_right, _cond_a_finite], a_i, b_i, interval_i)

    b_trans = jax.vmap(_apply_cond_b)(index, b, intervals)
    a_trans = jax.vmap(_apply_cond_a)(index, a, b_trans, intervals)
    b_trans = b_trans + a_trans #a * (x + 1) + b
    return a_trans, b_trans