import jax
import numpy as np
from typing import Callable, List
from jax import config
from jax import numpy as jnp
from itertools import product
from numpy.polynomial.hermite import hermval
config.update("jax_enable_x64", True)

def generate_prod_ind(
    indices: List[List[int]],
    select: Callable[[List[int]], bool] = lambda _: True):
    """
    Generate a product index array with optional selection.

    Parameters
    ----------
    indices : List[List[int]]
        List of index lists for each dimension.
    select : Callable[[List[int]], bool], optional
        Function to filter index tuples. Defaults to always True.

    Returns
    -------
    jax.numpy.ndarray
        Array of selected index tuples (shape: [n, d]).
    """
    list_ = indices[0]
    for i in range(1, len(indices)):
        list_ = list(product(list_,indices[i]))
        list_ = [tuple(a) + (b,) if isinstance(a, tuple) else (a, b) for a, b in list_]
        list_ = [elem for elem in list_ if select(elem)]
    return jnp.array(list_)

@jax.jit
def combine_psi(psi_list, quanta):
    """
    Combine 1D basis functions into multidimensional basis functions.

    Parameters
    ----------
    psi_list : list of ndarray
        List of arrays of 1D basis functions, each shape (npoints, nquanta).
    quanta : ndarray
        Array of quantum numbers, shape (nbas, ncoo).

    Returns
    -------
    jax.numpy.ndarray
        Combined basis functions, shape (npoints, nbas).
    """
    nbas, ncoo = quanta.shape
    npoints = len(psi_list[0])
    select_psi1d = [psi_list[i][:, quanta[:, i]] for i in range(ncoo)]
    psi = jnp.ones((npoints, nbas))
    for arr in select_psi1d:
        psi *= arr
    return psi

@jax.jit
def combine_dpsi(psi_list, dpsi_list, quanta):
    """
    Combine 1D basis and derivative functions into multidimensional derivatives.

    Parameters
    ----------
    psi_list : list of ndarray
        List of arrays of 1D basis functions, each shape (npoints, nquanta).
    dpsi_list : list of ndarray
        List of arrays of 1D basis function derivatives, each shape (npoints, nquanta).
    quanta : ndarray
        Array of quantum numbers, shape (nbas, ncoo).

    Returns
    -------
    jax.numpy.ndarray
        Multidimensional derivatives, shape (npoints, nbas, ncoo).
    """
    dpsi = []
    nbas, ncoo = quanta.shape
    npoints = len(psi_list[0])
    for j in range(ncoo):
        select_dpsi1d = [dpsi_list[i][:, quanta[:, i]] if i == j else psi_list[i][:, quanta[:, i]] for i in range(ncoo)]
        dpsi_j = jnp.ones((npoints, nbas))
        for arr in select_dpsi1d:
            dpsi_j *= arr
        dpsi.append(dpsi_j)
    return jnp.transpose(jnp.stack(dpsi), (1, 2, 0))


def hermval(x, c):
    """
    Evaluate a Hermite series at points x with coefficients c (JAX version).

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the Hermite series.
    c : array_like
        Hermite coefficients.

    Returns
    -------
    array_like
        Evaluated Hermite series at x.
    """
    def iter(carry, cc):
        c0, c1, nd = carry
        tmp = c0
        nd = nd - 1
        c0 = cc - c1*(2*(nd - 1))
        c1 = tmp + c1*x2
        return (c0, c1, nd), 0
    c = c.reshape(c.shape + (1,)*x.ndim)
    x2 = x*2
    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        carry, _ = jax.lax.scan(iter, (c0, c1, nd), np.flip(c, axis=0)[2:])
        c0, c1, nd = carry
    return c0 + c1*x2

def hermite(x, n):
    """
    Compute normalized Hermite functions for given quantum numbers.

    Parameters
    ----------
    x : array_like
        Points at which to evaluate.
    n : array_like
        Quantum numbers.

    Returns
    -------
    array_like
        Hermite functions evaluated at x for each n (shape: [len(n), len(x)]).
    """
    sqsqpi = jnp.sqrt(jnp.sqrt(jnp.pi))
    c = jnp.diag(1.0 / jnp.sqrt(2.0**n * jax.scipy.special.gamma(n+1)) / sqsqpi)
    f = hermval(x, c) * jnp.exp(-(x**2) / 2)
    return f.T

def hermder_jax(c):
    """
    Compute Hermite polynomial derivative coefficients (JAX version).

    Parameters
    ----------
    c : array_like
        Hermite coefficients.

    Returns
    -------
    array_like
        Coefficients of the derivative Hermite polynomial.
    """
    n = c.shape[0]
    idx = jnp.arange(1, n)
    return 2 * idx[:, None] * c[1:]

def _hermite_deriv(x, n):
    """
    Compute the derivative of Hermite functions d/dx(H_n(x)*exp(-x**2/2)).

    Parameters
    ----------
    x : array_like
        Points at which to evaluate.
    n : array_like
        Quantum numbers.

    Returns
    -------
    array_like
        Derivatives evaluated at x for each n (shape: [len(n), len(x)]).
    """
    sqsqpi = jnp.sqrt(jnp.sqrt(jnp.pi))
    c = jnp.diag(1.0 / jnp.sqrt(2.0**n * jax.scipy.special.gamma(n+1)) / sqsqpi)
    h = hermval(x, c)
    dh = hermval(x, hermder_jax(c))
    f = (dh - h * x) * jnp.exp(-(x**2) / 2)
    return f.T

hermite_f = jax.jit(jax.vmap(hermite, in_axes=(0, None)))
dhermite_f = jax.jit(jax.vmap(_hermite_deriv, in_axes=(0, None)))