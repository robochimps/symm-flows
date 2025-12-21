from typing import Callable, List, Optional
import numpy as np
import jax
from flax import linen as nn
from jax import config
from jax import numpy as jnp
import jax.nn.initializers as initializers
from numpy.typing import NDArray
from typing import Union
from .invertible_block import ActivationFunction, SingularValues, InvertibleResNetBlock, FixPoint
from .linear import Linear, LinearOnInterval
from .tanh import Tanh, Tanh2

config.update("jax_enable_x64", True)

class IResNet(nn.Module):
    """
    Invertible Residual Network (iResNet) with multiple wrappers and flexible architecture.

    Parameters
    ----------
    a : list or ndarray
        Output layer scaling parameters.
    b : list or ndarray
        Output layer shift parameters.
    intervals : ndarray
        Output intervals for each dimension.
    xmin : list or ndarray
        Minimum input values for normalization.
    xmax : list or ndarray
        Maximum input values for normalization.
    features : list of int
        List of layer sizes for each block.
    activations : list of callable, optional
        List of activation functions for each layer (default: LIPSWISH).
    lipschitz_constant : float, optional
        Lipschitz constant for spectral normalization (default: 0.9).
    opt_a : bool, optional
        Whether to optimize parameter a (default: True).
    opt_b : bool, optional
        Whether to optimize parameter b (default: True).
    no_resnet_blocks : int, optional
        Number of invertible ResNet blocks (default: 1).
    no_inv_iters : int, optional
        Number of fixed-point iterations for inversion (default: 30).
    _wrapper : callable, optional
        Wrapper transformation (default: Tanh).
    fix_point_method : FixPoint, optional
        Fixed-point iteration method for inversion (default: REGULAR).
    svd_method : SingularValues, optional
        SVD method for spectral normalization (default: SVD_MULTIPLE).
    """
    a: List[float] | NDArray[np.float64]
    b: List[float] | NDArray[np.float64]
    intervals: NDArray[np.float64]
    xmin: List[float] | NDArray[np.float64]
    xmax: List[float] | NDArray[np.float64]
    features: List[int]
    ## Significant optional quantities
    activations: Optional[List[Callable]] = (
        ActivationFunction.LIPSWISH,
        ActivationFunction.LIPSWISH,
    )
    lipschitz_constant: Optional[float] = 0.9
    opt_a: Optional[bool] = True
    opt_b: Optional[bool] = True
    no_resnet_blocks: Optional[int] = 1
    no_inv_iters: Optional[int] = 30

    ## Specific variables that usually dont require of change
    _wrapper: List[Callable] = Tanh
    fix_point_method: Optional[FixPoint] = FixPoint.REGULAR
    svd_method: Optional[SingularValues] = SingularValues.SVD_MULTIPLE
    
    def setup(self):
        self.linear_input = Linear(
            a = 0.995 / (self.xmax - self.xmin) * 2,
            b = -0.995 * (self.xmin + self.xmax) / (self.xmax - self.xmin),
            opt_a=False,
            opt_b=False,
        )
        self.linear_output = LinearOnInterval(
            a=self.a,
            b=self.b,
            intervals=self.intervals,
            opt_a=self.opt_a,
            opt_b=self.opt_b,
        )
        self.linear_ = [
            Linear(
                a=jnp.ones_like(jnp.asarray(self.a)),
                b=jnp.zeros_like(jnp.asarray(self.b)),
                opt_a=True,
                opt_b=True,
            )
            for _ in range(self.no_resnet_blocks)
        ]
        self.wrapper = self._wrapper()

        self.resnet = [
            InvertibleResNetBlock(
                features=self.features,
                activations=self.activations,
                lipschitz_constant=self.lipschitz_constant,
                svd_method=self.svd_method,
                fix_point_method=self.fix_point_method,
                no_inv_iters=self.no_inv_iters,
            )
            for _ in range(self.no_resnet_blocks)
        ]

    @nn.compact
    def __call__(self, x, inverse: bool = False, train: bool = False):
        if inverse:
            x = self.linear_input(x, inverse=True)
            x = self.wrapper(x, inverse=True)
            
            for block, linear in zip(reversed(self.resnet), reversed(self.linear_)):
                x = block(x, inverse=True)
                x = linear(x, inverse=True)
             
            x = self.wrapper(x, inverse=False)
            
            x = self.linear_output(x, inverse=True)
            
        else:
            x = self.linear_output(x, inverse=False)

            x = self.wrapper(x, inverse=True)

            for block, linear in zip(self.resnet, self.linear_):
                x = linear(x, inverse=False)
                x = block(x, inverse=False)
                

            x = self.wrapper(x, inverse=False)
            x = self.linear_input(x, inverse=False)

        return x


class IResNet2(nn.Module):
    """
    iResNet with only one wrapper. Range does not depend on input quadrature.

    Parameters
    ----------
    a : list or ndarray
        Output layer scaling parameters.
    b : list or ndarray
        Output layer shift parameters.
    intervals : ndarray
        Output intervals for each dimension.
    features : list of int
        List of layer sizes for each block.
    activations : list of callable, optional
        List of activation functions for each layer (default: LIPSWISH).
    lipschitz_constant : float, optional
        Lipschitz constant for spectral normalization (default: 0.9).
    opt_a : bool, optional
        Whether to optimize parameter a (default: True).
    opt_b : bool, optional
        Whether to optimize parameter b (default: True).
    no_resnet_blocks : int, optional
        Number of invertible ResNet blocks (default: 1).
    xmax : float or ndarray, optional
        Scaling for wrapper linear layer (default: 1.0).
    xshift : float or ndarray, optional
        Shift for wrapper linear layer (default: 0.0).
    group : array_like, optional
        Symmetry group matrices (default: None).
    _wrapper : callable, optional
        Wrapper transformation (default: Tanh2).
    fix_point_method : FixPoint, optional
        Fixed-point iteration method for inversion (default: REGULAR).
    svd_method : SingularValues, optional
        SVD method for spectral normalization (default: PAR_CLIP).
    no_inv_iters : int, optional
        Number of fixed-point iterations for inversion (default: 30).
    """
    a: List[float] | NDArray[np.float64]
    b: List[float] | NDArray[np.float64]
    intervals: NDArray[np.float64]
    features: List[int]

    ## Significant optional quantities
    activations: Optional[List[Callable]] = (
        ActivationFunction.LIPSWISH,
        ActivationFunction.LIPSWISH,
    )
    lipschitz_constant: Optional[float] = 0.9
    opt_a: Optional[bool] = True
    opt_b: Optional[bool] = True
    no_resnet_blocks: Optional[int] = 1
    
    xmax: Optional[float] | NDArray[np.float64] = 1.0
    xshift: Optional[float] | NDArray[np.float64] = 0.0
    group: Optional[Union[List[float], NDArray[np.float64]]] = None

    ## Specific variables that usually dont require of change
    _wrapper: List[Callable] = Tanh2
    fix_point_method: Optional[FixPoint] = FixPoint.REGULAR
    svd_method: Optional[SingularValues] = SingularValues.PAR_CLIP
    no_inv_iters: Optional[int] = 30

    def setup(self):
        self.wrapper_linear = Linear(
                a=self.xmax, 
                b=self.xshift,
                opt_a=False,
                opt_b=False,
                group=self.group)

        self.linear_output = LinearOnInterval(
            a=self.a,
            b=self.b,
            intervals=self.intervals,
            opt_a=self.opt_a,
            opt_b=self.opt_b,
            group=self.group,
        )
        
        self.linear_ = [
            Linear(
                a=jnp.ones_like( jnp.asarray(self.a)), #jnp.eye(len(jnp.asarray(self.a))),
                b=jnp.zeros_like(jnp.asarray(self.b)),
                opt_a=True,
                opt_b=True,
                group=self.group,
            )
            for _ in range(self.no_resnet_blocks)
        ]
        
        self.wrapper = self._wrapper()

        self.resnet = [
            InvertibleResNetBlock(
                features=self.features,
                activations=self.activations,
                lipschitz_constant=self.lipschitz_constant,
                svd_method=self.svd_method,
                fix_point_method=self.fix_point_method,
                no_inv_iters=self.no_inv_iters,
                group=self.group,
            )
            for _ in range(self.no_resnet_blocks)
        ]

    @nn.compact
    def __call__(self, x, inverse: bool = False, train: bool = False):
        if inverse:
            x = self.wrapper_linear(x, inverse=False)
            for block, linear in zip(reversed(self.resnet), reversed(self.linear_)):
                x = block(x, inverse=True)
                x = linear(x, inverse=True)
            
            x = self.wrapper(x, inverse=False)
            x = self.linear_output(x, inverse=True)
        else:
            x = self.linear_output(x, inverse=False)
            x = self.wrapper(x, inverse=True)

            for block, linear in zip(self.resnet, self.linear_):
                x = linear(x, inverse=False)
                x = block(x, inverse=False)
                
            x = self.wrapper_linear(x, inverse=True)
        return x


def clip_kernel_svd_multiple(params, lipschitz_constant): 
    """
    Enforce Lipschitz constant via individual singular values (SVD multiple block).

    Parameters
    ----------
    params : pytree
        Model parameters.
    lipschitz_constant : float
        Maximum allowed singular value.

    Returns
    -------
    pytree
        Parameters with singular values clipped.
    """

    def clip_fn(path, leaf):
        key = getattr(path[-1], 'key', None)
        if isinstance(key, str) and 'w_' in key:
            L, S, R = jnp.linalg.svd(leaf, full_matrices=False)
            S_clipped = jnp.minimum(S, lipschitz_constant)
            return (L * S_clipped) @ R
        return leaf
    return jax.tree_util.tree_map_with_path(clip_fn, params)

def clip_kernel_svd(params, lipschitz_constant):
    """
    Enforce Lipschitz constant via maximum singular value (SVD block).

    Parameters
    ----------
    params : pytree
        Model parameters.
    lipschitz_constant : float
        Maximum allowed singular value.

    Returns
    -------
    pytree
        Parameters with singular values clipped.
    """

    def clip_fn(path, leaf):
        key = getattr(path[-1], 'key', None)
        if isinstance(key, str) and 'w_' in key:
            L, S, R = jnp.linalg.svd(leaf, full_matrices=False)
            sv = jnp.max(S)
            switch = jnp.heaviside(lipschitz_constant, sv)
            leaf = switch * leaf /sv * lipschitz_constant + ( 1 - switch) * leaf
            return leaf
        return leaf
    return jax.tree_util.tree_map_with_path(clip_fn, params)

class UnitaryTransform(nn.Module):
    """
    Parametric unitary transformation using skew-symmetric matrix exponentiation.

    Parameters
    ----------
    n : int
        Size of the unitary matrix.
    """
    n: int
    
    def setup(self):
        self.idx = jnp.triu_indices(self.n, k=1)
        
    
    @nn.compact
    def __call__(self):
        a = self.param(
                "linear_a", initializers.zeros, (len(self.idx[0]),)
            )
        mat = jnp.zeros((self.n, self.n))
        A = mat.at[self.idx].set(a)
        
        M = 0.5 * (A - A.T)

        I = jnp.eye(len(A))

        Q = jnp.dot( I - M, jnp.linalg.inv( I + M))

        return Q
