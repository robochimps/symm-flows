import numpy as np
import jax
from enum import Enum
from typing import List, Optional, Union
from numpy.typing import NDArray
from flax import linen as nn
from jax import config
from jax import numpy as jnp

config.update("jax_enable_x64", True)


def _singular_values_fourier(kernel, input_shape):
    """
    Compute singular values of a kernel after 2D FFT (Fourier domain).

    Parameters
    ----------
    kernel : array_like
        The kernel matrix.
    input_shape : tuple
        Shape for FFT.

    Returns
    -------
    array_like
        Singular values of the transformed kernel.
    """
    transforms = jnp.fft.fft2(kernel, input_shape, axes=[0, 1])
    return jnp.linalg.svd(transforms, compute_uv=False)


def _singular_values(kernel, input_shape):
    """
    Compute singular values of a kernel.

    Parameters
    ----------
    kernel : array_like
        The kernel matrix.
    input_shape : tuple
        Shape of the kernel (unused).

    Returns
    -------
    array_like
        Singular values of the kernel.
    """
    return jnp.linalg.svd(kernel, compute_uv=False)


def _sigmoid(x):
    """
    Numerically stable sigmoid activation.

    Parameters
    ----------
    x : array_like
        Input value(s).

    Returns
    -------
    array_like
        Sigmoid of x.
    """
    return 1 / (1 + jnp.exp(-x))


def _lipswish(x):
    """
    Lipswish activation function.

    Parameters
    ----------
    x : array_like
        Input value(s).

    Returns
    -------
    array_like
        Lipswish of x.
    """
    return (x / 1.1) * _sigmoid(x)


class ActivationFunction(Enum):
    """
    Special types of activation functions for invertible MLP mappings.

    Members
    -------
    LIPSWISH : callable
        Lipswish activation function.
    RELU : callable
        ReLU activation function.
    IDENTITY : callable
        Identity function.
    """
    LIPSWISH = _lipswish
    RELU = nn.relu
    IDENTITY = lambda x: x


class SingularValues(Enum):
    """
    Choice of different computations of SVD of the kernel of the normalizing flow (NF).

    Members
    -------
    SVD : str
        Apply SVD on the kernel directly.
    FOURIER : str
        Fourier transform the kernel, then apply SVD.
    SVD_MULTIPLE : str
        Allow for different SVD values to be equal to the Lipschitz constant.
    PAR_CLIP : str
        Apply SVD of kernels via parameter update.
    PAR_CLIP_EQUIVARIANT : str
        Apply SVD of kernels via parameter update and symmetrize via group input.
    """
    SVD = "Apply SVD on the kernel directly"
    FOURIER = "Fourier transform the kernel, then apply SVD"
    SVD_MULTIPLE = (
        "Allow for different SVD values to be equal to the Lipschitz constant"
    )
    PAR_CLIP = "Apply SVD of kernels via parameter update"
    PAR_CLIP_EQUIVARIANT = "Apply SVD of kernels via parameter update and symmetrize via group input"

class FixPoint(Enum):
    """
    Choice of different computations of the fixed-point method for the inversion of the neural network.

    Members
    -------
    REGULAR : str
        Use the iteration x_{k+1} = T x_{k}.
    HALPERN : str
        Use the iteration x_{k+1} = x_0/(k+2) + (1 - 1/(k+2)) * T x_{k}.
    PICARD : str
        Use the iteration x_{k+1} = x_k/(k+2) + (1 - 1/(k+2)) * T x_{k}.
    """
    REGULAR = "Use the iteration x_{k+1} = T x_{k}"
    HALPERN = "Use the iteration x_{k+1} = x_0/(k+2)  + (1 - 1/(k+2)) * T x_{k}"
    PICARD = "Use the iteration x_{k+1} = x_k/(k+2)  + (1 - 1/(k+2)) * T x_{k}"


class _InvertibleDenseBlockFourier(nn.Module):
    """
    Invertible dense block using Fourier-based SVD for spectral normalization.

    Parameters
    ----------
    features : list of int
        List of layer sizes.
    activations : list of ActivationFunction
        List of activation functions for each layer.
    lipschitz_constant : float, optional
        Lipschitz constant for spectral normalization (default: 0.9).
    """
    features: List[int]
    activations: List[ActivationFunction]
    lipschitz_constant: Optional[float] = 0.9

    @nn.compact
    def __call__(self, x):
        size_ = x.shape[-1]
        for i, size in enumerate(self.features):
            kernel = self.param(
                f"w_{i}", jax.nn.initializers.glorot_uniform(), (size_, size)
            )
            sv = jnp.max(_singular_values_fourier(kernel, jnp.shape(kernel)))
            kernel = jax.lax.cond(
                sv >= self.lipschitz_constant,
                lambda a, b: self.lipschitz_constant * a / b,
                lambda a, b: a,
                kernel,
                sv,
            )
            bias = self.param(f"b_{i}", jax.nn.initializers.zeros, (size,))
            x = self.activations[i](jnp.dot(x, kernel) + bias)
            size_ = size
        return x


class _InvertibleDenseBlockSVD(nn.Module):
    """
    Invertible dense block using SVD for spectral normalization.

    Parameters
    ----------
    features : list of int
        List of layer sizes.
    activations : list of ActivationFunction
        List of activation functions for each layer.
    lipschitz_constant : float, optional
        Lipschitz constant for spectral normalization (default: 0.9).
    """
    features: List[int]
    activations: List[ActivationFunction]
    lipschitz_constant: Optional[float] = 0.9

    @nn.compact
    def __call__(self, x):
        size_ = x.shape[-1]
        for i, size in enumerate(self.features):
            kernel = self.param(
                f"w_{i}", jax.nn.initializers.glorot_uniform(), (size_, size)
            )
            sv = jnp.max(_singular_values(kernel, jnp.shape(kernel)))
            switch = jnp.heaviside(self.lipschitz_constant, sv)
            kernel = switch * kernel /sv * self.lipschitz_constant + \
                     ( 1 - switch) * kernel
            bias = self.param(f"b_{i}", jax.nn.initializers.zeros, (size,))
            x = self.activations[i](jnp.dot(x, kernel) + bias)
            size_ = size
        return x


class _InvertibleDenseBlockSVD_mult(nn.Module):
    """
    Invertible dense block using SVD with multiple singular value clipping.

    Parameters
    ----------
    features : list of int
        List of layer sizes.
    activations : list of ActivationFunction
        List of activation functions for each layer.
    lipschitz_constant : float, optional
        Lipschitz constant for spectral normalization (default: 0.9).
    """
    features: List[int]
    activations: List[ActivationFunction]
    lipschitz_constant: Optional[float] = 0.9

    @nn.compact
    def __call__(self, x):
        size_ = x.shape[-1]
        for i, size in enumerate(self.features):
            kernel = self.param(
                f"w_{i}", jax.nn.initializers.glorot_uniform(), (size_, size)
            )
            bias = self.param(f"b_{i}", jax.nn.initializers.zeros, (size,))
            L, S, R = jnp.linalg.svd(kernel, full_matrices=False)

            def update_S(x):
                return jax.lax.cond(
                    x >= self.lipschitz_constant,
                    lambda _: self.lipschitz_constant,
                    lambda _: x,
                    None,
                )

            S_updated = jax.vmap(update_S)(S)
            kernel = jnp.dot(L * S_updated, R)
            x = self.activations[i](jnp.dot(x, kernel) + bias)
            size_ = size
        return x


class _DenseBlock(nn.Module):
    """
    Single ResNet block: MLP(x) + x.
    To be invertible, the weight matrix can be normalized by its spectral norm within the parameter update (see clip_kernel functions in models.py).

    Parameters
    ----------
    features : list of int
        List of layer sizes.
    activations : list of ActivationFunction
        List of activation functions for each layer.
    """
    features: List[int]
    activations: List[ActivationFunction]

    @nn.compact
    def __call__(self, x):
        size_ = x.shape[-1]
        for i, size in enumerate(self.features):
            kernel = self.param(
                f"w_{i}", jax.nn.initializers.glorot_uniform(), (size_, size)
            )
            bias = self.param(f"b_{i}", jax.nn.initializers.zeros, (size,))
            x = self.activations[i](jnp.dot(x, kernel) + bias)
            size_ = size
        return x

class _EquivariantDenseBlock(nn.Module):
    """
    Single ResNet block with group equivariance: MLP(x) + x.
    To be invertible, the weight matrix can be normalized by its spectral norm within the parameter update (see clip_kernel functions in models.py).

    Parameters
    ----------
    features : list of int
        List of layer sizes.
    group : array_like
        Symmetry group matrices.
    activations : list of ActivationFunction
        List of activation functions for each layer.
    """
    features: List[int]
    group: Union[List[float], NDArray[np.float64]]
    activations: List[ActivationFunction]
    @nn.compact
    def __call__(self, x):
        size_ = x.shape[-1]
        x = jnp.einsum('...i,gki->...gk',x,self.group) #ki
        for i, size in enumerate(self.features):
            kernel = self.param(
                f"w_{i}", jax.nn.initializers.glorot_uniform(), (size, size_)
            )
            bias = self.param(f"b_{i}", jax.nn.initializers.zeros, (size,))
            x = self.activations[i](jnp.dot(x, kernel.T) + bias)
            size_ = size

        x = jnp.einsum('...gk,gki->...i',x,self.group)/ len(self.group)
        return x

class _Inverse(nn.Module):
    """
    Fixed-point iteration block for regular inversion: x_{k+1} = T x_{k}.
    """
    @nn.compact
    def __call__(self, x, k, dense):
        xk, x0 = x
        x = [x0 - dense(xk), x0]
        return x, x

class _Inverse_Halpern(nn.Module):
    """
    Fixed-point iteration block for Halpern inversion: x_{k+1} = x_0/(k+2) + (1 - 1/(k+2)) * T x_{k}.
    """
    @nn.compact
    def __call__(self, x, k, dense):
        xk, x0 = x
        Tx = x0 - dense(xk)
        x = [1/(k+2) * x0 + (1 - 1/(k+2)) * Tx, x0]
        return x, x

class _Inverse_Picardi(nn.Module):
    """
    Fixed-point iteration block for Picard inversion: x_{k+1} = x_k/(k+2) + (1 - 1/(k+2)) * T x_{k}.
    """
    @nn.compact
    def __call__(self, x, k, dense):
        xk, x0 = x
        Tx = x0 - dense(xk)
        x = [1/(k+2) * xk + (1 - 1/(k+2)) * Tx, x0]
        return x, x

class InvertibleResNetBlock(nn.Module):
    """
    Single invertible ResNet block: MLP(x) + x.

    Parameters
    ----------
    features : list of int
        List of layer sizes.
    activations : list of ActivationFunction or ActivationFunction, optional
        List of activation functions for each layer, or a single activation for all layers (default: LIPSWISH).
    fix_point_method : FixPoint, optional
        Fixed-point iteration method for inversion (default: REGULAR).
    svd_method : SingularValues, optional
        SVD method for spectral normalization (default: PAR_CLIP).
    no_inv_iters : int, optional
        Number of fixed-point iterations for inversion (default: 30).
    lipschitz_constant : float, optional
        Lipschitz constant for spectral normalization (default: 0.9).
    group : array_like, optional
        Symmetry group matrices (default: None).
    """
    features: List[int]
    activations: Optional[Union[List[ActivationFunction], ActivationFunction]] = (
        ActivationFunction.LIPSWISH
    )
    fix_point_method: Optional[FixPoint] = FixPoint.REGULAR
    svd_method: Optional[SingularValues] = SingularValues.PAR_CLIP #Clip in parameter update
    no_inv_iters: Optional[int] = 30
    lipschitz_constant: Optional[float] = 0.9
    group: Optional[Union[List[float], NDArray[np.float64]]] = None

    def setup(self):
        try:
            self._activations = [
                self.activations[i] if i < len(self.activations) else lambda x: x
                for i in range(len(self.features))
            ]
        except TypeError:
            self._activations = [self.activations for _ in range(len(self.features))]

        if self.svd_method == SingularValues.SVD_MULTIPLE:
            self.dense_block = _InvertibleDenseBlockSVD_mult(
                features=self.features,
                activations=self._activations,
                lipschitz_constant=self.lipschitz_constant,
            )
        elif self.svd_method == SingularValues.SVD:
            self.dense_block = _InvertibleDenseBlockSVD(
                features=self.features,
                activations=self._activations,
                lipschitz_constant=self.lipschitz_constant,
            )

        elif self.svd_method == SingularValues.FOURIER:
            self.dense_block = _InvertibleDenseBlockFourier(
                features=self.features,
                activations=self._activations,
                lipschitz_constant=self.lipschitz_constant,
            )
        elif self.svd_method == SingularValues.PAR_CLIP:
            self.dense_block = _DenseBlock(
                features=self.features,
                activations=self._activations,
            )
        elif self.svd_method == SingularValues.PAR_CLIP_EQUIVARIANT:
            self.dense_block = _EquivariantDenseBlock(
                features=self.features,
                group=self.group,
                activations=self._activations,
            )

        else:
            raise NameError(
                f"The choice of svd_method {self.svd_method} is not an implemented method"
            )
        
        
        if self.fix_point_method == FixPoint.REGULAR:
            def __inverse(self, x):
                units = nn.scan(
                    _Inverse,
                    variable_broadcast="params",
                    variable_carry="batch_stats",
                    split_rngs={"params": True},
                    in_axes=0,
                )
                x_, _ = units()([x,x], jnp.arange(self.no_inv_iters), self.dense_block)
                x, x0 = x_
                return x

        elif self.fix_point_method == FixPoint.HALPERN:
            def __inverse(self, x):
                units = nn.scan(
                    _Inverse_Halpern,
                    variable_broadcast="params",
                    variable_carry="batch_stats",
                    split_rngs={"params": True},
                    in_axes=0,
                )
                x_, _ = units()([x,x], jnp.arange(self.no_inv_iters), self.dense_block)
                x, x0 = x_
                return x

        elif self.fix_point_method == FixPoint.PICARD:
            def __inverse(self, x):
                units = nn.scan(
                    _Inverse_Picardi,
                    variable_broadcast="params",
                    variable_carry="batch_stats",
                    split_rngs={"params": True},
                    in_axes=0,
                )
                x_, _ = units()([x,x], jnp.arange(self.no_inv_iters), self.dense_block)
                x, x0 = x_
                return x
            
        self._inverse  = lambda x: __inverse(self, x)
            
        
    def _direct(self, x):
        return self.dense_block(x) + x
    

    @nn.compact
    def __call__(self, x, inverse: bool = False):
        if inverse:
            return self._inverse(x)
        return self._direct(x)