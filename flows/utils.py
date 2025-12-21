from typing import List

import jax
from jax import config
from jax import numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh

config.update("jax_enable_x64", True)


no_devices = len(jax.local_devices())
device_mesh = mesh_utils.create_device_mesh((no_devices,))
mesh = Mesh(device_mesh, ("g",))
print("default backend:", jax.default_backend(), no_devices, mesh)


def pad_devices_axis(
    arr, axis: int = 0, no_dev: int = len(jax.devices()), pad_value: float = 0.0
):
    """
    Pad an array along the specified axis so its size is a multiple of the number of devices.

    Useful for distributed computation where data must be evenly split across devices.

    Parameters
    ----------
    x : ndarray
        Input array to pad.
    axis : int, optional
        Axis along which to pad (default is 0).

    Returns
    -------
    x_padded : ndarray
        Padded array with size along the specified axis a multiple of the number of devices.
    """
    if arr.shape[axis] % no_dev != 0:
        shape = list(arr.shape)
        shape[axis] = arr.shape[axis] % no_dev
        zero_arr = jnp.full(shape, pad_value)
        return jnp.concatenate((arr, zero_arr), axis=axis)
    else:
        return arr


def flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten a nested dictionary, concatenating keys with a separator.

    Parameters
    ----------
    d : dict
        The dictionary to flatten.
    parent_key : str, optional
        The base key string for recursion (default is '').
    sep : str, optional
        Separator to use between concatenated keys (default is '.').

    Returns
    -------
    dict
        A flattened dictionary with concatenated keys.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep='.').items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d, sep='.'):
    """
    Unflatten a dictionary with concatenated keys into a nested dictionary.

    Parameters
    ----------
    d : dict
        The flattened dictionary to unflatten.
    sep : str, optional
        Separator used in the keys (default is '.').

    Returns
    -------
    dict
        A nested dictionary reconstructed from the flattened keys.
    """
    result_dict = {}
    for k, v in d.items():
        keys = k.split(sep)
        d = result_dict
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = v
    return result_dict


def batch(iterable, n=1):
    """
    Yield successive n-sized batches from an iterable.

    Parameters
    ----------
    iterable : iterable
        The input iterable to batch.
    n : int, optional
        The batch size (default is 1).

    Yields
    ------
    list
        Lists of length n (or less for the last batch) from the iterable.
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def chunks(lst, n):
    """
    Yield successive n-sized chunks from a list.

    Parameters
    ----------
    lst : list
        The list to split into chunks.
    n : int
        The chunk size.

    Yields
    ------
    list
        Chunks of the input list of size n (or less for the last chunk).
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def merge_dicts(a, b):
    """
    Merge two dictionaries recursively, with values from b overwriting those from a.

    Parameters
    ----------
    a : dict
        The first dictionary.
    b : dict
        The second dictionary. Values from b take precedence.

    Returns
    -------
    dict
        The merged dictionary.
    """
    from collections.abc import Mapping

    for k, v in b.items():
        if isinstance(v, Mapping) and k in a:
            a[k] = merge_dicts(a[k], v)
        else:
            a[k] = v
    return a
