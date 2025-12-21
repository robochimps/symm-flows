import operator
from functools import partial

import jax
from jax import config
from jax import numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from .utils import mesh, pad_devices_axis

config.update("jax_enable_x64", True)
config.update('jax_default_matmul_precision', 'highest')

def hamiltonian_quad_pot(
    params,
    model_x,
    model_r,
    basis,
    nbas,
    gmat_func,
    pot_func,
    pseudo_func=None,
    overlap_func=None,
):
    """
    Evaluate the Hamiltonian where the potential acts on quadrature space.

    Parameters
    ----------
    params : dict
        Model parameters.
    model_x : callable
        Forward model mapping.
    model_r : callable
        Inverse model mapping.
    basis : object
        Basis object with batching methods.
    nbas : int
        Number of basis functions.
    gmat_func : callable
        Function to compute G-matrix.
    pot_func : callable
        Potential energy function.
    pseudo_func : callable, optional
        Pseudopotential function (default: None).
    overlap_func : callable, optional
        Overlap function (default: None).

    Returns
    -------
    ndarray
        Hamiltonian and overlap matrices.
    """

    @partial(shard_map, mesh=mesh, in_specs=P("g"), out_specs=P())
    def matrix_elements(psi1, dpsi1, psi2, dpsi2, w, oper):
        gvib1, gvib2, gvib3, gvib4, gcor1, gcor2, grot, pseudo, pot, ovlp = oper
        keo_vib = (
            jnp.einsum("gik,gkl...,gjl,g->ij...", dpsi1, gvib1, dpsi2, w)
            + jnp.einsum("gik,gk...,gj,g->ij...", dpsi1, gvib2, psi2, w)
            + jnp.einsum("gi,gk...,gjk,g->ij...", psi1, gvib3, dpsi2, w)
            + jnp.einsum("gi,g...,gj,g->ij...", psi1, gvib4, psi2, w)
        )
        keo_cor = (
            jnp.einsum("gik,gk...,gj,g->ij...", dpsi1, gcor1, psi2, w)
            + jnp.einsum("gi,gk...,gjk,g->ij...", psi1, gcor1, dpsi2, w)
            + jnp.einsum("gi,g...,gj,g->ij...", psi1, gcor2, psi2, w)
        )
        keo_rot = jnp.einsum("gi,gj,g...,g->ij...", psi1, psi2, grot, w)
        poten = jnp.einsum("gi,gj,g...,g->ij...", psi1, psi2, pot, w)
        pseudo_pot = jnp.einsum("gi,gj,g...,g->ij...", psi1, psi2, pseudo, w)
        overlap = jnp.einsum("gi,gj,g...,g->ij...", psi1, psi2, ovlp, w)
        ham = 0.5 * (keo_vib + keo_rot + keo_cor) + pseudo_pot + poten
        return jax.lax.psum(jnp.array([ham, overlap]), axis_name="g")

    def batch_qua2(ibatch_qua1, psi, dpsi, w, oper):
        h_and_s = jax.lax.scan(
            lambda _, ibatch_qua2: (
                0,
                matrix_elements(
                    psi[ibatch_qua1],
                    dpsi[ibatch_qua1],
                    psi[ibatch_qua2],
                    dpsi[ibatch_qua2],
                    w,
                    oper,
                ),
            ),
            0,
            basis.batch_ind_qua,
        )[1]
        return jnp.concatenate(h_and_s, axis=2)

    def batch_qua1(psi, dpsi, w, oper):
        h_and_s = jax.lax.scan(
            lambda _, ibatch_qua1: (0, batch_qua2(ibatch_qua1, psi, dpsi, w, oper)),
            0,
            basis.batch_ind_qua,
        )[1]
        return jnp.concatenate(h_and_s, axis=1)[:, :nbas, :nbas]

    def operators(params, x):
        @partial(shard_map, mesh=mesh, in_specs=P("g"), out_specs=P("g"))
        def operators_spmd(r, gvib, grot, gcor, pseudo, pot, ovlp):
            df = _jac_x(model_x, params, r)
            dlog_det = _grad_log_abs_det_jac_x(model_x, params, r)
            gvib1 = jnp.einsum("gka,gab...,glb->gkl...", df, gvib, df)
            gvib2 = 0.5 * jnp.einsum("gka,gab...,gb->gk...", df, gvib, dlog_det)
            gvib3 = 0.5 * jnp.einsum("ga,gab...,gkb->gk...", dlog_det, gvib, df)
            gvib4 = 0.25 * jnp.einsum("ga,gab...,gb->g...", dlog_det, gvib, dlog_det)
            gcor1 = jnp.einsum("gka,ga...->gk...", df, gcor)
            gcor2 = jnp.einsum("ga,ga...->g...", dlog_det, gcor)
            return gvib1, gvib2, gvib3, gvib4, gcor1, gcor2, grot, pseudo, pot, ovlp

        r = model_r(params, x)
        gvib, grot, gcor = gmat_func(r)
        pot = pot_func(x)
        
        if pseudo_func is None:
            pseudo = jnp.zeros(pot.shape)
        else:
            pseudo = pseudo_func(r)

        if overlap_func is None:
            ovlp = jnp.ones(pot.shape)
        else:
            ovlp = overlap_func(r)

        return operators_spmd(r, gvib, grot, gcor, pseudo, pot, ovlp)

    def batch_coo(params, ibatch_coo):
        x = basis.batch_coo(ibatch_coo=ibatch_coo)
        x = jax.device_put(
            pad_devices_axis(x, axis=0, pad_value=0.0),
            NamedSharding(mesh, P("g")),  # !!! pad_value=? 0 or 1 or ...
        )

        w = basis.batch_weight(ibatch_coo=ibatch_coo)
        w = jax.device_put(pad_devices_axis(w, axis=0), NamedSharding(mesh, P("g")))

        psi = jax.lax.scan(
            lambda _, ibatch_qua: (
                0,
                basis.batch_psi(ibatch_coo=ibatch_coo, ibatch_qua=ibatch_qua),
            ),
            0,
            basis.batch_ind_qua,
        )[1]
        psi = jax.device_put(
            pad_devices_axis(psi, axis=1), NamedSharding(mesh, P(None, "g"))
        )

        dpsi = jax.lax.scan(
            lambda _, ibatch_qua: (
                0,
                basis.batch_dpsi(ibatch_coo=ibatch_coo, ibatch_qua=ibatch_qua),
            ),
            0,
            basis.batch_ind_qua,
        )[1]
        dpsi = jax.device_put(
            pad_devices_axis(dpsi, axis=1), NamedSharding(mesh, P(None, "g"))
        )

        oper = operators(params, x)
        return batch_qua1(psi, dpsi, w, oper)

    def sum_batch_coo(params):
        h_and_s = batch_coo(params, 0)
        h_and_s = jax.lax.scan(
            lambda carry, ibatch_coo: (carry + batch_coo(params, ibatch_coo), 0),
            h_and_s,
            basis.batch_ind_coo[1:],
        )[0]
        return h_and_s

    return sum_batch_coo(jax.lax.stop_gradient(params))

    

def hamiltonian(
    params,
    model_x,
    model_r,
    basis,
    nbas,
    gmat_func,
    pot_func,
    pseudo_func=None,
    overlap_func=None,
):
    """
    Construct the vibrational Hamiltonian matrix for a given basis and model.

    Parameters
    ----------
    params : dict
        Model parameters.
    model_x : callable
        Forward model mapping.
    model_r : callable
        Inverse model mapping.
    basis : object
        Basis object with batching methods.
    nbas : int
        Number of basis functions.
    gmat_func : callable
        Function to compute G-matrix.
    pot_func : callable
        Potential energy function.
    pseudo_func : callable, optional
        Pseudopotential function (default: None).
    overlap_func : callable, optional
        Overlap function (default: None).

    Returns
    -------
    ndarray
        Hamiltonian and overlap matrices.
    """

    @partial(shard_map, mesh=mesh, in_specs=P("g"), out_specs=P())
    def matrix_elements(psi1, dpsi1, psi2, dpsi2, w, oper):
        gvib1, gvib2, gvib3, gvib4, gcor, grot, pseudo, pot, ovlp = oper
        keo_vib = (
            jnp.einsum("gik,gkl...,gjl,g->ij...", dpsi1, gvib1, dpsi2, w)
            + jnp.einsum("gik,gk...,gj,g->ij...", dpsi1, gvib2, psi2, w)
            + jnp.einsum("gi,gk...,gjk,g->ij...", psi1, gvib3, dpsi2, w)
            + jnp.einsum("gi,g...,gj,g->ij...", psi1, gvib4, psi2, w)
        )
        keo_cor = jnp.einsum(
            "gik,gk...,gj,g->ij...", dpsi1, gcor, psi2, w
        ) - jnp.einsum("gi,gk...,gjk,g->ij...", psi1, gcor, dpsi2, w)
        keo_rot = jnp.einsum("gi,gj,g...,g->ij...", psi1, psi2, grot, w)
        poten = jnp.einsum("gi,gj,g...,g->ij...", psi1, psi2, pot, w)
        pseudo_pot = jnp.einsum("gi,gj,g...,g->ij...", psi1, psi2, pseudo, w)
        overlap = jnp.einsum("gi,gj,g...,g->ij...", psi1, psi2, ovlp, w)
        ham = 0.5 * (keo_vib + keo_rot + keo_cor) + pseudo_pot + poten
        return jax.lax.psum(jnp.array([ham, overlap]), axis_name="g")

    def batch_qua2(ibatch_qua1, psi, dpsi, w, oper):
        h_and_s = jax.lax.scan(
            lambda _, ibatch_qua2: (
                0,
                matrix_elements(
                    psi[ibatch_qua1],
                    dpsi[ibatch_qua1],
                    psi[ibatch_qua2],
                    dpsi[ibatch_qua2],
                    w,
                    oper,
                ),
            ),
            0,
            basis.batch_ind_qua,
        )[1]
        return jnp.concatenate(h_and_s, axis=2)

    def batch_qua1(psi, dpsi, w, oper):
        h_and_s = jax.lax.scan(
            lambda _, ibatch_qua1: (0, batch_qua2(ibatch_qua1, psi, dpsi, w, oper)),
            0,
            basis.batch_ind_qua,
        )[1]
        return jnp.concatenate(h_and_s, axis=1)[:, :nbas, :nbas]

    def operators(params, x):
        @partial(shard_map, mesh=mesh, in_specs=P("g"), out_specs=P("g"))
        def operators_spmd(r, gvib, grot, gcor, pseudo, pot, ovlp):
            df = _jac_x(model_x, params, r)
            dlog_det = _grad_log_abs_det_jac_x(model_x, params, r)
            gvib1 = jnp.einsum("gka,gab...,glb->gkl...", df, gvib, df)
            gvib2 = 0.5 * jnp.einsum("gka,gab...,gb->gk...", df, gvib, dlog_det)
            gvib3 = 0.5 * jnp.einsum("ga,gab...,gkb->gk...", dlog_det, gvib, df)
            gvib4 = 0.25 * jnp.einsum("ga,gab...,gb->g...", dlog_det, gvib, dlog_det)
            gcor = jnp.einsum("gka,ga...->gk...", df, gcor)
            return gvib1, gvib2, gvib3, gvib4, gcor, grot, pseudo, pot, ovlp

        r = model_r(params, x)
        gvib, grot, gcor = gmat_func(r)
        pot = pot_func(r)
        
        if pseudo_func is None:
            pseudo = jnp.zeros(pot.shape)
        else:
            pseudo = pseudo_func(r)

        if overlap_func is None:
            ovlp = jnp.zeros(pot.shape)
        else:
            ovlp = overlap_func(r)

        return operators_spmd(r, gvib, grot, gcor, pseudo, pot, ovlp)

    def batch_coo(params, ibatch_coo):
        x = basis.batch_coo(ibatch_coo=ibatch_coo)
        x = jax.device_put(
            pad_devices_axis(x, axis=0, pad_value=0.0),
            NamedSharding(mesh, P("g")),  # !!! pad_value=? 0 or 1 or ...
        )

        w = basis.batch_weight(ibatch_coo=ibatch_coo)
        w = jax.device_put(pad_devices_axis(w, axis=0), NamedSharding(mesh, P("g")))

        psi = jax.lax.scan(
            lambda _, ibatch_qua: (
                0,
                basis.batch_psi(ibatch_coo=ibatch_coo, ibatch_qua=ibatch_qua),
            ),
            0,
            basis.batch_ind_qua,
        )[1]
        psi = jax.device_put(
            pad_devices_axis(psi, axis=1), NamedSharding(mesh, P(None, "g"))
        )

        dpsi = jax.lax.scan(
            lambda _, ibatch_qua: (
                0,
                basis.batch_dpsi(ibatch_coo=ibatch_coo, ibatch_qua=ibatch_qua),
            ),
            0,
            basis.batch_ind_qua,
        )[1]
        dpsi = jax.device_put(
            pad_devices_axis(dpsi, axis=1), NamedSharding(mesh, P(None, "g"))
        )

        oper = operators(params, x)
        return batch_qua1(psi, dpsi, w, oper)

    def sum_batch_coo(params):
        h_and_s = batch_coo(params, 0)
        h_and_s = jax.lax.scan(
            lambda carry, ibatch_coo: (carry + batch_coo(params, ibatch_coo), 0),
            h_and_s,
            basis.batch_ind_coo[1:],
        )[0]
        return h_and_s

    return sum_batch_coo(jax.lax.stop_gradient(params))

def contraction_hamiltonian(
    model_x,
    model_r,
    basis,
    nbas,
    gmat_func,
    pot_func,
    contraction_mat,
    pseudo_func=None,
    overlap_func=None,
):
    """
    Construct the vibrational Hamiltonian matrix with a contracted basis.

    Parameters
    ----------
    model_x : callable
        Forward model mapping.
    model_r : callable
        Inverse model mapping.
    basis : object
        Basis object with batching methods.
    nbas : int
        Number of basis functions.
    gmat_func : callable
        Function to compute G-matrix.
    pot_func : callable
        Potential energy function.
    contraction_mat : ndarray
        Contraction matrix for basis transformation.
    pseudo_func : callable, optional
        Pseudopotential function (default: None).
    overlap_func : callable, optional
        Overlap function (default: None).

    Returns
    -------
    ndarray
        Hamiltonian and overlap matrices.
    """

    @partial(shard_map, mesh=mesh, in_specs=P("g"), out_specs=P())
    def matrix_elements(psi1, dpsi1, psi2, dpsi2, w, oper):
        gvib1, gvib2, gvib3, gvib4, gcor, grot, pseudo, pot, ovlp = oper
        keo_vib = (
            jnp.einsum("gik,gkl...,gjl,g->ij...", dpsi1, gvib1, dpsi2, w)
            + jnp.einsum("gik,gk...,gj,g->ij...", dpsi1, gvib2, psi2, w)
            + jnp.einsum("gi,gk...,gjk,g->ij...", psi1, gvib3, dpsi2, w)
            + jnp.einsum("gi,g...,gj,g->ij...", psi1, gvib4, psi2, w)
        )
        keo_cor = jnp.einsum(
            "gik,gk...,gj,g->ij...", dpsi1, gcor, psi2, w
        ) - jnp.einsum("gi,gk...,gjk,g->ij...", psi1, gcor, dpsi2, w)
        keo_rot = jnp.einsum("gi,gj,g...,g->ij...", psi1, psi2, grot, w)
        poten = jnp.einsum("gi,gj,g...,g->ij...", psi1, psi2, pot, w)
        pseudo_pot = jnp.einsum("gi,gj,g...,g->ij...", psi1, psi2, pseudo, w)
        overlap = jnp.einsum("gi,gj,g...,g->ij...", psi1, psi2, ovlp, w)
        ham = 0.5 * (keo_vib + keo_rot + keo_cor) + pseudo_pot + poten
        return jax.lax.psum(jnp.array([ham, overlap]), axis_name="g")

    def batch_qua2(ibatch_qua1, psi, dpsi, w, oper):
        h_and_s = jax.lax.scan(
            lambda _, ibatch_qua2: (
                0,
                matrix_elements(
                    psi[ibatch_qua1],
                    dpsi[ibatch_qua1],
                    psi[ibatch_qua2],
                    dpsi[ibatch_qua2],
                    w,
                    oper,
                ),
            ),
            0,
            basis.batch_ind_qua,
        )[1]
        return jnp.concatenate(h_and_s, axis=2)

    def batch_qua1(psi, dpsi, w, oper):
        h_and_s = jax.lax.scan(
            lambda _, ibatch_qua1: (0, batch_qua2(ibatch_qua1, psi, dpsi, w, oper)),
            0,
            basis.batch_ind_qua,
        )[1]
        return jnp.concatenate(h_and_s, axis=1)[:, :nbas, :nbas]

    def operators( x):
        @partial(shard_map, mesh=mesh, in_specs=P("g"), out_specs=P("g"))
        def operators_spmd(r, gvib, grot, gcor, pseudo, pot, ovlp):
            df = _jac_x_(model_x, r)
            dlog_det = _grad_log_abs_det_jac_x_(model_x, r)
            gvib1 = jnp.einsum("gka,gab...,glb->gkl...", df, gvib, df)
            gvib2 = 0.5 * jnp.einsum("gka,gab...,gb->gk...", df, gvib, dlog_det)
            gvib3 = 0.5 * jnp.einsum("ga,gab...,gkb->gk...", dlog_det, gvib, df)
            gvib4 = 0.25 * jnp.einsum("ga,gab...,gb->g...", dlog_det, gvib, dlog_det)
            gcor = jnp.einsum("gka,ga...->gk...", df, gcor)
            return gvib1, gvib2, gvib3, gvib4, gcor, grot, pseudo, pot, ovlp

        r = model_r(x)
        gvib, grot, gcor = gmat_func(r)
        pot = pot_func(r)
        
        if pseudo_func is None:
            pseudo = jnp.zeros(pot.shape)
        else:
            pseudo = pseudo_func(r)

        if overlap_func is None:
            ovlp = jnp.zeros(pot.shape)
        else:
            ovlp = overlap_func(r)

        return operators_spmd(r, gvib, grot, gcor, pseudo, pot, ovlp)

    def batch_coo(ibatch_coo):
        x = basis.batch_coo(ibatch_coo=ibatch_coo)
        x = jax.device_put(
            pad_devices_axis(x, axis=0, pad_value=0.0),
            NamedSharding(mesh, P("g")),  # !!! pad_value=? 0 or 1 or ...
        )

        w = basis.batch_weight(ibatch_coo=ibatch_coo)
        w = jax.device_put(pad_devices_axis(w, axis=0), NamedSharding(mesh, P("g")))

        psi = jax.lax.scan(
            lambda _, ibatch_qua: (
                0,
                basis.batch_psi(ibatch_coo=ibatch_coo, ibatch_qua=ibatch_qua),
            ),
            0,
            basis.batch_ind_qua,
        )[1]
        
        psi = jnp.einsum('lgi..., ij-> lgj...', psi, contraction_mat)
        psi = jax.device_put(
            pad_devices_axis(psi, axis=1), NamedSharding(mesh, P(None, "g"))
        )

        dpsi = jax.lax.scan(
            lambda _, ibatch_qua: (
                0,
                basis.batch_dpsi(ibatch_coo=ibatch_coo, ibatch_qua=ibatch_qua),
            ),
            0,
            basis.batch_ind_qua,
        )[1]
        dpsi = jnp.einsum('lgi..., ij-> lgj...', dpsi, contraction_mat)
        dpsi = jax.device_put(
            pad_devices_axis(dpsi, axis=1), NamedSharding(mesh, P(None, "g"))
        )
    
        oper = operators(x)
        return batch_qua1(psi, dpsi, w, oper)

    def sum_batch_coo():
        h_and_s = batch_coo(0)
        h_and_s = jax.lax.scan(
            lambda carry, ibatch_coo: (carry + batch_coo( ibatch_coo), 0),
            h_and_s,
            basis.batch_ind_coo[1:],
        )[0]
        return h_and_s

    return sum_batch_coo()

def hamiltonian_trace(
    params,
    model_x,
    model_r,
    basis,
    nbas,
    gmat_func,
    pot_func,
    pseudo_func=None,
    overlap_func=None,
    eigenvec=None,
    eigenvec_h=None,
):
    """
    Computes the trace of the Hamiltonian matrix, optionally in a transformed basis.

    Parameters
    ----------
    params : dict
        Model parameters.
    model_x : callable
        Forward model mapping.
    model_r : callable
        Inverse model mapping.
    basis : object
        Basis object with batching methods.
    nbas : int
        Number of basis functions.
    gmat_func : callable
        Function to compute G-matrix.
    pot_func : callable
        Potential energy function.
    pseudo_func : callable, optional
        Pseudopotential function (default: None).
    overlap_func : callable, optional
        Overlap function (default: None).
    eigenvec : ndarray, optional
        Basis set transformation matrix (default: None).
    eigenvec_h : ndarray, optional
        Transformed Hamiltonian matrix (default: None).

    Returns
    -------
    float
        Trace of the Hamiltonian matrix.
    """

    def batch_coo(params, ibatch_coo):

        @partial(shard_map, mesh=mesh, in_specs=P("g"), out_specs=P())
        def trace_h(
            rho, rho1_a, rho1_b, rho2, rho_h, w, r, gvib, grot, gcor, pseudo, pot
        ):
            if overlap_func is None:
                ovlp = jnp.zeros(pot.shape)
            else:
                ovlp = overlap_func(r)
            df = _jac_x(model_x, params, r)
            dlog_det = _grad_log_abs_det_jac_x(model_x, params, r)
            keo_vib = (
                jnp.einsum("gkl...,gka,gab...,glb,g->...", rho2, df, gvib, df, w)
                + 0.5
                * jnp.einsum(
                    "gk...,gka,gab...,gb,g->...", rho1_a, df, gvib, dlog_det, w
                )
                + 0.5
                * jnp.einsum(
                    "gk...,ga,gab...,gkb,g->...", rho1_b, dlog_det, gvib, df, w
                )
                + 0.25
                * jnp.einsum(
                    "g...,ga,gab...,gb,g->...", rho, dlog_det, gvib, dlog_det, w
                )
            )
            keo_cor = jnp.einsum(
                "gk...,gka,ga...,g->...", rho1_a, df, gcor, w
            ) - jnp.einsum("gk...,gka,ga...,g->...", rho1_b, df, gcor, w)
            keo_rot = jnp.einsum("g...,g...,g->...", rho, grot, w)
            poten = jnp.einsum("g...,g...,g->...", rho, pot, w)
            pseudo_pot = jnp.einsum("g...,g...,g->...", rho, pseudo, w)
            overlap = jnp.einsum("g...,g...,g->...", rho_h, ovlp, w)
            ham = 0.5 * (keo_vib + keo_rot + keo_cor) + pseudo_pot + poten
            return jax.lax.psum(jnp.array([ham, overlap]), axis_name="g")

        x = basis.batch_coo(ibatch_coo)
        x = jax.device_put(
            pad_devices_axis(x, axis=0, pad_value=0.0),
            NamedSharding(mesh, P("g")),  # !!! pad_values=? 0 or 1 or ...
        )

        w = basis.batch_weight(ibatch_coo)
        w = jax.device_put(pad_devices_axis(w, axis=0), NamedSharding(mesh, P("g")))

        r = model_r(params, x)

        gvib, grot, gcor = gmat_func(r)
        pot = pot_func(r)
        if pseudo_func is None:
            pseudo = jnp.zeros(pot.shape)
        else:
            pseudo = pseudo_func(r)

        if grot.ndim == 1:
            nbas_rot = 1
        else:
            nbas_rot = grot.shape[-1]

        if eigenvec is None:
            # when no `eigenvec` is passed in, assume identity matrix
            # of the size of ro-vibrational basis
            vec = jnp.transpose(
                jnp.eye(nbas * nbas_rot).reshape(nbas, nbas_rot, -1), (0, 2, 1)
            )
        else:
            vec = eigenvec

        if eigenvec_h is None:
            # when no `eigenvec_h` is passed in, assume identity matrix
            # of the size of ro-vibrational basis
            vec_h = jnp.transpose(
                jnp.eye(nbas * nbas_rot).reshape(nbas, nbas_rot, -1), (0, 2, 1)
            )
        else:
            vec_h = eigenvec_h

        rho, rho1_a, rho1_b, rho2 = basis.batch_dens(
            ibatch_coo, bra_vec=vec, ket_vec=vec, only_rho=False
        )
        rho = jax.device_put(pad_devices_axis(rho, axis=0), NamedSharding(mesh, P("g")))
        rho1_a = jax.device_put(
            pad_devices_axis(rho1_a, axis=0), NamedSharding(mesh, P("g"))
        )
        rho1_b = jax.device_put(
            pad_devices_axis(rho1_b, axis=0), NamedSharding(mesh, P("g"))
        )
        rho2 = jax.device_put(
            pad_devices_axis(rho2, axis=0), NamedSharding(mesh, P("g"))
        )

        rho_h = basis.batch_dens(ibatch_coo, bra_vec=vec, ket_vec=vec_h, only_rho=True)
        rho_h = jax.device_put(
            pad_devices_axis(rho_h, axis=0), NamedSharding(mesh, P("g"))
        )

        tr_h, tr_s = trace_h(
            *[
                jax.lax.stop_gradient(elem)
                for elem in (rho, rho1_a, rho1_b, rho2, rho_h, w)
            ],
            r,
            gvib,
            grot,
            gcor,
            pseudo,
            pot
        )

        tr_h = jnp.sum(tr_h)
        tr_s = jnp.sum(tr_s)

        return tr_h, tr_h - tr_s

    def grad_trace(params, ibatch):
        return jax.grad(lambda *args: batch_coo(*args)[1])(params, ibatch)

    @jax.custom_jvp
    def sum_batch_coo(params):
        tr = batch_coo(params, 0)[0]
        tr = jax.lax.scan(
            lambda carry, ibatch: (
                carry + batch_coo(params, ibatch)[0],
                0,
            ),
            tr,
            basis.batch_ind_coo[1:],
        )[0]
        return tr

    @sum_batch_coo.defjvp
    def sum_batch_coo_jvp(prim, tang):
        (params,) = prim
        (params_dot,) = tang
        prim_out = sum_batch_coo(params)
        tr_dot = grad_trace(params, 0)
        tr_dot = jax.lax.scan(
            lambda carry, ibatch: (
                jax.tree_util.tree_map(
                    lambda a, b: a + b,
                    carry,
                    grad_trace(params, ibatch),
                ),
                0,
            ),
            tr_dot,
            basis.batch_ind_coo[1:],
        )[0]
        tang_out = jax.tree_util.tree_reduce(
            operator.add,
            jax.tree_util.tree_map(lambda a, b: jnp.sum(a * b), tr_dot, params_dot),
        )
        return prim_out, tang_out

    return sum_batch_coo(params)


def dipole(
    params,
    model_x,
    model_r,
    basis,
    nbas,
    dipole_func,
    overlap_func=None,
):
    """
    Compute the dipole matrix elements for a given basis and model.

    Parameters
    ----------
    params : dict
        Model parameters.
    model_x : callable
        Forward model mapping.
    model_r : callable
        Inverse model mapping.
    basis : object
        Basis object with batching methods.
    nbas : int
        Number of basis functions.
    dipole_func : callable
        Dipole function.
    overlap_func : callable, optional
        Overlap function (default: None).

    Returns
    -------
    ndarray
        Dipole and overlap matrices.
    """

    @partial(shard_map, mesh=mesh, in_specs=P("g"), out_specs=P())
    def matrix_elements(psi1, psi2, w, oper):
        dipole_val, ovlp = oper
        
        dipole_mat = jnp.einsum("gi,gj,ga...,g->ija...", psi1, psi2, dipole_val, w)
        overlap = jnp.einsum("gi,gj,ga...,g->ija...", psi1, psi2, ovlp, w)
        return jax.lax.psum(jnp.array([dipole_mat, overlap]), axis_name="g")

    def batch_qua2(ibatch_qua1, psi, w, oper):
        h_and_s = jax.lax.scan(
            lambda _, ibatch_qua2: (
                0,
                matrix_elements(
                    psi[ibatch_qua1],
                    psi[ibatch_qua2],
                    w,
                    oper,
                ),
            ),
            0,
            basis.batch_ind_qua,
        )[1]
        return jnp.concatenate(h_and_s, axis=2)

    def batch_qua1(psi, w, oper):
        h_and_s = jax.lax.scan(
            lambda _, ibatch_qua1: (0, batch_qua2(ibatch_qua1, psi, w, oper)),
            0,
            basis.batch_ind_qua,
        )[1]
        return jnp.concatenate(h_and_s, axis=1)[:, :nbas, :nbas]

    def operators(params, x):
        r = model_r(params, x)
        dipole_val = dipole_func(r)
        if overlap_func is None:
            ovlp = jnp.ones(dipole_val.shape)
        else:
            ovlp = overlap_func(r)[:,None] * jnp.ones(dipole_val.shape[1])[None, :]
        return dipole_val, ovlp

    def batch_coo(params, ibatch_coo):
        x = basis.batch_coo(ibatch_coo=ibatch_coo)
        x = jax.device_put(
            pad_devices_axis(x, axis=0, pad_value=0.0),
            NamedSharding(mesh, P("g")),  # !!! pad_value=? 0 or 1 or ...
        )

        w = basis.batch_weight(ibatch_coo=ibatch_coo)
        w = jax.device_put(pad_devices_axis(w, axis=0), NamedSharding(mesh, P("g")))

        psi = jax.lax.scan(
            lambda _, ibatch_qua: (
                0,
                basis.batch_psi(ibatch_coo=ibatch_coo, ibatch_qua=ibatch_qua),
            ),
            0,
            basis.batch_ind_qua,
        )[1]
        psi = jax.device_put(
            pad_devices_axis(psi, axis=1), NamedSharding(mesh, P(None, "g"))
        )

        oper = operators(params, x)
        return batch_qua1(psi, w, oper)

    def sum_batch_coo(params):
        h_and_s = batch_coo(params, 0)
        h_and_s = jax.lax.scan(
            lambda carry, ibatch_coo: (carry + batch_coo(params, ibatch_coo), 0),
            h_and_s,
            basis.batch_ind_coo[1:],
        )[0]
        return h_and_s

    return sum_batch_coo(params) #Allow for gradient

def eigenvalues(hmat, smat=None, pert=None):
    """
    Compute eigenvalues and eigenvectors of a (generalized) eigenproblem.

    Parameters
    ----------
    hmat : ndarray
        Hamiltonian matrix.
    smat : ndarray, optional
        Overlap matrix (default: None).
    pert : ndarray, optional
        Perturbation matrix (default: None).

    Returns
    -------
    e : ndarray
        Eigenvalues.
    w1 : ndarray
        Eigenvectors (possibly transformed).
    w2 : ndarray
        Hamiltonian applied to eigenvectors (possibly transformed).
    """

    if smat is not None:
        s_diag, s_vec = jnp.linalg.eigh(smat)
        s_invsqrt = s_vec @ jnp.diag( 1 / jnp.sqrt(s_diag)) @ s_vec.T
        h = s_invsqrt @ hmat @ s_invsqrt
    else:
        s_invsqrt = jnp.eye(hmat.shape[0])
        h = hmat
    e, v = jnp.linalg.eigh(h)
    if pert is not None:
        v = pert @ v
    w1 = s_invsqrt @ v
    w2 = s_invsqrt @ h @ v
    return e, w1, w2


def _jac_x(model, params, x_batch, **kwargs):
    """
    Compute the Jacobian of the model with respect to x for a batch of inputs.

    Parameters
    ----------
    model : callable
        Model function.
    params : dict
        Model parameters.
    x_batch : ndarray
        Batch of input coordinates.
    **kwargs :
        Additional arguments to the model.

    Returns
    -------
    ndarray
        Jacobian matrices for each input in the batch.
    """

    def jac(x):
        return jax.jacrev(model, argnums=1)(params, x, **kwargs)

    return jax.vmap(jac, in_axes=0)(x_batch)

def _jac_x_(model, x_batch, **kwargs):
    """
    Compute the Jacobian of the model with respect to x (no params) for a batch of inputs.

    Parameters
    ----------
    model : callable
        Model function.
    x_batch : ndarray
        Batch of input coordinates.
    **kwargs :
        Additional arguments to the model.

    Returns
    -------
    ndarray
        Jacobian matrices for each input in the batch.
    """

    def jac(x):
        return jax.jacrev(model, argnums=0)(x, **kwargs)

    return jax.vmap(jac, in_axes=0)(x_batch)



def _grad_log_abs_det_jac_x(model, params, x_batch, **kwargs):
    """
    Compute the gradient of the log absolute value of the determinant of the Jacobian for a batch of inputs.

    Parameters
    ----------
    model : callable
        Model function.
    params : dict
        Model parameters.
    x_batch : ndarray
        Batch of input coordinates.
    **kwargs :
        Additional arguments to the model.

    Returns
    -------
    ndarray
        Gradients for each input in the batch.
    """

    def det(x):
        return jnp.log(
            jnp.abs(jnp.linalg.det(jax.jacrev(model, argnums=1)(params, x, **kwargs)))
        )

    return jax.vmap(jax.grad(det), in_axes=0)(x_batch)


def _grad_log_abs_det_jac_x_(model, x_batch, **kwargs):
    """
    Compute the gradient of the log absolute value of the determinant of the Jacobian for a batch of inputs (no params).

    Parameters
    ----------
    model : callable
        Model function.
    x_batch : ndarray
        Batch of input coordinates.
    **kwargs :
        Additional arguments to the model.

    Returns
    -------
    ndarray
        Gradients for each input in the batch.
    """

    def det(x):
        return jnp.log(
            jnp.abs(jnp.linalg.det(jax.jacrev(model, argnums=0)(x, **kwargs)))
        )

    return jax.vmap(jax.grad(det), in_axes=0)(x_batch)
