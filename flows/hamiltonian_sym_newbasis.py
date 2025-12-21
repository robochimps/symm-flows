import operator
from functools import partial

import jax
from jax import config
from jax import numpy as jnp
from .basis.direct_basis import combine_psi, combine_dpsi

config.update("jax_enable_x64", True)
config.update('jax_default_matmul_precision', 'highest')


def hamiltonian_podolsky(
    params,
    bas_info,
    grid_info,
    model_x,
    model_r,
    gmat_func,
    detgmat_func,
    ddetgmat_func,
    pot_func,
    U = None,
    overlap_func=None,
):
    """
    Construct the vibrational Hamiltonian matrix using the Podolsky form.

    Parameters
    ----------
    params : dict
        Model parameters.
    bas_info : tuple
        Basis information (quanta, list_quanta, psi_functions, dpsi_functions, ncoo).
    grid_info : tuple
        Grid information (batch_x, batch_w, batch_ind).
    model_x : callable
        Forward model mapping.
    model_r : callable
        Inverse model mapping.
    gmat_func : callable
        Function to compute G-matrix.
    detgmat_func : callable
        Function to compute determinant of G-matrix.
    ddetgmat_func : callable
        Function to compute derivative of determinant of G-matrix.
    pot_func : callable
        Potential energy function.
    U : ndarray, optional
        Symmetry transformation matrix (default: None).
    overlap_func : callable, optional
        Overlap function (default: None).

    Returns
    -------
    ndarray
        Hamiltonian matrix.
    """

    def matrix_elements(psi, dpsi, w, oper):
        gvib1, gvib2, gvib3, gcor, grot, pot, ovlp = oper
        keo_vib = (
            jnp.einsum("gik,gkl...,gjl,g->ij...", dpsi, gvib1, dpsi, w)
            + jnp.einsum("gik,gk...,gj,g->ij...", dpsi, gvib2, psi, w)
            + jnp.einsum("gi,gk...,gjk,g->ij...", psi, gvib2, dpsi, w)
            + jnp.einsum("gi,g...,gj,g->ij...", psi, gvib3, psi, w)
        )
        poten = jnp.einsum("gi,gj,g...,g->ij...", psi, psi, pot, w)
        #overlap = jnp.einsum("gi,gj,g...,g->ij...", psi, psi, ovlp, w)
        return 0.5 * keo_vib + poten

    def operators(params, x):
        def operators_spmd(r, gvib, grot, gcor, detg, ddetg, pot, ovlp):
            df = _jac_x(model_x, params, r)
            det = _abs_det_jac_x(model_x, params, r)
            ddet = _grad_abs_det_jac_x(model_x, params, r)
            idet = 1.0 / det

            A = (0.5 * jnp.einsum('g,ga->ga',idet,ddet) + jnp.einsum('g,ga->ga',-(1/4)*detg**(-1),ddetg))
            gvib1 = jnp.einsum('gka,gab,glb->gkl',df,gvib,df)
            gvib2 = jnp.einsum('gka,gab,gb->gk',df,gvib,A)
            gvib3 = jnp.einsum('ga,gab,gb->g',A,gvib,A)
            gcor  = jnp.einsum("gka,ga...->gk...", df, gcor)
            return gvib1, gvib2, gvib3, gcor, grot, pot, ovlp

        r = model_r(params, x)
        gvib, grot, gcor = gmat_func(r)
        detg = detgmat_func(r)
        ddetg = ddetgmat_func(r)
        pot = pot_func(r)

        if overlap_func is None:
            ovlp = jnp.ones(pot.shape)
        else:
            ovlp = overlap_func(r)

        return operators_spmd(r, gvib, grot, gcor, detg, ddetg, pot, ovlp)

    quanta,list_quanta,psi_functions,dpsi_functions,ncoo = bas_info
    batch_x,batch_w,batch_ind = grid_info
    
    def batch_coo(params, ibatch):
        x = batch_x[ibatch]
        w = batch_w[ibatch]

        psi_1d = [psi_functions[i](x[:, i], list_quanta[i]) for i in range(ncoo)]
        dpsi_1d = [dpsi_functions[i](x[:, i], list_quanta[i]) for i in range(ncoo)]

        psi = combine_psi(psi_1d, quanta)
        dpsi = combine_dpsi(psi_1d, dpsi_1d, quanta)
        
        if U is not None:
            psi = jnp.einsum('ij,gj->gi',U,psi)
            dpsi = jnp.einsum('ij,gjk->gik',U,dpsi)

        oper = operators(params, x)
        return matrix_elements(psi, dpsi, w, oper)
    
    def sum_batch_coo(params):
        ham = batch_coo(params, 0)
        ham = jax.lax.scan(
            lambda carry, ibatch: (carry + batch_coo(params, ibatch), 0),
            ham,
            batch_ind[1:],
        )[0]
        return ham

    return sum_batch_coo(params)#sum_batch_coo(jax.lax.stop_gradient(params))

def hamiltonian_trace_podolsky(
    params,
    bas_info,
    grid_info,
    model_x,
    model_r,
    gmat_func,
    detgmat_func,
    ddetgmat_func,
    pot_func,
    U = None,
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
    bas_info : tuple
        Basis information (quanta, list_quanta, psi_functions, dpsi_functions, ncoo).
    grid_info : tuple
        Grid information (batch_x, batch_w, batch_ind).
    model_x : callable
        Forward model mapping.
    model_r : callable
        Inverse model mapping.
    gmat_func : callable
        Function to compute G-matrix.
    detgmat_func : callable
        Function to compute determinant of G-matrix.
    ddetgmat_func : callable
        Function to compute derivative of determinant of G-matrix.
    pot_func : callable
        Potential energy function.
    U : ndarray, optional
        Symmetry transformation matrix (default: None).
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
    
    quanta,list_quanta,psi_functions,dpsi_functions,ncoo = bas_info
    batch_x,batch_w,batch_ind = grid_info

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

    def batch_coo(params, ibatch):
        def trace_h(
            rho, rho1_a, rho1_b, rho2, rho_h, w, r, gvib, grot, gcor, detg, ddetg, pot
        ):
            if overlap_func is None:
                ovlp = jnp.ones(pot.shape)
            else:
                ovlp = overlap_func(r)
            
            df = _jac_x(model_x, params, r)
            det = _abs_det_jac_x(model_x, params, r)
            ddet = _grad_abs_det_jac_x(model_x, params, r)
            idet = 1.0 / det

            A = (0.5 * jnp.einsum('g,ga->ga',idet,ddet) + jnp.einsum('g,ga->ga',-(1/4)*detg**(-1),ddetg))
            gvib1 = jnp.einsum('gka,gab...,glb->gkl...',df,gvib,df)
            gvib2 = jnp.einsum('gka,gab...,gb->gk...',df,gvib,A)
            gvib3 = jnp.einsum('ga,gab...,gb->g...',A,gvib,A)
            #gcor  = jnp.einsum("gka,ga...->gk...", df,gcor)
            
            keo_vib = (jnp.einsum("gab...,gab...,g->...", rho2,gvib1,w)
                + jnp.einsum("ga...,ga...,g->...", rho1_a,gvib2,w)
                + jnp.einsum("gb...,gb...,g->...", rho1_b,gvib2,w)
                + jnp.einsum("g...,g...,g->...", rho,gvib3,w))

            #keo_cor = jnp.einsum(
            #    "gk...,gka,ga...,g->...", rho1_a, df, gcor, w
            #) - jnp.einsum("gk...,gka,ga...,g->...", rho1_b, df, gcor, w)
            #keo_rot = jnp.einsum("g...,g...,g->...", rho, grot, w)
            poten = jnp.einsum("g...,g...,g->...", rho, pot, w)
            overlap = jnp.einsum("g...,g...,g->...", rho_h, ovlp, w)
            ham = 0.5 * keo_vib + poten
            return ham, overlap

        x = batch_x[ibatch]
        w = batch_w[ibatch]

        r = model_r(params, x)

        gvib, grot, gcor = gmat_func(r)
        detg = detgmat_func(r)
        ddetg = ddetgmat_func(r)
        pot = pot_func(r)

        if grot.ndim == 1:
            nbas_rot = 1
        else:
            nbas_rot = grot.shape[-1]

        psi_1d = [psi_functions[i](x[:, i], list_quanta[i]) for i in range(ncoo)]
        dpsi_1d = [dpsi_functions[i](x[:, i], list_quanta[i]) for i in range(ncoo)]

        psi = combine_psi(psi_1d, quanta)
        dpsi = combine_dpsi(psi_1d, dpsi_1d, quanta)

        if U is not None:
            psi = jnp.einsum('ij,gj->gi',U,psi)
            dpsi = jnp.einsum('ij,gjk->gik',U,dpsi)
        
        psi_h = jnp.einsum('ij,gi->gj',vec_h,psi)
        psi = jnp.einsum('ij,gi->gj',vec,psi)
        dpsi = jnp.einsum('ij,gik->gjk',vec,dpsi)
        
        rho = jnp.einsum('gi,gi->g',psi,psi)
        rho1_a = jnp.einsum('gik,gi->gk',dpsi,psi)
        rho1_b = jnp.einsum('gi,gik->gk',psi,dpsi)
        rho2 = jnp.einsum('gik,gil->gkl',dpsi,dpsi)
        rho_h = jnp.einsum('gi,gi->g',psi,psi_h)

        tr_h, tr_s = trace_h(
            *[
                jax.lax.stop_gradient(elem)
                for elem in (rho, rho1_a, rho1_b, rho2, rho_h, w)
            ],
            r,
            gvib,
            grot,
            gcor,
            detg,
            ddetg,
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
            batch_ind[1:],
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
            batch_ind[1:],
        )[0]
        tang_out = jax.tree_util.tree_reduce(
            operator.add,
            jax.tree_util.tree_map(lambda a, b: jnp.sum(a * b), tr_dot, params_dot),
        )
        return prim_out, tang_out

    return sum_batch_coo(params)

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

def _abs_det_jac_x(model, params, x_batch, **kwargs):
    """
    Compute the absolute value of the determinant of the Jacobian for a batch of inputs.

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
        Absolute determinant values for each input in the batch.
    """

    def det(x):
        return jnp.abs(jnp.linalg.det(jax.jacrev(model, argnums=1)(params, x, **kwargs)))
    return jax.vmap(det, in_axes=0)(x_batch)

def _grad_abs_det_jac_x(model, params, x_batch, **kwargs):
    """
    Compute the gradient of the absolute value of the determinant of the Jacobian for a batch of inputs.

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
        return jnp.abs(jnp.linalg.det(jax.jacrev(model, argnums=1)(params, x, **kwargs)))
    return jax.vmap(jax.grad(det), in_axes=0)(x_batch)

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

