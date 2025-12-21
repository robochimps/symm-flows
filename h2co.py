import os
from functools import partial
import joblib
import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
import Tasmanian

from jax import config
from pyhami.keo import Molecule, batch_Gmat, batch_pseudo, Detgmat, dDetgmat
from pyhami.H2CO.h2co_AYTY import poten as potv
from pyhami.H2CO import coords_x2yz

from flows.models.linear import compute_a_b
from flows.hamiltonian import eigenvalues
from flows.hamiltonian_sym_newbasis import hamiltonian_podolsky as hamiltonian_podolsky_sym
from flows.hamiltonian_sym_newbasis import hamiltonian_trace_podolsky as hamiltonian_trace_podolsky_sym
from flows.symmetry_functions import getP, build_U_C2v
from flows.models.invertible_block import ActivationFunction, SingularValues
from flows.models.models import IResNet2, clip_kernel_svd_multiple
from flows.basis.direct_basis import generate_prod_ind, hermite_f, dhermite_f


config.update("jax_enable_x64", True)

#molecule and potential and kinetic energy operators

NCOO = 6
masses = np.array([12.0, 15.99491463, 1.00782505, 1.00782505]) ## Taken from exemol webpage
Molecule.masses = masses

def r_to_r2(r):
    return r.at[...,5].add(jnp.pi)

def coords_molecule(r):
    r2 = r_to_r2(r)
    return coords_x2yz.internal_to_cartesian(r2)

#Molecule.internal_to_cartesian = coords_x2yz.internal_to_cartesian
Molecule.internal_to_cartesian = coords_molecule

#rref = jnp.array([1.20337419, 1.10377465, 1.10377465, 2.1265833,  2.1265833, np.pi])
rref = jnp.array([1.20337419, 1.10377465, 1.10377465, 2.1265833,  2.1265833, 0.0])

def gmat(x):
    g = batch_Gmat(x)
    npoints = len(g)
    gvib = g[:, :NCOO, :NCOO]
    grot = jnp.zeros(npoints)
    gcor = jnp.zeros((npoints, NCOO))
    return gvib, grot, gcor

def _potential(x):
    x = r_to_r2(x)
    return potv(x) 

potential = jax.jit(jax.vmap(_potential))

def batch_pseudo_disp(x):
    return batch_pseudo(x)

def detg(x):
    return jax.jit(jax.vmap(Detgmat, in_axes=0))(x)

def ddetg(x):
    return jax.jit(jax.vmap(dDetgmat, in_axes=0))(x)

if __name__ == "__main__":
    restart = int(sys.argv[1])
    pmax = 9 # int(sys.argv[1])
    nblocks = 5 # int(sys.argv[2])#no blocks flows
    ckpt_dir = f"h2co_checkpoints/h2co_se100_iresnet_nblocks_{nblocks}_pmax_{pmax}_sym"
    
    batch_size_coo = 15000
    #batch_size_coo = 5000
    batch_size_qua = 100000
    no_train_sets = 1

    no_points_per_set = [n for n in range(26, 26 + no_train_sets*2,2)]
    no_points_per_set += [28]  # add testing set
    #no_points_per_set = [n for n in range(40, 40 + no_train_sets*2,2)]
    #no_points_per_set += [40]  # add testing set
    
    polyadd = np.array([2, 2, 2, 1, 1, 1])
    select_quanta = lambda ind: np.sum(np.array(ind) * polyadd[:len(ind)]) <= pmax
    list_quanta = [np.arange(pmax+1), np.arange(pmax+1), np.arange(pmax+1), np.arange(pmax+1), np.arange(pmax+1), np.arange(pmax+1)]
    quanta = generate_prod_ind(list_quanta, select_quanta)
    nbas = len(quanta)
    print("######### Number of basis functions #########")
    print(f"                   {nbas}")
    print("#############################################")
    
    basis_types = ['hermite','hermite','hermite','hermite','hermite','hermite']
    
    basis_map = {
        'hermite': (hermite_f, dhermite_f),
        }

    psi_functions = [basis_map[basis_types[i]][0] for i in range(NCOO)]
    dpsi_functions = [basis_map[basis_types[i]][1] for i in range(NCOO)]
    
    #Create symmetry operator
    permute = [jnp.array([[1,3],[2,4]])] #permute 1 with 2 and 3 with 4.
    invert = [jnp.array([5])]            #invert coordinate 5
    P_c2v = getP(permute,invert,NCOO)

    for iset, n in enumerate(no_points_per_set):
        grid = Tasmanian.TasmanianSparseGrid()
        grid.makeGlobalGrid(
            NCOO, 0, n, "qptotal",
            "gauss-hermite", [2, 2, 2, 1, 1, 1]
        )
        x = grid.getPoints()
        w = grid.getQuadratureWeights()
        w /= np.prod(np.exp(-(x**2)), axis=-1)
       
        print('len grid:',len(x))
        
        
        len_x = len(x)
        nbatch = int(np.ceil(len_x / batch_size_coo))
        
        pad = nbatch * batch_size_coo - len_x
        if pad > 0:
            x = np.append(x, np.zeros((pad, NCOO)), axis=0)
            w = np.append(w, np.zeros((pad)))
        
        if iset == 0:
            x_train = jnp.array(x.reshape(nbatch,batch_size_coo,NCOO))
            w_train = jnp.array(w.reshape(nbatch,batch_size_coo))
            ind_train = jnp.arange(nbatch)
            print('padded and batched train grid is of dimension:',np.shape(x_train))
        else:
            x_test = jnp.array(x.reshape(nbatch,batch_size_coo,NCOO))
            w_test = jnp.array(w.reshape(nbatch,batch_size_coo))
            ind_test = jnp.arange(nbatch)
            print('padded and batched test grid is of dimension:',np.shape(x_test))

    # flow model
    xmin = np.min(x_test, axis=(0, 1))
    xmax = np.max(x_test, axis=(0, 1))
    print(
        "Min and max values of quadrature coords accross all basis sets:\n", xmin, xmax
    )

    interval = jnp.array([[0, 5.0], [0, 5.0], [0, 5.0], [0, jnp.pi], [0, jnp.pi], [-jnp.pi, jnp.pi]])
    a = jnp.array([15.0, 15.0, 15.0, 15.0, 15.0, 15.0])
    b = jnp.array([-15.0,  -15.0, -15.0, -15.0, -15.0, -15.0])

    #Compute optimal xshift for zero blocks
    a_trans, b_trans = compute_a_b(a, b, interval)
    xshift = jnp.arctanh((rref-b_trans)/a_trans)
    xmax = jnp.array([15.0, 15.0, 15.0, 15.0, 15.0, 15.0])
    xshift = -xshift*xmax
    model = IResNet2(
        a=a,
        b=b,
        opt_a = False,
        opt_b = False,
        xmax=xmax,
        xshift=xshift,
        intervals=interval,
        features=[8, 8, NCOO],
        activations=[
            ActivationFunction.LIPSWISH,
            ActivationFunction.LIPSWISH,
            ActivationFunction.LIPSWISH,
        ],
        no_resnet_blocks=nblocks, #5
        no_inv_iters = 30,
        svd_method = SingularValues.PAR_CLIP_EQUIVARIANT, #SingularValues.PAR_CLIP_EQUIVARIANT
        group = P_c2v,
        #_wrapper = Identity, #No wrapper = Tanh2
    )
    
    x = np.zeros((1,NCOO))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if restart == 0:
        params = model.init(jax.random.PRNGKey(0), x)
        params = clip_kernel_svd_multiple(params, lipschitz_constant=0.1) #0.1
        epoch_start = 0
    elif restart == 1:
        print(f"restart from the latest-epoch parameters stored in folder '{ckpt_dir}'")
        params = joblib.load(f"{ckpt_dir}/"+'params.json') #Load from own folder
        with open(f"{ckpt_dir}/"+'loss') as f:
            for line in f:
                pass
            epoch_start = int(line.split('loss')[0])

    print('Starting at epoch: ', epoch_start)
    model_r = lambda params, x: model.apply(params, x, inverse=True)
    model_x = lambda params, x: model.apply(params, x, inverse=False)
    
    def get_r_min_max(params,x):
        r_batch = model_r(params,x[0])
        rmin = np.min(r_batch, axis=0)
        rmax = np.max(r_batch, axis=0)
        for n in range(1,len(x)):
            r_batch = model_r(params,x[n])
            rmin = np.minimum(rmin,np.min(r_batch, axis=0))
            rmax = np.maximum(rmax,np.max(r_batch, axis=0))
        return rmin,rmax 
    
    rmin,rmax = get_r_min_max(params,x_test)
    print("Min and max values of physical coords:\n", rmin, rmax)
    
    # loss and test functions
    no_states = 100
    no_states_A1 = 38

    def eigensolve(par, no_states):
        bas = quanta,list_quanta,psi_functions,dpsi_functions,NCOO
        grid = x_test,w_test,ind_test
        h = hamiltonian_podolsky_sym(par, bas, grid, model_x, model_r, gmat, detg, ddetg, potential)
        e, w1, w2 = eigenvalues(h)
        return e[:no_states], h, w1[:,:no_states]

    eigensolve_jit = partial(jax.jit, static_argnums=(1))(eigensolve)

    def eigensolve_sym(par, no_states, U):
        bas = quanta,list_quanta,psi_functions,dpsi_functions,NCOO
        grid = x_test,w_test,ind_test 
        h = hamiltonian_podolsky_sym(par, bas, grid, model_x, model_r, gmat, detg, ddetg, potential, U)
        e, w1, w2 = eigenvalues(h)
        return e[:no_states], h, w1[:,:no_states]

    eigensolve_sym_jit = partial(jax.jit, static_argnums=(1))(eigensolve_sym)

    def loss_grad_fn_sym(par, no_states, U): #Symmetry version
        bas = quanta,list_quanta,psi_functions,dpsi_functions,NCOO
        grid = x_train,w_train,ind_train
        h = hamiltonian_podolsky_sym(par, bas, grid, model_x, model_r, gmat, detg, ddetg, potential, U)
        e, w1, w2 = eigenvalues(h)
        return jax.value_and_grad(hamiltonian_trace_podolsky_sym)(
            par,
            bas,
            grid,
            model_x,
            model_r,
            gmat,
            detg,
            ddetg,
            potential,
            U=U,
            eigenvec=w1[:, :no_states],
            eigenvec_h=w2[:, :no_states],
        )

    def loss_grad_fn(par, no_states):
        bas = quanta,list_quanta,psi_functions,dpsi_functions,NCOO
        grid = x_train,w_train,ind_train #train
        h = hamiltonian_podolsky_sym(par, bas, grid, model_x, model_r, gmat, detg, ddetg, potential)
        e, w1, w2 = eigenvalues(h)
        return jax.value_and_grad(hamiltonian_trace_podolsky_sym)(
            par,
            bas,
            grid,
            model_x,
            model_r,
            gmat,
            detg,
            ddetg,
            potential,
            eigenvec=w1[:, :no_states],
            eigenvec_h=w2[:, :no_states],
        )

    print("First few eigenvalues on a test set using initial params:")
    e,h,w1 = eigensolve_jit(params, no_states)
    print('loss e',jnp.sum(e))
    print(e[0],e[:100]-e[0])

    U, labels, blocks = build_U_C2v(quanta, swap_pairs=[(1,2),(3,4)], inv_col=5)
    U_A1 = jnp.array(U[blocks["A1"],:])
    
    
    e_sym,h_sym,_ = eigensolve_sym_jit(params, no_states_A1, U_A1)
    print(e_sym[0], e_sym[:no_states_A1] - e_sym[0])
    print('loss train',jnp.sum(e_sym))

    model_r_batch = jax.vmap(model_r, in_axes=(None, 0))
    model_x_batch = jax.vmap(model_x, in_axes=(None, 0))

    @jax.jit
    def invertibility(params, x):
        def _inversion(params, x):
            r = model_r_batch(params, x)
            return model_x_batch(params, r)
        return jnp.max(jnp.abs(_inversion(params, x) - x))

    print(f"Invertibility {invertibility(params, x_test)}")

    # optimisation
    optx = optax.adam(learning_rate=0.001)
    opt_state = optx.init(params)
    
    @partial(jax.jit, static_argnums=(2,))
    def update_params(par, opt_state, no_states, U):
        loss_val, grad = loss_grad_fn(par, no_states)
        #loss_val, grad = loss_grad_fn_sym(par, no_states, U) #U included for sym
        updates, opt_state = optx.update(grad, opt_state)
        par = optax.apply_updates(par, updates)
        par = clip_kernel_svd_multiple(par, lipschitz_constant=0.9)
        #par = clip_kernel_svd(par, lipschitz_constant=0.9)
        return loss_val, par, opt_state

    print(f"input 'pmax' = {pmax}")
    out_file = open(f"{ckpt_dir}/energies_A1", "a")
    out_file2 = open(f"{ckpt_dir}/energies", "a")
    loss_file = open(f"{ckpt_dir}/loss", "a")
    
    for i in range(epoch_start,10001): 
        loss_val, params, opt_state = update_params(params, opt_state, no_states, np.zeros(1))
        #loss_val, params, opt_state = update_params(params, opt_state, no_states_A1, U_A1)
        print(i, loss_val)

        if i % 1000 == 0: #reinitialize Adam optimizer every 1000th iteration
            joblib.dump(params, f"{ckpt_dir}/"+f"params_{i}.json")
            opt_state = optx.init(params)

        if i % 30 == 0:
            e, h, _ = eigensolve_jit(params, no_states)
            e_sym,h_sym, _ = eigensolve_sym_jit(params, no_states_A1, U_A1)
            loss_val_A1 = np.sum(e_sym[:no_states_A1])
            loss_val = np.sum(e[:no_states])
            print("Test loss_A1:", loss_val_A1)
            print("Test loss_all:", loss_val)
            print("First few eigenvalues on a test set (sym):\n", e_sym[0], e_sym[:10] - e_sym[0])
            print("First few eigenvalues on a test set:\n", e[0], e[:10] - e[0])    
            error_inv = invertibility(params, x_train)
            print('Error inverse:',error_inv)
            
            rmin,rmax = get_r_min_max(params,x_test)
            print("Min and max values of physical coords:\n", rmin, rmax)

            out_file.write(
                f"{i:6d}" + " ".join(f"{elem:18.12f}" for elem in e_sym[:no_states_A1]) + "\n"
            )
            out_file.flush()
            os.fsync(out_file)

            out_file2.write(
                f"{i:6d}" + " ".join(f"{elem:18.12f}" for elem in e[:no_states]) + "\n"
            )
            out_file2.flush()
            os.fsync(out_file2)

            print_loss = "  ".join([
                f"{i}",
                f"loss_A1 {loss_val_A1:20.12f}",
                f"loss {loss_val:20.12f}",
                f"err inv {error_inv:1.2e}",
            ])

            loss_file.write(print_loss + "\n")
            loss_file.flush()
            os.fsync(loss_file)
            
            joblib.dump(params, f"{ckpt_dir}/"+'params.json')
            print(f"store updated parameters in folder '{ckpt_dir}'")

    out_file.close()
