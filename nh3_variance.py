import joblib
import jax
jax.config.update('jax_platforms', 'cpu')
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax
import Tasmanian
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
import flax.linen as nn

from pyhami.keo import Molecule, batch_Gmat, batch_dGmat, batch_pseudo, Detgmat, dDetgmat, com
from flows.models.linear import compute_a_b
from flows.models.invertible_block import ActivationFunction, SingularValues
from flows.models.models import IResNet2, Linear, clip_kernel_svd_multiple, clip_kernel_svd, Tanh2
from flows.basis.direct_basis import generate_prod_ind, hermite_f, dhermite_f
from flows.symmetry_functions import symmetrize_grid_c2v, build_P_G12, build_U_C2v


def tanh(x):
    return (jnp.exp(x) - jnp.exp(-x))/(jnp.exp(x) + jnp.exp(-x))

def arctanh(x):
    return .5 * jnp.log((1 + x)/ (1 - x))

class wrapper_sym(nn.Module):
    """
    Apply tanh/arctanh on coordinates 0,1,2,5 and a 2D radial tanh mapping
    on coordinates 3,4 (s4,s5).
    """
    R_max: float = 1.0#jnp.sqrt(6) * np.pi / 3

    @nn.compact
    def __call__(self, x, inverse: bool = False):
        assert x.shape[-1] == NCOO, "Expected last dim = NCOO"

        idx_rest = jnp.array([0, 1, 2, 5])
        x_rest = x[..., idx_rest]
        s4 = x[..., 3]
        s5 = x[..., 4]

        if not inverse:
            x_rest = tanh(x_rest)

            r = jnp.sqrt(s4**2 + s5**2 + 1e-12)     # avoid zero division
            r_prime = self.R_max * tanh(r)
            factor = jnp.where(r > 0.0, r_prime / r, 0.0)
            s4 = factor * s4
            s5 = factor * s5
        else:
            x_rest = arctanh(x_rest)

            r_prime = jnp.sqrt(s4**2 + s5**2 + 1e-12)
            eps = 1e-12
            arg = jnp.clip(r_prime / self.R_max, -1.0 + eps, 1.0 - eps)
            r = arctanh(arg)

            factor = jnp.where(r_prime > 0.0, r / r_prime, 0.0)
            s4 = factor * s4
            s5 = factor * s5

        x = x.at[..., idx_rest].set(x_rest)
        x = x.at[..., 3].set(s4)
        x = x.at[..., 4].set(s5)
        return x

#molecule and potential and kinetic energy operators
NCOO = 6
rref = jnp.array([1.01159999,1.01159999,1.01159999,0.0,0.0,0.0])
pmax = 9
nblocks = 5 #no blocks flows

batch_size_coo = 5000
batch_size_qua = 100000
no_train_sets = 1

no_points_per_set = [n for n in range(48, 48 + no_train_sets*2,2)]

polyadd = np.array([4,4,4,2,2,1])
select_quanta = lambda ind: np.sum(np.array(ind) * polyadd[:len(ind)]) <= pmax


P_g12 = build_P_G12()

interval = jnp.array([[0, 5.0], [0, 5.0], [0, 5.0], [-jnp.pi, jnp.pi], [-jnp.pi, jnp.pi], [-jnp.pi, jnp.pi]])
a = jnp.array([15.0, 15.0, 15.0, 15.0, 15.0, 15.0])
b = jnp.array([-15.0,  -15.0, -15.0, -15.0, -15.0, -15.0])

#Compute optimal xshift for zero blocks
a_trans, b_trans = compute_a_b(a, b, interval)
xshift = jnp.arctanh((rref-b_trans)/a_trans)
xmax = jnp.array([15.0, 15.0, 15.0, 15.0, 15.0, 15.0])
xshift = -xshift*xmax
model2 = IResNet2(
    a=a,
    b=b,
    opt_a = False,
    opt_b = False,
    xmax=xmax,
    xshift=xshift,
    intervals=interval,
    features=[8, 8, NCOO],
    activations=[ActivationFunction.LIPSWISH] * nblocks,
    no_resnet_blocks=nblocks, #5
    no_inv_iters = 30,
    svd_method = SingularValues.PAR_CLIP_EQUIVARIANT,
    group = jnp.array(P_g12),
    _wrapper = wrapper_sym,
)

ckpt_dir = f"nh3_se100_iresnet_nblocks_5_pmax_16_sym"
params_folder = './nh3_results'
loc = params_folder + f"/{ckpt_dir}/"+'params.json'
params2 = joblib.load(loc)
print('Params loaded')

model2_r = lambda params, x: model2.apply(params, x, inverse=True)
model2_x = lambda params, x: model2.apply(params, x, inverse=False)

for iset, n in enumerate(no_points_per_set):
    grid = Tasmanian.TasmanianSparseGrid()
    grid.makeGlobalGrid(
            NCOO, 0, n, "qptotal",
            "gauss-hermite", [4,4,4,2,2,1]#[2, 2, 2, 1, 1, 1]
    )
    x = grid.getPoints()
    w = grid.getQuadratureWeights()
    #w /= np.prod(np.exp(-(x**2)), axis=-1)
    
    print('len grid:',len(x))

def _jac_x(model, params, x_batch, **kwargs):
    def jac(x):
        return jax.jacrev(model, argnums=1)(params, x, **kwargs)

    return jax.vmap(jac, in_axes=0)(x_batch)

r = model2_r(params2, x)
delta_r = jnp.max(r, axis=0) - jnp.min(r, axis=0)
delta_q = jnp.max(x, axis=0) - jnp.min(x, axis=0)

jac_r = _jac_x(model2_x, params2, r)

DBGSM = jnp.einsum('gqr, g-> qr', jnp.abs(jac_r), w) * delta_r[None,:] / delta_q[:,None]**2

var2_NH3 = DBGSM/jnp.linalg.norm(DBGSM, axis=1)[:, None]

plt.rc('font', **{'family':'sans-serif', 'sans-serif': ['Helvetica'], 'size': 14},)
plt.rc('text', usetex=True)

norm = LogNorm()
colors = ["white", "yellow", "red"]
custom_cmap = LinearSegmentedColormap.from_list("white_red_blue", colors)

plt.figure(figsize=(6, 6))
x_labels_NH3  = ["$r_1$", "$r_2$", "$r_3$", "$s_4$", "$s_5$", "$\\rho$"] 
x_labels_H2CO = ["$r_1$", "$r_2$", "$r_3$", "$\\theta_1$", "$\\theta_2$", "$\\tau$"]
x_labels_ = [x_labels_H2CO, x_labels_NH3]
y_labels = ["$q_1$", "$q_2$", "$q_3$", "$q_4$", "$q_5$", "$q_6$"]

plt.imshow(var2_NH3  + 1e-6, cmap=custom_cmap, norm=norm, origin='upper')
for i in range(var2_NH3.shape[0]):
    for j in range(var2_NH3.shape[1]):
        text = f"{np.log10(var2_NH3[i,j]+1e-16):.2f}"
        plt.text(j, i, text, ha="center", va="center")

plt.xticks(ticks=range(len(x_labels_NH3)), labels=x_labels_NH3, fontsize=18)
plt.yticks(ticks=range(len(y_labels)), labels=y_labels, fontsize=18)

# Colorbar
cb = plt.colorbar(shrink = 0.8)
font_size = 16 # Adjust as appropriate.
cb.ax.tick_params(labelsize=font_size)
plt.tight_layout()
plt.savefig('Variance'+ckpt_dir+'.png', dpi = 1000)
plt.show()


