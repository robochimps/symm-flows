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
from flows.symmetry_functions import symmetrize_grid_c2v, getP, build_U_C2v
from flows.models.invertible_block import ActivationFunction, SingularValues
from flows.models.models import IResNet2, Linear, clip_kernel_svd_multiple, clip_kernel_svd, Tanh2
from flows.basis.direct_basis import generate_prod_ind, hermite_f, dhermite_f

#molecule and potential and kinetic energy operators

NCOO = 6
pmax = 9
nblocks = 5 #no blocks flows

batch_size_coo = 15000
batch_size_qua = 100000
no_train_sets = 1

rref = jnp.array([1.20337419, 1.10377465, 1.10377465, 2.1265833,  2.1265833, 0.0])
no_points_per_set = [n for n in range(26, 26 + no_train_sets*2,2)]

#Create symmetry operator
permute = [jnp.array([[1,3],[2,4]])] #permute 1 with 2 and 3 with 4.
invert = [jnp.array([5])]            #invert coordinate 5
P_c2v = getP(permute,invert,NCOO)


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
    svd_method = SingularValues.PAR_CLIP_EQUIVARIANT, #SingularValues.PAR_CLIP
    group = P_c2v,
    #_wrapper = Identity, #No wrapper = Tanh2
)

ckpt_dir = f"h2co_se100_iresnet_nblocks_5_pmax_16_sym"
params_folder = './h2co_checkpoints'
loc = params_folder + f"/{ckpt_dir}/"+'params.json'
params = joblib.load(loc)
print('Params loaded')

model_r = lambda params, x: model.apply(params, x, inverse=True)
model_x = lambda params, x: model.apply(params, x, inverse=False)

for iset, n in enumerate(no_points_per_set):
    grid = Tasmanian.TasmanianSparseGrid()
    grid.makeGlobalGrid(
        NCOO, 0, n, "qptotal",
        "gauss-hermite", [2, 2, 2, 1, 1, 1]
    )
    x = grid.getPoints()
    w = grid.getQuadratureWeights()
    #w /= np.prod(np.exp(-(x**2)), axis=-1)
    
    print('len grid:',len(x))

def _jac_x(model, params, x_batch, **kwargs):
    def jac(x):
        return jax.jacrev(model, argnums=1)(params, x, **kwargs)

    return jax.vmap(jac, in_axes=0)(x_batch)

r = model_r(params, x)
delta_r = jnp.max(r, axis=0) - jnp.min(r, axis=0)
delta_q = jnp.max(x, axis=0) - jnp.min(x, axis=0)

jac_r = _jac_x(model_x, params, r)

DBGSM = jnp.einsum('gqr, g-> qr', jnp.abs(jac_r), w) * delta_r[None,:] / delta_q[:,None]**2

var2_H2CO = DBGSM/jnp.linalg.norm(DBGSM, axis=1)[:, None]


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

# Heatmaps
plt.imshow(var2_H2CO  + 1e-6, cmap=custom_cmap, norm=norm, origin='upper')
# Add log values as text inside each square
for i in range(var2_H2CO.shape[0]):
    for j in range(var2_H2CO.shape[1]):
        text = f"{np.log10(var2_H2CO[i,j]+1e-16):.2f}"
        plt.text(j, i, text, ha="center", va="center")

plt.xticks(ticks=range(len(x_labels_H2CO)), labels=x_labels_H2CO, fontsize=18)
plt.yticks(ticks=range(len(y_labels)), labels=y_labels, fontsize=18)

# Colorbar
cb = plt.colorbar(shrink = 0.8)
font_size = 16 # Adjust as appropriate.
cb.ax.tick_params(labelsize=font_size)
plt.tight_layout()
plt.savefig('Variance'+ckpt_dir+'.png', dpi = 1000)
plt.show()

