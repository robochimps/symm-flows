import jax
import jax.numpy as jnp
import numpy as np
import math
import scipy

from collections import defaultdict
from collections import defaultdict

jax.config.update("jax_enable_x64", True)


def reduce_orbits_from_maps(x, w, maps):
    """
    Reduce a set of points and weights by identifying orbits under symmetry maps.

    Parameters
    ----------
    x : ndarray
        Array of points.
    w : ndarray
        Array of weights.
    maps : ndarray
        Symmetry maps (group actions).

    Returns
    -------
    x_reduced : ndarray
        Reduced set of points (orbit representatives).
    w_reduced : ndarray
        Corresponding weights, summed over orbits.
    """
    N              = x.shape[0]
    rep_index      = jnp.min(maps, axis=0)
    keep_mask      = (jnp.arange(N) == rep_index)
    sorted_vals    = jnp.sort(maps, axis=0)
    changes        = sorted_vals[1:] != sorted_vals[:-1]
    orbit_size     = jnp.sum(changes, axis=0) + 1
    contrib_at_row = jnp.where(keep_mask, orbit_size, 0)
    return x[keep_mask], contrib_at_row[keep_mask] * w[keep_mask]

def symmetrize_grid_c2v(x,w,P):
    """
    Symmetrize a grid and weights under the C2v group.

    Parameters
    ----------
    x : ndarray
        Grid points.
    w : ndarray
        Weights.
    P : list of ndarray
        List of symmetry operations (matrices).

    Returns
    -------
    x_sym : ndarray
        Symmetrized grid points.
    w_sym : ndarray
        Symmetrized weights.
    """
    N,M = jnp.shape(x)
    maps = []
    for p in P:
        maps.append(row_map(jnp.dot(x,p), x))
    maps = jnp.stack(maps, axis=0)  # (N, G)
    return reduce_orbits_from_maps(x, w, maps)

def row_map(x1, x, batch=1024, atol=1e-10):
    """
    Return idx so that x1[k] == x[idx[k]] (within atol).
    Does the broadcasted compare in batches to avoid N×N.

    Parameters
    ----------
    x1 : ndarray
        Array of points to map.
    x : ndarray
        Reference array of points.
    batch : int, optional
        Batch size for comparison (default: 1024).
    atol : float, optional
        Absolute tolerance for comparison (default: 1e-10).

    Returns
    -------
    idx : ndarray
        Indices mapping x1 to x.
    """
    x1 = jnp.asarray(x1); x = jnp.asarray(x)
    N = x.shape[0]
    out = np.empty(N, dtype=np.int32)
    for s in range(0, N, batch):
        e = min(s + batch, N)
        sub = x1[s:e]  # (B,d)
        eq  = jnp.all(jnp.isclose(sub[:,None,:], x[None,:,:], atol=atol, rtol=0.0), axis=-1)  # (B,N)
        out[s:e] = np.asarray(jnp.argmax(eq, axis=1))
    return jnp.array(out)

def get_permute_mat(i, j, N):
    """
    Construct a permutation matrix for swapping indices i and j in N dimensions.

    Parameters
    ----------
    i : int
        First index to swap.
    j : int
        Second index to swap.
    N : int
        Dimension of the matrix.

    Returns
    -------
    P : ndarray
        Permutation matrix.
    """
    perm = jnp.arange(N)
    orig = perm                              # read-from copy to avoid cascading writes
    perm = perm.at[i].set(orig[j])           # set i-positions to original j-values
    perm = perm.at[j].set(orig[i])           # set j-positions to original i-values
    P = jnp.eye(N)[perm]
    return P

def get_inversion_mat(i, NCOO):
    """
    Construct a diagonal matrix for inversion (reflection) about coordinate i.

    Parameters
    ----------
    i : int
        Index to invert.
    NCOO : int
        Dimension of the matrix.

    Returns
    -------
    P : ndarray
        Inversion matrix.
    """
    list_ = jnp.ones(NCOO)
    list_ = list_.at[i].set(-1)   # set -1 for inversion (symmetric coordinate around 0)
    P = jnp.diag(list_)           # create matrix
    return P

def getP(permute, invert, NCOO):
    """
    Generate all possible symmetry operations from primitive symmetry matrices.

    Parameters
    ----------
    permute : list of tuple
        List of index pairs to permute.
    invert : list of tuple
        List of indices to invert.
    NCOO : int
        Number of coordinates.

    Returns
    -------
    P : list of ndarray
        List of symmetry operation matrices.
    """
    P = [jnp.eye(NCOO)] #Include identity
    for i,j in permute:
        P.append(get_permute_mat(i,j,NCOO)) #Include permutations (H1,H2)
    for i in invert:
        P.append(get_inversion_mat(*i,NCOO)) #Include inversion (tau -> -tau)

    #Generate all possible symmetry operations from primitive symmetry matrices
    for i in range(len(P)):
        for j in range(i,len(P)):
            p2 = jnp.dot(P[i],P[j])
            if any(jnp.array_equal(p2, p) for p in P) == False:
                P.append(p2)
    return P

# ----- main: build U for C2v(M) = {E, P12, I, P12·I} -----

def build_U_C2v(quanta, swap_pairs=[(1,2), (3,4)], inv_col=5,
                irrep_names=("A1","A2","B1","B2"),
                char_table=np.array([[ 1,  1,  1,  1],   # A1
                                     [ 1,  1, -1, -1],   # A2
                                     [ 1, -1,  1, -1],   # B1
                                     [ 1, -1, -1,  1]],  # B2
                                    dtype=int),
                tol=1e-12):
    """
    Build the symmetry-adapted basis U for the C2v(M) group.

    Parameters
    ----------
    quanta : ndarray
        Quantum number table.
    swap_pairs : list of tuple, optional
        Column pairs to swap for permutation (default: [(1,2), (3,4)]).
    inv_col : int, optional
        Column index for inversion (default: 5).
    irrep_names : tuple, optional
        Names of irreducible representations (default: ("A1","A2","B1","B2")).
    char_table : ndarray, optional
        Character table (default: standard C2v(M)).
    tol : float, optional
        Tolerance for normalization (default: 1e-12).

    Returns
    -------
    U : ndarray
        Orthonormal change of basis matrix.
    labels : ndarray
        Array of irrep names for each row of U.
    blocks : dict
        Mapping irrep name to row indices in U.
    """

    Q = np.asarray(quanta)
    n = Q.shape[0]

    # Group elements: order [E, P12, I, P12·I]
    pE  = np.arange(n)                        # identity permutation
    pP  = _perm_from_column_swaps(Q, swap_pairs)
    pI  = pE.copy()
    pPI = pP.copy()

    # Sign vectors from inversion parity on coordinate `inv_col`
    sE  = np.ones(n, dtype=int)
    sP  = np.ones(n, dtype=int)
    sI  = 1 - 2*((Q[:, inv_col] & 1) != 0)    # +1 for even, -1 for odd
    sPI = sI.copy()

    perms = [pE,  pP,  pI,  pPI]
    signs = [sE,  sP,  sI,  sPI]

    # Orbits under the *permutation* subgroup {E, P12}
    visited = np.zeros(n, dtype=bool)
    orbits = []
    for i in range(n):
        if visited[i]:
            continue
        imgs = np.unique(np.array([pE[i], pP[i]]))
        visited[imgs] = True
        orbits.append(imgs)

    # Build symmetry-adapted rows via projection operators
    U_rows = []
    labels = []
    for imgs in orbits:
        rep_i = int(imgs[0])  # orbit representative index
        # For each irrep r (row in char_table), build v_r ∝ Σ_g χ_r(g) D(g) e_rep
        for r, chi in enumerate(char_table):
            v = np.zeros(n, dtype=float)
            for g in range(4):
                j = perms[g][rep_i]      # image index under op g
                v[j] += chi[g] * signs[g][rep_i]  # accumulate with character and sign
            # normalize on orbit support; may vanish if projection incompatible with stabilizer
            norm2 = np.sum(v[imgs]**2)
            if norm2 > tol:
                v /= np.sqrt(norm2)
                U_rows.append(v)
                labels.append(irrep_names[r])

        # If the orbit has 2 elements, we expect 2 nonzero rows; if 1 element (fixed),
        # we expect 1 nonzero row. The selection above handles both.

    U = np.vstack(U_rows)
    # Sanity: U should be square (n rows for n states)
    if U.shape != (n, n):
        raise RuntimeError(f"Constructed U has shape {U.shape}, expected {(n,n)}. "
                           "Check duplicates or your symmetry definitions.")

    # Optional orthonormality check:
    # print("||U U^T - I|| =", np.linalg.norm(U @ U.T - np.eye(n)))

    labels = np.array(labels, dtype=object)
    blocks = {name: np.where(labels == name)[0] for name in irrep_names}
    return U, labels, blocks

def _row_index_map(quanta):
    """
    Map row tuple -> list of indices (to handle duplicates).

    Parameters
    ----------
    quanta : ndarray
        Quantum number table.

    Returns
    -------
    m : dict
        Mapping from row tuple to list of indices.
    """
    m = defaultdict(list)
    for idx, row in enumerate(quanta):
        m[tuple(row.tolist())].append(idx)
    return m

def _perm_from_column_swaps(quanta, swap_pairs):
    """
    Build the row permutation p induced by swapping columns in swap_pairs.

    Parameters
    ----------
    quanta : ndarray
        Quantum number table.
    swap_pairs : list of tuple
        List of column index pairs to swap.

    Returns
    -------
    p : ndarray
        Row permutation indices.
    """
    Q = np.asarray(quanta)
    n, d = Q.shape
    Qsw = Q.copy()
    # apply all column swaps
    for (i, j) in swap_pairs:
        Qsw[:, [i, j]] = Qsw[:, [j, i]]
    # map each swapped row back to an index of the original table
    mp = _row_index_map(Q)
    used_counts = defaultdict(int)
    p = np.empty(n, dtype=int)
    for k, row in enumerate(Qsw):
        key = tuple(row.tolist())
        lst = mp.get(key, [])
        if not lst:
            raise ValueError(f"No partner found in quanta for swapped row #{k}: {row}")
        c = used_counts[key]
        if c >= len(lst):
            raise ValueError("Duplicate rows cause ambiguity; please disambiguate.")
        p[k] = lst[c]
        used_counts[key] = c + 1
    return p

### NH3 ###
def g12_character_table():
    """
    Character table for permutation–inversion group G12 (isomorphic to D3h(M)).

    Returns
    -------
    irreps : list
        List of irreducible representation names.
    classes : list
        List of class names.
    classes_ms : list
        List of class members.
    character_table : ndarray
        Character table.
    """

    irreps = ["A1'", "A1''", "A2'", "A2''", "E'", "E''"]

    classes = ["E","(23)","(123)","E*","(23)*","(123)*"]

    classes_ms = [["E"],
                 ["(12)", "(23)", "(13)"],
                 ["(123)", "(132)"],
                 ["E*"],
                 ["(12)*", "(23)*", "(13)*"],
                 ["(123)*", "(132)*"]]

    # Characters by irrep, ordered to match `classes` above
    character_table = np.array([
         [ 1,  1,  1,  1,  1,  1],
         [ 1,  1,  1, -1, -1, -1],
         [ 1, -1,  1,  1, -1,  1],
         [ 1, -1,  1, -1,  1, -1],
         [ 2,  0, -1,  2,  0, -1],
         [ 2,  0, -1, -2,  0,  1]])

    return irreps, classes, classes_ms, character_table

def g12_ops():
    """
    Return the 12 operations of G12 with their data.

    Returns
    -------
    ops : dict
        Dictionary of operation names to operation data.
    """
    E     = [0,1,2]
    P12   = [1,0,2]
    P23   = [0,2,1]
    P13   = [2,1,0]
    P132  = [1,2,0] #P123
    P123  = [2,0,1] #P132

    ang_p = +2*np.pi/3
    ang_m = -2*np.pi/3

    ops = { "E":    dict(perm=E,    bend=("rot", 0.0),    tau_sign=+1),
           "(12)":  dict(perm=P12,  bend=("refl", ang_m), tau_sign=-1),
           "(23)":  dict(perm=P23,  bend=("refl", 0.0),   tau_sign=-1),
           "(13)":  dict(perm=P13,  bend=("refl", ang_p), tau_sign=-1),
           "(123)": dict(perm=P123, bend=("rot", ang_m),  tau_sign=+1),
           "(132)": dict(perm=P132, bend=("rot", ang_p),  tau_sign=+1),

           "E*":    dict(perm=E,    bend=("rot", 0.0),    tau_sign=-1),
           "(12)*": dict(perm=P12,  bend=("refl", ang_m), tau_sign=+1),
           "(23)*": dict(perm=P23,  bend=("refl", 0.0),   tau_sign=+1),
           "(13)*": dict(perm=P13,  bend=("refl", ang_p), tau_sign=+1),
           "(123)*":dict(perm=P123, bend=("rot", ang_m),  tau_sign=-1),
           "(132)*":dict(perm=P132, bend=("rot", ang_p),  tau_sign=-1)}
    return ops


def d_element(j, mp, m, beta):
    """
    Compute a single element d^j_{m',m}(beta) of the Wigner small-d matrix using the standard sum.

    Parameters
    ----------
    j : float
        Total angular momentum (can be half-integer).
    mp : float
        m' quantum number (can be half-integer).
    m : float
        m quantum number (can be half-integer).
    beta : float
        Rotation angle in radians.

    Returns
    -------
    float
        Value of the Wigner small-d matrix element.
    """
    # Half-angle trig
    c = np.cos(beta / 2.0)
    s = np.sin(beta / 2.0)

    # Integers that appear in factorials
    jm   = int(round(j + m))
    j_m  = int(round(j - m))
    jmp  = int(round(j + mp))
    j_mp = int(round(j - mp))

    # t range: t in [max(0, m - m'), min(j+m, j-m')]
    m_minus_mp = int(round(m - mp))
    t_min = max(0, m_minus_mp)
    t_max = min(jm, j_mp)

    # prefactor: sqrt[(j+m)!(j-m)!(j+m')!(j-m')!]
    log_pref = 0.5 * (math.lgamma(jm+1) + math.lgamma(j_m+1) + math.lgamma(jmp+1) + math.lgamma(j_mp+1))

    acc = 0.0
    for t in range(t_min, t_max + 1):
        # denominator: (j+m-t)! t! (m'-m+t)! (j-m'-t)!
        jm_t      = jm - t
        mp_m_t    = (-m_minus_mp) + t      # (m' - m + t)
        j_mp_t    = j_mp - t               # (j - m' - t)
        log_den   = math.lgamma(jm_t+1) + math.lgamma(t+1) + math.lgamma(mp_m_t+1) + math.lgamma(j_mp_t+1)

        # powers
        pow_c = int(round(2*j + m - mp - 2*t))
        pow_s = int(round(mp - m + 2*t))

        # sign (-1)^{t - m' + m} = (-1)^{t + (m - m')}
        sign = -1 if ((t + m_minus_mp) & 1) else 1
        acc += sign * np.exp(log_pref - log_den) * (c ** pow_c) * (s ** pow_s)
    return acc

def wigner_small_d(j, beta):
    """
    Full Wigner small-d matrix d^j(beta), shape (2j+1, 2j+1).
    j can be integer or half-integer (e.g. 3.5). Returns float64.
    Row/col order is m', m from -j..+j.
    """
    two_j = int(round(2*j))
    if not np.isclose(2*j, two_j):
        raise ValueError("j must be integer or half-integer so that 2*j is integer.")
    # m values: -j, -j+1, ..., +j  (step 1; half-integer allowed)
    m_vals = np.arange(-two_j, two_j+1, 2, dtype=float) / 2.0
    d = np.empty((two_j+1, two_j+1), dtype=float)
    for ri, mp in enumerate(m_vals):          # rows: m'
        for ci, m  in enumerate(m_vals):      # cols: m
            d[ri, ci] = d_element(j, mp, m, beta)
    return d

def rot_shell_matrix(N, phi):
    """
    Rotation in the (s4,s5) HO shell N = n4 + n5.
    Basis ordering: |k, N-k>, k=0..N (k is n4).
    This is d^{j}(beta) with j=N/2 and beta = 2*phi, mapped as:
        M[r, k] = d^j_{m'=r-j, m=k-j}(2*phi).
    """
    j = 0.5 * N
    beta = 2.0 * phi
    return wigner_small_d(j, beta)  # shape (N+1, N+1)

def refl_shell_matrix(N, alpha):
    """
    Reflection in the (s4, s5) plane across the line at angle `alpha`
    with respect to the s4-axis, acting within the 2D isotropic HO shell N=n4+n5.

    Basis ordering: |k, N-k>, k=0..N (i.e. k=n4, N-k=n5).

    Implements:  M = R(alpha) @ P_s4 @ R(-alpha)
    where
      - R(phi) = rot_shell_matrix(N, phi)  (your existing rotation)
      - P_s4   = diag((-1)^(N-k))  (flip s5 -> -s5)
    """
    # rotation by +alpha and -alpha (use transpose for inverse since R is orthogonal)
    R_plus  = rot_shell_matrix(N, alpha)
    R_minus = R_plus.T  # == rot_shell_matrix(N, -alpha) if rot is orthogonal

    # mirror across s4-axis: s5 -> -s5  ==> phase (-1)^{n5} with n5 = N - k
    P_s4 = np.diag([ 1.0 if ((N - k) % 2) == 0 else -1.0 for k in range(N+1) ])

    # conjugate the axis-mirror by the rotation
    return R_plus @ P_s4 @ R_minus


def apply_permute(quanta_row, perm, active):
    """
    Permute a subset of columns in a single row.
    """
    r = list(quanta_row)
    cols = tuple(active)
    perm = tuple(perm)
    m = len(cols)
    if len(perm) != m:
        raise ValueError(f"perm and cols must have same length (got {len(perm)} vs {m}).")

    old = [r[c] for c in cols]
    #OLD PERMUTE
    #new = [old[perm[k]] for k in range(m)]
    new = [None]*m
    for k in range(m):
        new[perm[k]] = old[k]

    for c, v in zip(cols, new):
        r[c] = v
    return tuple(r)

def build_orbit(quanta, i0, ops, stretch_cols, bend_cols):
    """
    Build one joint orbit (stretches × bend-shell), ignoring tau entirely.

    Returns
    -------
    idx_grid : (m, N+1) int array
        Indices into `quanta` for each (unique stretch image, k) with k=0..N.
    stretch_list : list of length m
        Unique stretch triples generated by the distinct permutations in `ops`.
    N : int
        Bend shell number, N = n4 + n5.
    """
    Q = np.asarray(quanta, dtype=int)
    rows = [tuple(r) for r in Q]
    pos  = {rows[i]: i for i in range(len(rows))}
    c4, c5 = bend_cols

    base = rows[i0]
    N    = int(base[c4] + base[c5])

    # spectators: everything except stretches & bends (no tau)
    all_cols   = set(range(Q.shape[1]))
    spect_cols = sorted(all_cols - set(stretch_cols) - set(bend_cols))
    spect_key  = tuple(base[c] for c in spect_cols)

    # collect unique S3 permutations (ignore stars—many ops share the same perm)
    seen_perm = set()
    perm_list = []
    for _, meta in ops.items():
        perm = tuple(meta["perm"])
        if perm not in seen_perm:
            seen_perm.add(perm)
            perm_list.append(perm)

    # unique stretch images under those perms
    base_st = tuple(base[c] for c in stretch_cols)
    st_set, stretch_list = set(), []
    #OLD PERMUTE
    #for perm in perm_list:
        #old = list(base_st)
        #st_img = tuple(old[perm[k]] for k in range(len(stretch_cols)))
        #if st_img not in st_set:
            #st_set.add(st_img)
            #stretch_list.append(st_img)
    for perm in perm_list:
        old = list(base_st)
        new = [None] * len(stretch_cols)
        for k, p in enumerate(perm):
            new[p] = old[k]
        st_img = tuple(new)
        if st_img not in st_set:
            st_set.add(st_img)
            stretch_list.append(st_img)

    m = len(stretch_list)

    # fill (m, N+1) index grid
    idx_grid = np.empty((m, N+1), dtype=int)
    for a, st in enumerate(stretch_list):
        for k in range(N+1):
            row = list(base)
            # set stretches
            for val, c in zip(st, stretch_cols):
                row[c] = val
            # set bends to (k, N-k)
            row[c4], row[c5] = k, N - k
            # restore spectators exactly
            for c, v in zip(spect_cols, spect_key):
                row[c] = v
            j = pos.get(tuple(row), -1)
            if j < 0:
                raise RuntimeError(
                    f"Missing basis row for stretch {st}, (n4,n5)=({k},{N-k}), spectators {spect_key}"
                )
            idx_grid[a, k] = j
    return idx_grid, stretch_list, N

def D_stretch(stretch_list, ops):
    """Return dict name -> (m,m) permutation matrix on stretch images for *all* ops."""
    m   = len(stretch_list)
    pos = {st: a for a, st in enumerate(stretch_list)}
    Dst = {}
    for name, meta in ops.items():
        perm = meta["perm"]
        M = np.zeros((m, m))
        for a, st in enumerate(stretch_list):
            old = list(st)
            #OLD PERMUTE
            #st_img = tuple(old[perm[k]] for k in range(len(st)))
            new = [None] * len(st)
            for k, p in enumerate(perm):
                new[p] = old[k]
            st_img = tuple(new)
            M[pos[st_img], a] = 1.0
        Dst[name] = M
    return Dst

def D_bend(N, ops):
    """Return dict name -> (N+1, N+1) for bend action of *all* ops."""
    Db = {}
    for name, meta in ops.items():
        kind, param = meta["bend"]
        if kind == "rot":
            Db[name] = rot_shell_matrix(N, param)
        elif kind == "refl":
            Db[name] = refl_shell_matrix(N, param)
        else:
            raise ValueError(f"Unknown bend kind: {kind}")
    return Db

def build_D_G12(quanta, idx_grid, stretch_list, N, ops, tau_col=5):
    """
    Reducible representation D(g) on one joint orbit (stretches × bend-shell × optional τ).
    - If τ not present in idx_grid (2D), starred ops reuse the base block.
    - If τ is present (2D or 3D) and you want to apply umbrella parity, we build P_tau and
      right-multiply it for any op with tau_sign == -1.
    """
    # shape bookkeeping
    # keep a *stable* order of names
    all_names  = list(ops.keys())
    base_names = [nm for nm in all_names if not nm.endswith('*')]

    # build stretch/bend blocks for *base* names in the same order
    Dst_base = D_stretch(stretch_list, {nm: ops[nm] for nm in base_names})  # (m,m)
    Db_base  = D_bend(N,              {nm: ops[nm] for nm in base_names})  # (NK,NK)

    # base kron blocks: (Dst ⊗ Db) because idx_grid[a,k,...] is flattened with k fastest
    base_blocks = {}
    for nm in base_names:
        base_blocks[nm] = np.kron(Dst_base[nm], Db_base[nm])  

    # fill all 12 ops in the order of ops.keys()
    D = {}
    for name in all_names:
        base_name = name[:-1] if name.endswith('*') else name
        if base_name not in base_blocks:
            raise KeyError(f"Missing base block for op '{name}' (base '{base_name}').")

        M = base_blocks[base_name].copy()
        D[name] = M
    return D

# --- projector & SALCs ---
def build_projector_full(irreps, irrep, character_table, classes_ms, D_by_name):
    idx_ir = irreps.index(irrep)
    dim     = character_table[idx_ir, 0]  # dim = χ(E)
    # map op name to class column
    class_col = {}
    for c_idx, names in enumerate(classes_ms):
        for nm in names:
            class_col[nm] = c_idx

    G = sum(len(names) for names in classes_ms)  # = 12
    acc = np.zeros_like(next(iter(D_by_name.values())))
    for nm, M in D_by_name.items():
        acc += character_table[idx_ir, class_col[nm]] * M
    return (dim / G) * acc

def rows_from_projector(P, tol=1e-10):
    U, S, _ = np.linalg.svd(P)
    keep = S > (1 - tol)
    return U[:, keep].T  # each row is a SALC

def fix_row_signs(U):
    U = U.copy()
    for r in range(U.shape[0]):
        j = np.argmax(np.abs(U[r]))
        if U[r, j] < 0: U[r] *= -1
    return U

def map_irrep_tau(irrep,irreps,character_table):
    index      = irreps.index(irrep)
    tau_char   = character_table[3] #odd tau functions transforms as A2'' (even as A1)
    irrep_char = character_table[index]
    char_new   = tau_char * irrep_char
    match = np.dot(character_table,char_new[:,None]).T/np.linalg.norm(character_table,axis=1)/np.linalg.norm(char_new)
    index_new = np.argmax(match)
    return irreps[index_new]

def build_U_G12(quanta, ops,stretch_cols, bend_cols, tau_col, tol=1e-10):
    """
    Build symmetry-adapted linear combos (SALCs) for the *full* G12 group
    (D3h(M)), acting on stretches×bend-shell with fixed spectators and tau
    included via the starred operations.

    Returns:
      U       : (n, n) orthonormal transformation matrix
      labels  : (n,) irrep names for each row (A1', A2', E', A1'', A2'', E'')
      blocks  : dict irrep -> np.array row indices for that block
    """
    irreps, classes, classes_ms, character_table = g12_character_table()
    #irreps, classes, classes_ms, character_table = s3_character_table()

    Q = np.asarray(quanta, dtype=int)
    n = len(Q)
    rows = [tuple(r) for r in Q]

    seen = np.zeros(len(quanta), dtype=bool)
    U_rows, labels = [], []

    for i0 in range(len(quanta)):
        if seen[i0]:
            continue

        idx_grid, stretch_list, N = build_orbit(quanta, i0, ops, stretch_cols, bend_cols)
        #orbit_idx = np.unique(idx_grid.ravel())        # <-- deduplicate indices
        orbit_idx = idx_grid.ravel(order="C")
        #a = idx_grid.ravel()
        #indexes = np.unique(a, return_index=True)[1]
        #orbit_idx = np.array([a[index] for index in sorted(indexes)])

        seen[orbit_idx] = True
        #print('quanta[orbit_idx,:] : ',quanta[orbit_idx,:])
        # Reducible representation D(g) on this orbit, for *all 12* ops
        D = build_D_G12(quanta, idx_grid, stretch_list, N, ops)
        # Optional sanity: projectors sum to identity on the orbit
        # (handy check while debugging numerical issues)
        I_orb = np.eye(orbit_idx.size)
        Psum  = np.zeros_like(I_orb)
        for ir in irreps:
            Psum += build_projector_full(irreps, ir, character_table, classes_ms, D)
        if np.linalg.norm(Psum - I_orb) > 1e-6:
            # Not fatal, but useful to warn
            print(f"[warn] projector sum != I on orbit starting at {i0}: "
                  f"||sum P - I||={np.linalg.norm(Psum - I_orb):.3e}")

        # Project for each irrep and collect SALCs

        sign_tau = -2 * (np.mod(quanta[orbit_idx,tau_col],2) - 0.5)
        assert np.abs(np.sum(sign_tau))==len(sign_tau), "SALC error, mixed tau"
        sign_tau = sign_tau[0]

        for ir in irreps:
            P = build_projector_full(irreps, ir, character_table, classes_ms, D)
            #print('Idempotency:',np.sum(np.abs(P@P-P)))
            #print('irrep:',ir)
            salcs = rows_from_projector(P, tol=tol)  # each row lives on this orbit
            ###Change ir if sign_tau == -1 ###
            if sign_tau == -1:
                ir_ = map_irrep_tau(ir,irreps,character_table)
            else:
                ir_ = ir
            for rvec in salcs:
                full = np.zeros(n)
                full[orbit_idx] = rvec
                U_rows.append(full)
                labels.append(ir_)

    # Stack and polish
    U = np.vstack(U_rows)
    U = fix_row_signs(U)

    # Final sanity
    if U.shape != (n, n):
        raise RuntimeError(
            f"U has shape {U.shape}, expected {(n, n)}. "
            "This usually means your basis is not closed per orbit under G12 "
            "(e.g., missing some stretch images, missing bend-shell members, "
            "or missing τ partners needed by starred ops)."
        )

    labels = np.array(labels, dtype=object)
    blocks = {ir: np.where(labels == ir)[0] for ir in irreps}
    return U, labels, blocks

def perm_block(perm):
    P = np.zeros((3,3))
    for i,j in enumerate(perm):
        P[i,j] = 1
    return P

def bend_block(bend):
    kind, ang = bend
    ang = -ang
    if kind == "rot":
        return np.array([[np.cos(ang), -np.sin(ang)],
                         [np.sin(ang),  np.cos(ang)]])
    elif kind == "refl":
        ca, sa = np.cos(ang), np.sin(ang)
        return np.array([[ ca, sa],
                         [ sa,-ca]])
    else:
        raise ValueError("Unknown bend op")

def tau_block(sign):
    return np.array([[float(sign)]])

def g12_matrix(op):
    P = perm_block(op["perm"])
    B = bend_block(op["bend"])
    T = tau_block(op["tau_sign"])
    return scipy.linalg.block_diag(P, B, T)

def build_P_G12():
    ops = g12_ops()  # your definition
    G12 = []
    for name, op in ops.items():
        G12.append(g12_matrix(op))
    return G12


