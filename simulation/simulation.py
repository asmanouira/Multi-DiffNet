# examples/simulation.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np

from multi_diffnet.data import MultiModalData


# ============================================================
# Helpers: symmetry, SPD checks
# ============================================================

def sym(A):
    return 0.5 * (A + A.T)


def eigminmax(A):
    w = np.linalg.eigvalsh(sym(A))
    return float(w.min()), float(w.max())


def cond_number(A):
    wmin, wmax = eigminmax(A)
    return float(wmax / max(wmin, 1e-15))


def enforce_min_eig(A, min_eig_target=0.2, eps=1e-12):
    A = sym(A)
    w = np.linalg.eigvalsh(A)
    lam_min = float(w.min())
    shift = 0.0

    if lam_min < min_eig_target:
        shift = (min_eig_target - lam_min) + eps
        A = A + shift * np.eye(A.shape[0])

    return sym(A), float(shift), lam_min


# ============================================================
# Block indexing for 2 omics x 2 tissues
# ============================================================

def make_blocks(p):
    return {
        "omic1_tissue1_A1": np.arange(0, p),
        "omic2_tissue1_B1": np.arange(p, 2 * p),
        "omic1_tissue2_A2": np.arange(2 * p, 3 * p),
        "omic2_tissue2_B2": np.arange(3 * p, 4 * p),
    }


def block_category(name_i, name_j):
    def parse(n):
        parts = n.split("_")
        omic = int(parts[0].replace("omic", ""))
        tissue = int(parts[1].replace("tissue", ""))
        return omic, tissue

    oi, ti = parse(name_i)
    oj, tj = parse(name_j)

    if name_i == name_j:
        return "intra_block"

    if ti == tj and oi != oj:
        return "intra_tissue_inter_omic"

    if ti != tj and oi == oj:
        return "inter_tissue_intra_omic"

    return "inter_tissue_inter_omic"


def all_category_keys():
    return [
        "intra_block",
        "intra_tissue_inter_omic",
        "inter_tissue_intra_omic",
        "inter_tissue_inter_omic",
    ]


# ============================================================
# Candidate edge pools
# ============================================================

def upper_pairs_within(idx):
    idx = np.asarray(idx)
    pairs = []

    for a in range(len(idx)):
        i = int(idx[a])
        for b in range(a + 1, len(idx)):
            j = int(idx[b])
            pairs.append((i, j))

    return pairs


def upper_pairs_between(idx1, idx2):
    idx1 = np.asarray(idx1)
    idx2 = np.asarray(idx2)

    pairs = []

    for i in idx1:
        for j in idx2:
            ii, jj = int(i), int(j)

            if ii < jj:
                pairs.append((ii, jj))
            elif jj < ii:
                pairs.append((jj, ii))

    pairs = list(set(pairs))
    pairs.sort()

    return pairs


def build_category_edge_pool(p):
    blocks = make_blocks(p)
    names = list(blocks.keys())

    pool = {k: [] for k in all_category_keys()}

    for a in range(len(names)):
        for b in range(a, len(names)):
            ni, nj = names[a], names[b]
            ci, cj = blocks[ni], blocks[nj]

            cat = block_category(ni, nj)

            if ni == nj:
                pairs = upper_pairs_within(ci)
            else:
                pairs = upper_pairs_between(ci, cj)

            pool[cat].extend(pairs)

    for k in pool:
        pool[k] = list(set(pool[k]))
        pool[k].sort()

    return pool, blocks


# ============================================================
# Sparse S0
# ============================================================

def sample_uniform_away_from_zero(rng, low, high, min_abs):
    for _ in range(10_000):
        x = float(rng.uniform(low, high))
        if abs(x) >= min_abs:
            return x

    x = float(rng.uniform(low, high))

    return x if abs(x) > 0 else min_abs


def pick_edges_from_pool(rng, pool_pairs, prop):
    m = int(np.round(float(prop) * len(pool_pairs)))
    m = int(np.clip(m, 0, len(pool_pairs)))

    if m == 0:
        return []

    idx = rng.choice(len(pool_pairs), size=m, replace=False)

    return [pool_pairs[i] for i in idx]


def make_structured_sparse_S0(
    d,
    rng,
    category_pool,
    prop_by_cat,
    weight_scale=0.4,
    min_abs_S0=0.1,
    diag_margin=1.0,
    min_eig_target=1e-4,
):
    S = np.zeros((d, d))
    chosen_all = []

    for cat, pairs in category_pool.items():
        prop = float(prop_by_cat.get(cat, 0.0))
        chosen = pick_edges_from_pool(rng, pairs, prop)
        chosen_all.extend([(i, j, cat) for i, j in chosen])

    for i, j, cat in chosen_all:
        w = sample_uniform_away_from_zero(
            rng,
            -weight_scale,
            weight_scale,
            min_abs_S0,
        )
        S[i, j] = w
        S[j, i] = w

    for i in range(d):
        S[i, i] = np.sum(np.abs(S[i, :])) + diag_margin

    S, _, _ = enforce_min_eig(
        S,
        min_eig_target=min_eig_target,
    )

    return S, chosen_all


# ============================================================
# Woodbury latent baseline
# ============================================================

def theta0_from_S0_and_U(S0, U):
    S0 = sym(S0)

    SU = S0 @ U
    M = np.eye(U.shape[1]) + U.T @ SU

    Theta0 = S0 - SU @ np.linalg.inv(M) @ SU.T

    return sym(Theta0)


def L0_true_from_S0_and_U(S0, U):
    S0 = sym(S0)

    SU = S0 @ U
    M = np.eye(U.shape[1]) + U.T @ SU

    L0 = SU @ np.linalg.inv(M) @ SU.T

    return sym(L0)


# ============================================================
# Structured Delta
# ============================================================

def make_structured_delta(
    d,
    rng,
    category_pool,
    prop_by_cat,
    delta_range=(-1.5, 1.5),
    min_abs_delta=0.5,
    share_support_with=None,
):
    D = np.zeros((d, d))
    support = {}

    low, high = delta_range

    for cat, pairs in category_pool.items():
        prop = float(prop_by_cat.get(cat, 0.0))

        if share_support_with is not None and cat in share_support_with:
            chosen = share_support_with[cat]
        else:
            chosen = pick_edges_from_pool(rng, pairs, prop)

        support[cat] = chosen

        for i, j in chosen:
            w = sample_uniform_away_from_zero(
                rng,
                low,
                high,
                min_abs_delta,
            )
            D[i, j] = w
            D[j, i] = w

    return sym(D), support


# ============================================================
# One fixed simulation scenario
# ============================================================

def simulate_one_scenario(
    p=20,
    r=2,
    n0=100,
    n_low=100,
    n_high=100,
    seed=0,
    verbose=True,
):
    """
    Simulate one structured MultiDiffNet scenario.

    This is a simplified single-scenario version of the original
    DiffNet Woodbury simulator.

    Returns
    -------
    data : MultiModalData
        Object directly usable by MultiDiffNet.fit(data).

    truth : dict
        Dictionary containing true precision matrices,
        differential matrices, latent component, and block structure.
    """

    rng = np.random.default_rng(seed)

    d = 4 * p

    category_pool, blocks = build_category_edge_pool(p)

    propS0 = dict(
        intra_block=0.10,
        intra_tissue_inter_omic=0.00,
        inter_tissue_intra_omic=0.00,
        inter_tissue_inter_omic=0.00,
    )

    propDl = dict(
        intra_block=0.10,
        intra_tissue_inter_omic=0.00,
        inter_tissue_intra_omic=0.00,
        inter_tissue_inter_omic=0.00,
    )

    propDh = dict(
        intra_block=0.20,
        intra_tissue_inter_omic=0.00,
        inter_tissue_intra_omic=0.00,
        inter_tissue_inter_omic=0.00,
    )

    # latent loading matrix U
    A1 = rng.normal(0.1, 0.01, size=(p, r))
    B1 = rng.normal(0.8, 0.01, size=(p, r))
    A2 = rng.normal(0.4, 0.01, size=(p, r))
    B2 = rng.normal(1.8, 0.01, size=(p, r))

    U = np.vstack([A1, B1, A2, B2])

    # sparse baseline component S0
    S0, chosenS0 = make_structured_sparse_S0(
        d=d,
        rng=rng,
        category_pool=category_pool,
        prop_by_cat=propS0,
        weight_scale=0.4,
        min_abs_S0=0.1,
        diag_margin=1.0,
        min_eig_target=1e-4,
    )

    # baseline precision matrix
    Theta0_raw = theta0_from_S0_and_U(S0, U)

    Theta0, shift0, lammin0 = enforce_min_eig(
        Theta0_raw,
        min_eig_target=0.3,
    )

    L0_true = L0_true_from_S0_and_U(S0, U)

    # differential networks
    Delta_low_raw, supDl = make_structured_delta(
        d=d,
        rng=rng,
        category_pool=category_pool,
        prop_by_cat=propDl,
        delta_range=(-1.5, 1.5),
        min_abs_delta=0.5,
        share_support_with=None,
    )

    Delta_high_raw, supDh = make_structured_delta(
        d=d,
        rng=rng,
        category_pool=category_pool,
        prop_by_cat=propDh,
        delta_range=(-2.0, 2.0),
        min_abs_delta=0.5,
        share_support_with=supDl,
    )

    Theta_low_raw = sym(Theta0 + Delta_low_raw)
    Theta_high_raw = sym(Theta0 + Delta_high_raw)

    lam_min_all = min(
        np.linalg.eigvalsh(Theta_low_raw).min(),
        np.linalg.eigvalsh(Theta_high_raw).min(),
    )

    global_shift = max(
        0.0,
        0.3 - lam_min_all + 1e-12,
    )

    Theta_low = Theta_low_raw + global_shift * np.eye(d)
    Theta_high = Theta_high_raw + global_shift * np.eye(d)

    # simulated Gaussian data
    X0 = rng.multivariate_normal(
        np.zeros(d),
        np.linalg.inv(Theta0),
        size=n0,
    )

    XL = rng.multivariate_normal(
        np.zeros(d),
        np.linalg.inv(Theta_low),
        size=n_low,
    )

    XH = rng.multivariate_normal(
        np.zeros(d),
        np.linalg.inv(Theta_high),
        size=n_high,
    )

    # Convert to MultiDiffNet format:
    # rows = features, columns = samples
    data_0 = X0.T

    data_K = np.vstack([XL, XH]).T

    labels = np.array(
        [0] * n_low + [1] * n_high,
        dtype=int,
    )

    dims = {
        "omic1_tissue1_A1": p,
        "omic2_tissue1_B1": p,
        "omic1_tissue2_A2": p,
        "omic2_tissue2_B2": p,
    }

    feature_names = {
        name: [f"{name}_f{i}" for i in range(p)]
        for name in dims
    }

    data = MultiModalData(
        data_0=data_0,
        data_K=data_K,
        labels=labels,
        K=2,
        dims=dims,
        group_names=np.array(["low", "high"]),
        feature_names=feature_names,
        sample_names_baseline=[f"baseline_{i}" for i in range(n0)],
        sample_names_exposed=[
            *[f"low_{i}" for i in range(n_low)],
            *[f"high_{i}" for i in range(n_high)],
        ],
        baseline_group="baseline",
    )

    truth = dict(
        Theta0=Theta0,
        Theta_low=Theta_low,
        Theta_high=Theta_high,
        Delta_low=Theta_low - Theta0,
        Delta_high=Theta_high - Theta0,
        U=U,
        S0=S0,
        L0_true=L0_true,
        chosenS0=chosenS0,
        supDl=supDl,
        supDh=supDh,
        blocks=blocks,
        category_pool=category_pool,
        params=dict(
            p=p,
            r=r,
            d=d,
            n0=n0,
            n_low=n_low,
            n_high=n_high,
            seed=seed,
            propS0=propS0,
            propDl=propDl,
            propDh=propDh,
        ),
    )

    if verbose:
        print("=== One simulated MultiDiffNet scenario ===")
        print(f"p per block = {p}")
        print(f"total dimension d = {d}")
        print(f"latent rank r = {r}")
        print(f"n0 = {n0}, n_low = {n_low}, n_high = {n_high}")
        print(f"lambda_min Theta0_raw = {lammin0:.3e}")
        print(f"baseline shift = {shift0:.3e}")
        print(f"group global shift = {global_shift:.3e}")
        print(
            "condition numbers:",
            cond_number(Theta0),
            cond_number(Theta_low),
            cond_number(Theta_high),
        )

    return data, truth