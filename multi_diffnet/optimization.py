from __future__ import annotations

import numpy as np
from numpy.linalg import norm

from .operators import (
    sym,
    symmetrize,
    fro_norm_tensor,
    l1_soft_threshold,
    l1_prox_offdiag,
    prox_rank_psd,
    prox_logdet,
    min_eig,
    make_spd,
    pd_inverse,
    pd_logdet,
    offdiag_l1,
    nuclear_norm_psd,
)


def initialize_group_statistics(data_K, labels, K):
    """
    Compute empirical statistics for exposed groups.

    Parameters
    ----------
    data_K : ndarray of shape (p, n_exposed)
        Concatenated feature matrix for non-baseline samples.

    labels : ndarray of shape (n_exposed,)
        Integer labels in ``{0, ..., K-1}`` for exposed samples.

    K : int
        Number of exposed groups.

    Returns
    -------
    tuple
        Mu : ndarray of shape (K, p)
            Group-specific empirical means.

        Sam_cov : ndarray of shape (p, p, K)
            Group-specific empirical covariance matrices.

        prob : ndarray of shape (K,)
            Proportion of exposed samples in each group.

        n_per_group : ndarray of shape (K,)
            Number of samples in each exposed group.
    """
    p, n1 = data_K.shape
    Mu = np.zeros((K, p))
    Sam_cov = np.zeros((p, p, K))
    prob = np.zeros(K)
    n_per_group = np.zeros(K, dtype=int)

    for k in range(K):
        idx = np.where(labels == k)[0]
        n_per_group[k] = idx.size
        prob[k] = idx.size / max(n1, 1)

        if idx.size == 0:
            Sam_cov[:, :, k] = np.eye(p) * 1e-5
            continue

        Xk = data_K[:, idx]
        muk = Xk.mean(axis=1)
        Ck = Xk - muk[:, None]
        Mu[k, :] = muk
        Sam_cov[:, :, k] = sym((Ck @ Ck.T) / max(idx.size, 1))

    return Mu, Sam_cov, prob, n_per_group


def baseline_admm_latent_only(
    Sigma0,
    lambda_1=0.1,
    mu0=2.5,
    rho=1.0,
    max_iter=500,
    tol=1e-4,
    rtol=1e-3,
):
    """
    Estimate the baseline precision matrix with a latent low-rank component.

    The baseline model is decomposed as:

    ``Theta_0 = S_0 - P_0``

    where ``S_0`` is sparse and ``P_0`` is positive semidefinite. The latent
    component returned as ``L_0`` is equal to ``-P_0``.

    Parameters
    ----------
    Sigma0 : ndarray of shape (p, p)
        Empirical covariance matrix of the baseline group.

    lambda_1 : float, default=0.1
        Sparsity penalty applied to the sparse baseline component ``S_0``.

    mu0 : float, default=2.5
        Nuclear norm penalty controlling the low-rank latent component.

    rho : float, default=1.0
        ADMM penalty parameter.

    max_iter : int, default=500
        Maximum number of ADMM iterations.

    tol : float, default=1e-4
        Absolute convergence tolerance.

    rtol : float, default=1e-3
        Relative convergence tolerance.

    Returns
    -------
    dict
        Dictionary containing ``Theta_0``, ``S_0``, ``P_0``, and ``L_0``.
    """
    p = Sigma0.shape[0]
    Theta0 = np.eye(p)
    S0 = np.eye(p)
    P0 = np.zeros((p, p))
    X = np.zeros((p, p))

    for _ in range(max_iter):
        Theta_prev = Theta0.copy()

        W = S0 - P0 - X - (1.0 / rho) * Sigma0
        Theta0 = prox_logdet(W, rho)
        S0 = l1_prox_offdiag(Theta0 + P0 + X, lambda_1 / rho)
        P0 = prox_rank_psd(S0 - X - Theta0, tau=mu0 / rho)
        X = X + Theta0 - S0 + P0

        r_pri = norm(Theta0 - S0 + P0, "fro")
        s_dual = rho * norm(Theta0 - Theta_prev, "fro")
        eps_pri = np.sqrt(p * p) * tol + rtol * max(
            norm(Theta0, "fro"), norm(S0, "fro"), norm(P0, "fro")
        )
        eps_dual = np.sqrt(p * p) * tol + rtol * rho * norm(X, "fro")

        if r_pri <= eps_pri and s_dual <= eps_dual:
            break

    P0 = prox_rank_psd(P0, tau=0.0)

    return {
        "Theta_0": make_spd(Theta0),
        "S_0": sym(S0),
        "P_0": sym(P0),
        "L_0": -sym(P0),
    }


def ama_xi_stable_vectorized(
    B,
    K_c,
    lambda_2_vec,
    lambda_3,
    kappa=1.0,
    maxiter_ama=200,
    eps=1e-4,
    Xi_init=None,
):
    """
    Update group-specific differential matrices using AMA.

    This routine estimates the tensor ``Xi`` corresponding to differential
    precision matrices ``Delta_k``. It applies sparsity penalties within each
    group and optional fusion penalties between groups.

    Parameters
    ----------
    B : ndarray of shape (p, p, K)
        Input tensor from the ADMM update.

    K_c : ndarray of shape (2, L) or None
        Pairwise group comparison matrix. Each column contains two group indices.

    lambda_2_vec : ndarray of shape (K,)
        Group-specific sparsity penalties.

    lambda_3 : float
        Fusion penalty between differential matrices.

    kappa : float, default=1.0
        AMA step or penalty parameter.

    maxiter_ama : int, default=200
        Maximum number of AMA iterations.

    eps : float, default=1e-4
        Convergence tolerance.

    Xi_init : ndarray of shape (p, p, K), optional
        Optional initial value for ``Xi``.

    Returns
    -------
    ndarray of shape (p, p, K)
        Updated differential matrix tensor.
    """
    p, _, K = B.shape
    lambda_2_vec = np.asarray(lambda_2_vec, dtype=float).reshape(-1)
    L = 0 if K_c is None else K_c.shape[1]

    Xi = np.zeros_like(B) if Xi_init is None else Xi_init.copy()
    V = np.zeros((p, p, max(1, L)))
    Delta_dual = np.zeros_like(V)

    if L > 0:
        e_k12 = np.zeros((K, L))
        for l in range(L):
            k1, k2 = int(K_c[0, l]), int(K_c[1, l])
            e_k12[k1, l] = 1.0
            e_k12[k2, l] = -1.0
    else:
        e_k12 = None

    mask_off = ~np.eye(p, dtype=bool)

    for _ in range(maxiter_ama):
        Xi_prev = Xi.copy()

        Z = (
            B + np.tensordot(Delta_dual, e_k12.T, axes=([2], [0]))
            if L > 0
            else B.copy()
        )

        Xi = Z.copy()

        # Sparse update for each group-specific differential matrix.
        for k in range(K):
            Xi_k = Xi[:, :, k]
            Xi_k[mask_off] = l1_soft_threshold(
                Xi_k[mask_off], lambda_2_vec[k] / kappa
            )
            np.fill_diagonal(Xi_k, 0.0)
            Xi[:, :, k] = sym(Xi_k)

        # Fusion update between pairs of exposed groups.
        if L > 0 and lambda_3 > 0:
            for l in range(L):
                k1, k2 = int(K_c[0, l]), int(K_c[1, l])
                Omega = sym(
                    Xi[:, :, k1] - Xi[:, :, k2] - Delta_dual[:, :, l] / kappa
                )
                np.fill_diagonal(Omega, 0.0)
                V[:, :, l] = l1_prox_offdiag(Omega, lambda_3 / kappa)
                np.fill_diagonal(V[:, :, l], 0.0)
                Delta_dual[:, :, l] += kappa * (
                    V[:, :, l] - Xi[:, :, k1] + Xi[:, :, k2]
                )

        rel = fro_norm_tensor(Xi - Xi_prev) / (fro_norm_tensor(Xi_prev) + 1e-12)
        if rel < eps:
            break

    return Xi


def update_thetaK_supervised_fixed(
    Sam_cov,
    Theta_0,
    lambda_2_vec,
    lambda_3,
    rho=1.0,
    maxiter_admm=50,
    maxiter_ama=200,
    eps=1e-4,
    init=None,
):
    """
    Update exposed-group precision and differential matrices.

    Conditional on the current baseline precision matrix ``Theta_0``, this
    function jointly estimates all exposed-group precision matrices
    ``Theta_k`` and their differential components ``Delta_k``.

    Parameters
    ----------
    Sam_cov : ndarray of shape (p, p, K)
        Empirical covariance matrices for exposed groups.

    Theta_0 : ndarray of shape (p, p)
        Baseline precision matrix.

    lambda_2_vec : ndarray of shape (K,)
        Sparsity penalties for group-specific differential matrices.

    lambda_3 : float
        Fusion penalty between differential matrices.

    rho : float, default=1.0
        ADMM penalty parameter.

    maxiter_admm : int, default=50
        Maximum number of ADMM iterations.

    maxiter_ama : int, default=200
        Maximum number of inner AMA iterations.

    eps : float, default=1e-4
        Convergence tolerance.

    init : dict, optional
        Optional warm start containing ``Theta``, ``Delta``, and ``Phi``.

    Returns
    -------
    dict
        Dictionary containing updated ``Theta``, ``Delta``, and ``Phi`` tensors.
    """
    p, _, K = Sam_cov.shape

    if init is None:
        Theta = np.stack([make_spd(Theta_0) for _ in range(K)], axis=-1)
        Xi = np.zeros((p, p, K))
        Phi = np.zeros((p, p, K))
    else:
        Theta = init["Theta"].copy()
        Xi = init["Delta"].copy()
        Phi = init["Phi"].copy()

    K_c = (
        np.array([(i, j) for i in range(K) for j in range(i + 1, K)]).T
        if K >= 2
        else None
    )

    for _ in range(maxiter_admm):
        Theta_prev = Theta.copy()
        Xi_prev = Xi.copy()

        for k in range(K):
            A = Theta_0 + Xi[:, :, k] - Phi[:, :, k] - (1.0 / rho) * Sam_cov[:, :, k]
            Theta[:, :, k] = make_spd(prox_logdet(A, rho))

        B = Theta + Phi - Theta_0[:, :, None]
        Xi = symmetrize(
            ama_xi_stable_vectorized(
                B, K_c, lambda_2_vec, lambda_3, rho, maxiter_ama, eps, Xi_prev
            )
        )

        Phi = Phi + Theta - Theta_0[:, :, None] - Xi

        relTheta = fro_norm_tensor(Theta - Theta_prev) / (
            fro_norm_tensor(Theta_prev) + 1e-12
        )
        relXi = fro_norm_tensor(Xi - Xi_prev) / (fro_norm_tensor(Xi_prev) + 1e-12)

        if relTheta < eps and relXi < eps:
            break

    return {"Theta": symmetrize(Theta), "Delta": symmetrize(Xi), "Phi": symmetrize(Phi)}


def joint_objective(
    Sigma0,
    Sam_cov,
    n0,
    n_per_group,
    S0,
    P0,
    Delta,
    lambda_1,
    lambda_2_vec,
    lambda_3,
    mu0,
    jitter=1e-6,
):
    """
    Compute the full joint objective function.

    The objective combines the baseline likelihood, exposed-group likelihoods,
    baseline sparsity, latent low-rank penalty, differential sparsity, and
    fusion penalty between groups.

    Parameters
    ----------
    Sigma0 : ndarray of shape (p, p)
        Baseline empirical covariance matrix.

    Sam_cov : ndarray of shape (p, p, K)
        Exposed-group empirical covariance matrices.

    n0 : int
        Number of baseline samples.

    n_per_group : ndarray of shape (K,)
        Number of samples per exposed group.

    S0 : ndarray of shape (p, p)
        Sparse baseline component.

    P0 : ndarray of shape (p, p)
        Positive semidefinite latent component.

    Delta : ndarray of shape (p, p, K)
        Differential precision matrices.

    lambda_1 : float
        Baseline sparsity penalty.

    lambda_2_vec : ndarray of shape (K,)
        Differential sparsity penalties.

    lambda_3 : float
        Fusion penalty between exposed groups.

    mu0 : float
        Nuclear norm penalty for the latent component.

    jitter : float, default=1e-6
        Numerical stabilization value.

    Returns
    -------
    float
        Value of the full joint objective.
    """
    Theta0 = make_spd(S0 - P0, eps=jitter)
    K = Delta.shape[2]

    obj = n0 * (-pd_logdet(Theta0, jitter) + float(np.trace(Sigma0 @ Theta0)))

    for k in range(K):
        Thetak = make_spd(Theta0 + Delta[:, :, k], eps=jitter)
        obj += int(n_per_group[k]) * (
            -pd_logdet(Thetak, jitter) + float(np.trace(Sam_cov[:, :, k] @ Thetak))
        )

    obj += float(lambda_1) * offdiag_l1(S0)
    obj += float(mu0) * nuclear_norm_psd(P0)

    for k in range(K):
        obj += float(lambda_2_vec[k]) * offdiag_l1(Delta[:, :, k])

    if K >= 2 and lambda_3 > 0:
        for k1 in range(K):
            for k2 in range(k1 + 1, K):
                obj += float(lambda_3) * offdiag_l1(
                    Delta[:, :, k1] - Delta[:, :, k2]
                )

    return float(obj)


def baseline_joint_update_linearized(
    Sigma0,
    Sam_cov,
    n0,
    n_per_group,
    S0,
    P0,
    Delta,
    lambda_1,
    mu0,
    step_init=1e-3,
    jitter=1e-6,
):
    """
    Update the baseline parameters using all exposed groups.

    This is a linearized proximal update for the baseline precision matrix.
    Unlike the initial baseline-only ADMM step, this update depends on both
    baseline samples and all exposed groups through their differential matrices.

    Parameters
    ----------
    Sigma0 : ndarray of shape (p, p)
        Baseline empirical covariance matrix.

    Sam_cov : ndarray of shape (p, p, K)
        Exposed-group empirical covariance matrices.

    n0 : int
        Number of baseline samples.

    n_per_group : ndarray of shape (K,)
        Number of samples per exposed group.

    S0 : ndarray of shape (p, p)
        Current sparse baseline component.

    P0 : ndarray of shape (p, p)
        Current PSD latent component.

    Delta : ndarray of shape (p, p, K)
        Current differential matrices.

    lambda_1 : float
        Baseline sparsity penalty.

    mu0 : float
        Nuclear norm penalty for the latent component.

    step_init : float, default=1e-3
        Initial gradient step size.

    jitter : float, default=1e-6
        Numerical stabilization value.

    Returns
    -------
    dict
        Updated baseline matrices ``S_0``, ``P_0``, ``L_0``, ``Theta_0``,
        and the selected step size ``step_used``.
    """
    Theta0 = make_spd(S0 - P0, eps=jitter)
    p, K = Theta0.shape[0], Delta.shape[2]

    G = n0 * (Sigma0 - pd_inverse(Theta0, jitter))

    for k in range(K):
        G += int(n_per_group[k]) * (
            Sam_cov[:, :, k] - pd_inverse(Theta0 + Delta[:, :, k], jitter)
        )

    G = sym(G)

    obj_old = joint_objective(
        Sigma0, Sam_cov, n0, n_per_group, S0, P0, Delta,
        lambda_1, np.zeros(K), 0.0, mu0, jitter
    )

    step = float(step_init)

    for _ in range(20):
        Z = Theta0 - step * G
        S_new, P_new = S0.copy(), P0.copy()

        for _ in range(25):
            S_new = l1_prox_offdiag(Z + P_new, step * lambda_1)
            P_new = prox_rank_psd(S_new - Z, tau=step * mu0)

        Theta_new = sym(S_new - P_new)
        me0 = min_eig(Theta_new)

        if me0 < jitter:
            bump = jitter - me0 + 1e-10
            Theta_new += bump * np.eye(p)
            S_new += bump * np.eye(p)

        if all(min_eig(Theta_new + Delta[:, :, k]) >= jitter for k in range(K)):
            obj_new = joint_objective(
                Sigma0, Sam_cov, n0, n_per_group, S_new, P_new, Delta,
                lambda_1, np.zeros(K), 0.0, mu0, jitter
            )

            if np.isfinite(obj_new) and obj_new <= obj_old + 1e-8:
                return {
                    "S_0": sym(S_new),
                    "P_0": sym(P_new),
                    "L_0": -sym(P_new),
                    "Theta_0": make_spd(S_new - P_new, jitter),
                    "step_used": step,
                }

        step *= 0.5

    return {
        "S_0": sym(S0),
        "P_0": sym(P0),
        "L_0": -sym(P0),
        "Theta_0": make_spd(S0 - P0, jitter),
        "step_used": 0.0,
    }