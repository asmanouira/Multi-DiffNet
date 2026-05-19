"""
Model evaluation metrics for DiffNet-Latent.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import slogdet

from .operators import sym


def count_edges_offdiag(M, tol=1e-8):
    """
    Count undirected edges from off-diagonal entries.

    Parameters
    ----------
    M : ndarray
        Precision matrix.

    tol : float
        Threshold for nonzero edges.

    Returns
    -------
    int
    """
    A = np.abs(M) > tol

    np.fill_diagonal(A, False)

    return int(A.sum() // 2)


def ebic_single(
    S,
    Theta,
    n,
    gamma=0.5,
    edge_tol=1e-8,
):
    """
    Compute eBIC score for a Gaussian graphical model.

    Parameters
    ----------
    S : ndarray
        Empirical covariance matrix.

    Theta : ndarray
        Precision matrix.

    n : int
        Number of samples.

    gamma : float
        eBIC hyperparameter.

    edge_tol : float
        Edge threshold.

    Returns
    -------
    tuple
        (ebic, n_edges, log_likelihood)
    """
    p = Theta.shape[0]

    E = count_edges_offdiag(
        Theta,
        edge_tol,
    )

    sign, logdet = slogdet(sym(Theta))

    if sign <= 0:
        return np.inf, E, -np.inf

    ll = 0.5 * n * (
        logdet - float(np.trace(S @ Theta))
    )

    ebic = (
        -2.0 * ll
        + E * np.log(max(n, 2))
        + 4.0 * gamma * E * np.log(max(p, 2))
    )

    return float(ebic), int(E), float(ll)