"""
Low-level mathematical operators and proximal functions
used in DiffNet-Latent.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import eigh


def sym(A):
    """
    Symmetrize a square matrix.

    Parameters
    ----------
    A : ndarray

    Returns
    -------
    ndarray
    """
    return 0.5 * (A + A.T)


def symmetrize(X):
    """
    Symmetrize a matrix or tensor.

    Parameters
    ----------
    X : ndarray

    Returns
    -------
    ndarray
    """
    if X.ndim == 2:
        return sym(X)

    if X.ndim == 3:
        return 0.5 * (X + X.swapaxes(0, 1))

    raise ValueError(f"Unsupported tensor rank {X.ndim}")


def fro_norm_tensor(X):
    """
    Frobenius norm of a tensor.

    Parameters
    ----------
    X : ndarray

    Returns
    -------
    float
    """
    return float(np.sqrt(np.sum(np.asarray(X) ** 2)))


def l1_soft_threshold(x, lam):
    """
    Element-wise soft-thresholding operator.
    """
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def l1_prox_offdiag(M, lam):
    """
    Apply L1 proximal operator on off-diagonal entries only.
    """
    M = sym(M)

    out = M.copy()

    if lam > 0:
        mask = ~np.eye(M.shape[0], dtype=bool)
        out[mask] = l1_soft_threshold(out[mask], lam)

    np.fill_diagonal(out, np.diag(M))

    return sym(out)


def prox_rank_psd(C, tau):
    """
    Nuclear norm proximal operator on PSD matrices.
    """
    C = sym(C)

    d, Q = eigh(C)

    return sym(
        Q @ np.diag(np.maximum(d - tau, 0.0)) @ Q.T
    )


def prox_logdet(A, rho):
    """
    Proximal operator of the negative log-determinant barrier.
    """
    A = sym(A)

    d, Q = eigh(A)

    x = 0.5 * (
        d + np.sqrt(np.maximum(d * d + 4.0 / rho, 0.0))
    )

    return sym(Q @ np.diag(x) @ Q.T)


def min_eig(M):
    """
    Minimum eigenvalue of a symmetric matrix.
    """
    return float(np.min(np.linalg.eigvalsh(sym(M))))


def make_spd(M, eps=1e-6):
    """
    Force a matrix to be symmetric positive definite.
    """
    M = sym(M)

    me = min_eig(M)

    if me < eps:
        M = M + (eps - me + 1e-10) * np.eye(M.shape[0])

    return sym(M)


def pd_inverse(M, jitter=1e-6):
    """
    Stable inverse for SPD matrices.
    """
    d, Q = eigh(sym(M))

    d = np.maximum(d, jitter)

    return sym(Q @ np.diag(1.0 / d) @ Q.T)


def pd_logdet(M, jitter=1e-6):
    """
    Stable log-determinant for SPD matrices.
    """
    d, _ = eigh(sym(M))

    return float(np.sum(np.log(np.maximum(d, jitter))))


def offdiag_l1(M):
    """
    Off-diagonal L1 norm.
    """
    A = np.abs(np.asarray(M))

    return float(
        A.sum() - np.abs(np.diag(M)).sum()
    )


def nuclear_norm_psd(P):
    """
    Nuclear norm of a PSD matrix.
    """
    vals = np.linalg.eigvalsh(sym(P))

    return float(np.maximum(vals, 0.0).sum())