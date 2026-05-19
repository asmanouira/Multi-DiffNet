from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from .data import MultiModalData
from .optimization import *
from .metrics import *
from .io import *

class MultiDiffNet:
    """
    Joint differential network estimator with latent variables.

    This class estimates:
        - a shared baseline precision matrix,
        - group-specific differential precision matrices,
        - a low-rank latent component for the baseline group.

    The model supports an arbitrary number of modalities through the
    ``MultiModalData`` object produced by the data-loading utilities.

    Parameters
    ----------
    lambda_1 : float, default=0.1
        Sparsity penalty applied to the baseline sparse component ``S_0``.

    lambda_2 : float, array-like, or Mapping, default=0.6
        Sparsity penalty applied to differential matrices. It can be:
            - a scalar shared by all exposed groups,
            - a vector with one value per exposed group,
            - a dictionary mapping group names to penalty values.

    lambda_3 : float, default=0.01
        Fusion penalty encouraging similarity between differential matrices.

    mu0 : float, default=2.5
        Nuclear norm penalty controlling the baseline latent component.

    rho : float, default=1.0
        ADMM penalty parameter.

    eps : float, default=1e-2
        Convergence tolerance.

    maxiter_baseline : int, default=1000
        Maximum number of baseline ADMM iterations.

    maxiter_admm : int, default=20
        Maximum number of ADMM iterations for exposed-group updates.

    maxiter_ama : int, default=20
        Maximum number of inner AMA iterations.

    maxiter_outer : int, default=20
        Maximum number of outer alternating optimization iterations.

    baseline_step_init : float, default=1e-3
        Initial step size for the linearized baseline update.

    joint_jitter : float, default=1e-6
        Numerical stabilization value.

    verbose : bool, default=True
        Whether to print optimization progress.

    Attributes
    ----------
    fit_ : dict or None
        Fitted model outputs. It is ``None`` before calling ``fit``.
    """

    def __init__(
        self,
        lambda_1=0.1,
        lambda_2=0.6,
        lambda_3=0.01,
        mu0=2.5,
        rho=1.0,
        eps=1e-2,
        maxiter_baseline=1000,
        maxiter_admm=20,
        maxiter_ama=20,
        maxiter_outer=20,
        baseline_step_init=1e-3,
        joint_jitter=1e-6,
        verbose=True,
    ):
        """
        Initialize the DiffNet latent-variable estimator.

        Parameters are stored as class attributes and used later by ``fit``.
        No model estimation is performed during initialization.
        """
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.mu0 = mu0
        self.rho = rho
        self.eps = eps
        self.maxiter_baseline = maxiter_baseline
        self.maxiter_admm = maxiter_admm
        self.maxiter_ama = maxiter_ama
        self.maxiter_outer = maxiter_outer
        self.baseline_step_init = baseline_step_init
        self.joint_jitter = joint_jitter
        self.verbose = verbose
        self.fit_ = None

    def _lambda2_vec(self, group_names):
        """
        Convert the user-provided lambda_2 parameter into a vector.

        Parameters
        ----------
        group_names : array-like
            Names of exposed groups.

        Returns
        -------
        ndarray of shape (K,)
            Group-specific sparsity penalties.

        Raises
        ------
        ValueError
            If a non-scalar lambda_2 vector does not have one value per group.
        """
        if isinstance(self.lambda_2, Mapping):
            return np.array([
                float(self.lambda_2.get(str(g), np.mean(list(self.lambda_2.values()))))
                for g in group_names
            ])

        arr = np.asarray(self.lambda_2, dtype=float)

        if arr.ndim == 0:
            return np.repeat(float(arr), len(group_names))

        if arr.size != len(group_names):
            raise ValueError("lambda_2 vector must have one value per exposed group")

        return arr.reshape(-1)

    def fit(self, data: MultiModalData):
        """
        Fit the joint latent differential network model.

        Parameters
        ----------
        data : MultiModalData
            Preprocessed multi-modal data object containing baseline and exposed
            samples, labels, feature dimensions, and group names.

        Returns
        -------
        dict
            Fitted model output containing:
                - ``estimate_0``: baseline quantities,
                - ``estimate_K``: exposed-group quantities,
                - ``lambda_2_vec``: resolved group-specific penalties.
        """
        data_0, data_K, labels, K = data.data_0, data.data_K, data.labels, data.K
        lambda_2_vec = self._lambda2_vec(data.group_names)

        Sigma0 = sym(np.cov(data_0, rowvar=True))
        n0 = data_0.shape[1]

        Mu, Sam_cov, prob, n_per_group = initialize_group_statistics(data_K, labels, K)

        base0 = baseline_admm_latent_only(
            Sigma0, self.lambda_1, self.mu0, self.rho, self.maxiter_baseline
        )

        S0, P0, Theta0 = base0["S_0"], base0["P_0"], base0["Theta_0"]

        outK = update_thetaK_supervised_fixed(
            Sam_cov,
            Theta0,
            lambda_2_vec,
            self.lambda_3,
            self.rho,
            self.maxiter_admm,
            self.maxiter_ama,
            self.eps,
        )

        ThetaK, DeltaK, PhiK = outK["Theta"], outK["Delta"], outK["Phi"]

        history = [
            joint_objective(
                Sigma0,
                Sam_cov,
                n0,
                n_per_group,
                S0,
                P0,
                DeltaK,
                self.lambda_1,
                lambda_2_vec,
                self.lambda_3,
                self.mu0,
                self.joint_jitter,
            )
        ]

        if self.verbose:
            print(f"[JOINT] iter=0 obj={history[-1]:.6e}")

        for t in range(1, self.maxiter_outer + 1):
            Theta0_prev = Theta0.copy()
            Delta_prev = DeltaK.copy()
            obj_prev = history[-1]

            base = baseline_joint_update_linearized(
                Sigma0,
                Sam_cov,
                n0,
                n_per_group,
                S0,
                P0,
                DeltaK,
                self.lambda_1,
                self.mu0,
                self.baseline_step_init,
                self.joint_jitter,
            )

            S0, P0, Theta0 = base["S_0"], base["P_0"], base["Theta_0"]

            outK = update_thetaK_supervised_fixed(
                Sam_cov,
                Theta0,
                lambda_2_vec,
                self.lambda_3,
                self.rho,
                self.maxiter_admm,
                self.maxiter_ama,
                self.eps,
                {"Theta": ThetaK, "Delta": DeltaK, "Phi": PhiK},
            )

            ThetaK, DeltaK, PhiK = outK["Theta"], outK["Delta"], outK["Phi"]

            obj = joint_objective(
                Sigma0,
                Sam_cov,
                n0,
                n_per_group,
                S0,
                P0,
                DeltaK,
                self.lambda_1,
                lambda_2_vec,
                self.lambda_3,
                self.mu0,
                self.joint_jitter,
            )

            history.append(obj)

            rel_obj = abs(obj - obj_prev) / (abs(obj_prev) + 1e-12)
            rel_theta0 = norm(Theta0 - Theta0_prev, "fro") / (
                norm(Theta0_prev, "fro") + 1e-12
            )
            rel_delta = fro_norm_tensor(DeltaK - Delta_prev) / (
                fro_norm_tensor(Delta_prev) + 1e-12
            )

            if self.verbose:
                print(
                    f"[JOINT] iter={t} obj={obj:.6e} "
                    f"rel_obj={rel_obj:.3e} "
                    f"rel_theta0={rel_theta0:.3e} "
                    f"rel_delta={rel_delta:.3e}"
                )

            if rel_obj < self.eps and rel_theta0 < self.eps and rel_delta < self.eps:
                break

        self.fit_ = {
            "estimate_0": {
                "Theta_0": sym(Theta0),
                "S_0": sym(S0),
                "L_0": -sym(P0),
                "P_0": sym(P0),
                "Sigma0": Sigma0,
                "n0": n0,
                "joint_objective_history": history,
            },
            "estimate_K": {
                "K": K,
                "Mu": Mu,
                "Theta": symmetrize(ThetaK),
                "Delta": symmetrize(DeltaK),
                "prob": prob,
                "member": labels,
                "Sam_cov": Sam_cov,
                "n_per_group": n_per_group,
                "Phi": symmetrize(PhiK),
            },
            "lambda_2_vec": lambda_2_vec,
        }

        return self.fit_

    def save_results(self, out_dir: str | Path, data: MultiModalData, save_baseline=True):
        """
        Save fitted model outputs to CSV files.

        This method saves baseline blocks, group-specific precision blocks,
        differential blocks, optimization history, eigenvalues of the latent
        component, and a run summary.

        Parameters
        ----------
        out_dir : str or Path
            Output directory where result files will be written.

        data : MultiModalData
            Data object used to fit the model. It provides modality dimensions,
            feature names, and group names.

        save_baseline : bool, default=True
            Whether to save baseline ``S_0`` modality blocks.

        Raises
        ------
        RuntimeError
            If the model has not been fitted before calling this method.
        """
        if self.fit_ is None:
            raise RuntimeError("Call fit(data) before save_results")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = block_slices_from_dims(data.dims)

        def save_matrix(M, path, rows=None, cols=None):
            """
            Save a matrix as a CSV file with optional row and column names.
            """
            pd.DataFrame(M, index=rows, columns=cols).to_csv(path)

        def save_blocks(M, folder, prefix, group_name):
            """
            Save diagonal and cross-modality blocks from a full matrix.
            """
            folder = Path(folder)
            folder.mkdir(parents=True, exist_ok=True)
            safe = str(group_name).replace("/", "_").replace(" ", "_")
            names = list(data.dims.keys())

            # Save modality-specific diagonal blocks.
            for a in names:
                I = idx[a]
                save_matrix(
                    M[np.ix_(I, I)],
                    folder / f"{prefix}_{safe}_{a}.csv",
                    data.feature_names[a],
                    data.feature_names[a],
                )

            # Save pairwise cross-modality blocks.
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    I, J = idx[a], idx[b]
                    save_matrix(
                        M[np.ix_(I, J)],
                        folder / f"{prefix}_{safe}_{a}__{b}.csv",
                        data.feature_names[a],
                        data.feature_names[b],
                    )

        est0, estK = self.fit_["estimate_0"], self.fit_["estimate_K"]

        if save_baseline:
            save_blocks(est0["S_0"], out_dir / "baseline_S0_blocks", "S0", "baseline")

        for k, g in enumerate(data.group_names):
            gdir = out_dir / f"group_{k:02d}_{str(g).replace('/', '_').replace(' ', '_')}"

            save_blocks(estK["Theta"][:, :, k], gdir, "Theta", g)
            save_blocks(estK["Delta"][:, :, k], gdir, "Delta", g)

            pd.DataFrame([
                {
                    "group_index": k,
                    "group_name": str(g),
                    "n_samples_group": int(estK["n_per_group"][k]),
                    "prob_group": float(estK["prob"][k]),
                }
            ]).to_csv(gdir / "group_summary.csv", index=False)

        pd.DataFrame({"joint_objective": est0["joint_objective_history"]}).to_csv(
            out_dir / "joint_objective_history.csv", index=False
        )

        vals = np.linalg.eigvalsh(sym(est0["P_0"]))
        pd.DataFrame({"eigval_P0": np.sort(vals)[::-1]}).to_csv(
            out_dir / "P0_eigenvalues.csv", index=False
        )

        eb0, E0, ll0 = ebic_single(est0["Sigma0"], est0["Theta_0"], est0["n0"])

        pd.DataFrame([
            {
                "baseline_group": data.baseline_group,
                "K": data.K,
                "groups": " | ".join(map(str, data.group_names)),
                "dims": str(data.dims),
                "rank_P0": int(np.sum(vals > 1e-6)),
                "ebic_baseline": eb0,
                "edges_baseline": E0,
                "ll_baseline": ll0,
                "joint_objective_last": float(est0["joint_objective_history"][-1]),
            }
        ]).to_csv(out_dir / "run_summary.csv", index=False)
