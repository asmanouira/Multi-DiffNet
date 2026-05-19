from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd


@dataclass
class MultiModalData:
    """
    Container storing processed multi-modal supervised data.

    Attributes
    ----------
    data_0 : ndarray of shape (p, n_baseline)
        Concatenated feature matrix corresponding to the baseline/control group.

    data_K : ndarray of shape (p, n_exposed)
        Concatenated feature matrix corresponding to all case/exposed
        or non-baseline groups.

    labels : ndarray of shape (n_exposed,)
        Integer-encoded labels for exposed samples.

    K : int
        Number of exposed groups.

    dims : dict[str, int]
        Dictionary mapping each modality name to its number of features.

    group_names : ndarray
        Array containing the exposed group names.

    feature_names : dict[str, list[str]]
        Dictionary mapping modality names to feature names.

    sample_names_baseline : list[str]
        Sample identifiers belonging to the baseline group.

    sample_names_exposed : list[str]
        Sample identifiers belonging to exposed groups.

    baseline_group : str
        Name of the baseline/reference group.
    """
    data_0: np.ndarray
    data_K: np.ndarray
    labels: np.ndarray
    K: int
    dims: dict[str, int]
    group_names: np.ndarray
    feature_names: dict[str, list[str]]
    sample_names_baseline: list[str]
    sample_names_exposed: list[str]
    baseline_group: str


def global_center_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply global centering to a dataframe.

    Each feature column is centered by subtracting its global mean.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with samples in rows and features in columns.

    Returns
    -------
    pandas.DataFrame
        Globally centered dataframe.
    """
    return df - df.mean(axis=0)


def group_center_df(df: pd.DataFrame, group_labels) -> pd.DataFrame:
    """
    Apply group-wise centering to a dataframe.

    For each group, feature means are computed independently and
    subtracted from samples belonging to that group.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with samples in rows and features in columns.

    group_labels : array-like
        Group label associated with each sample.

    Returns
    -------
    pandas.DataFrame
        Group-centered dataframe.
    """
    out = df.copy()
    groups = pd.Series(group_labels, index=df.index).astype(str)
    for g in groups.unique():
        mask = groups == g
        out.loc[mask] = out.loc[mask] - out.loc[mask].mean(axis=0)
    return out


def scale_df(df: pd.DataFrame, min_scale: float = 1e-6):
    """
    Scale dataframe features using their global standard deviation.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    min_scale : float, default=1e-6
        Minimum allowed standard deviation to avoid division by zero.

    Returns
    -------
    tuple
        scaled_df : pandas.DataFrame
            Scaled dataframe.

        sd : pandas.Series
            Feature-wise standard deviations.
    """
    sd = df.std(axis=0, ddof=1).clip(lower=min_scale)
    return df / sd, sd


def scale_df_by_baseline(
    df: pd.DataFrame,
    group_labels,
    baseline_group: str,
    min_scale: float = 1e-6,
):
    """
    Scale features using statistics computed only on the baseline group.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    group_labels : array-like
        Group labels for all samples.

    baseline_group : str
        Name of the baseline/reference group.

    min_scale : float, default=1e-6
        Minimum allowed standard deviation.

    Returns
    -------
    tuple
        scaled_df : pandas.DataFrame
            Scaled dataframe.

        sd : pandas.Series
            Baseline-derived standard deviations.

    Raises
    ------
    ValueError
        If no baseline samples are found.
    """
    groups = pd.Series(group_labels, index=df.index).astype(str)
    mask = groups == str(baseline_group)
    if mask.sum() == 0:
        raise ValueError(f"No baseline samples found for baseline_group={baseline_group!r}")
    sd = df.loc[mask].std(axis=0, ddof=1).clip(lower=min_scale)
    return df / sd, sd


def load_multimodal_supervised(
    modalities: Mapping[str, str | Path | pd.DataFrame],
    meta_path: str | Path | pd.DataFrame,
    *,
    group_col: str = "Group",
    baseline_group: str = "sPTC",
    feature_limits: Optional[Mapping[str, int]] = None,
    do_global_center: bool = False,
    do_group_center: bool = True,
    do_scale: bool = False,
    scale_mode: str = "global",
    min_scale: float = 1e-6,
) -> MultiModalData:
    """
    Load and preprocess supervised multi-modal data.

    This function supports an arbitrary number of modalities.
    All modalities are automatically aligned using common sample identifiers.

    Each modality must contain:
        - samples in rows,
        - features in columns.

    Parameters
    ----------
    modalities : Mapping[str, str | Path | pandas.DataFrame]
        Dictionary mapping modality names to CSV file paths
        or already loaded pandas DataFrames.

        Example
        -------
        {
            "miRNA_normal": "XN.csv",
            "miRNA_tumor": "XT.csv",
            "gene_normal": "YN.csv"
        }

    meta_path : str | Path | pandas.DataFrame
        Metadata table containing group labels.

    group_col : str, default="Group"
        Column name containing phenotype or group labels.

    baseline_group : str, default="sPTC"
        Name of the baseline/reference group.

    feature_limits : dict[str, int], optional
        Optional dictionary specifying the maximum number
        of features to retain for each modality.

    do_global_center : bool, default=False
        Whether to globally center features.

    do_group_center : bool, default=True
        Whether to center features independently within each group.

    do_scale : bool, default=False
        Whether to scale features.

    scale_mode : {"global", "baseline"}, default="global"
        Scaling strategy:
            - "global": use all samples,
            - "baseline": use only baseline samples.

    min_scale : float, default=1e-6
        Minimum allowed scaling value.

    Returns
    -------
    MultiModalData
        Structured object containing processed data matrices,
        labels, dimensions, and metadata.

    Raises
    ------
    ValueError
        If:
            - no modalities are provided,
            - no common samples are found,
            - no baseline samples are found,
            - no exposed samples are found.
    """

    if not modalities:
        raise ValueError("modalities must contain at least one modality")

    def _read(obj):
        """
        Internal helper function used to load CSV files
        or copy existing DataFrames.
        """
        if isinstance(obj, pd.DataFrame):
            return obj.copy()
        return pd.read_csv(obj, index_col=0)

    meta = _read(meta_path)
    if group_col not in meta.columns:
        if meta.shape[1] == 1:
            meta = meta.rename(columns={meta.columns[0]: group_col})
        else:
            raise ValueError(f"group_col={group_col!r} not found in metadata")
    meta[group_col] = meta[group_col].astype(str)

    feature_limits = dict(feature_limits or {})
    dfs: dict[str, pd.DataFrame] = {}
    patients = meta.index
    # ------------------------------------------------------------------
    # Load all modalities and identify common samples
    # ------------------------------------------------------------------
    for name, obj in modalities.items():
        df = _read(obj)
        if name in feature_limits and feature_limits[name] is not None:
            df = df.iloc[:, : int(feature_limits[name])]
        dfs[name] = df.copy()
        patients = patients.intersection(df.index)

    if len(patients) == 0:
        raise ValueError("No common samples between modalities and metadata")

    meta = meta.loc[patients].copy()
    feature_names: dict[str, list[str]] = {}
    arrays = []
    dims: dict[str, int] = {}
    # ------------------------------------------------------------------
    # Preprocess each modality independently
    # ------------------------------------------------------------------
    for name, df in dfs.items():
        df = df.loc[patients].copy()
        feature_names[name] = df.columns.astype(str).tolist()

        if do_global_center:
            df = global_center_df(df)
        if do_group_center:
            df = group_center_df(df, meta[group_col])
        if do_scale:
            if scale_mode == "global":
                df, _ = scale_df(df, min_scale=min_scale)
            elif scale_mode == "baseline":
                df, _ = scale_df_by_baseline(df, meta[group_col], baseline_group, min_scale)
            else:
                raise ValueError("scale_mode must be 'global' or 'baseline'")

        X = df.T.to_numpy(dtype=float)
        arrays.append(X)
        dims[name] = int(X.shape[0])

    # ------------------------------------------------------------------
    # Concatenate all modalities
    # ------------------------------------------------------------------
    X_concat = np.vstack(arrays)
    labels_str = meta[group_col].astype(str).to_numpy()

    idx_base = np.where(labels_str == str(baseline_group))[0]
    idx_exp = np.where(labels_str != str(baseline_group))[0]
    if idx_base.size == 0:
        raise ValueError(f"No baseline samples found for baseline_group={baseline_group!r}")
    if idx_exp.size == 0:
        raise ValueError("No exposed/non-baseline samples found")

    groups_exp = labels_str[idx_exp]
    group_names = np.unique(groups_exp)
    group_to_int = {g: i for i, g in enumerate(group_names)}
    labels = np.array([group_to_int[g] for g in groups_exp], dtype=int)
    # ------------------------------------------------------------------
    # Build final structured object
    # ------------------------------------------------------------------
    return MultiModalData(
        data_0=X_concat[:, idx_base],
        data_K=X_concat[:, idx_exp],
        labels=labels,
        K=int(len(group_names)),
        dims=dims,
        group_names=group_names,
        feature_names=feature_names,
        sample_names_baseline=meta.index[idx_base].astype(str).tolist(),
        sample_names_exposed=meta.index[idx_exp].astype(str).tolist(),
        baseline_group=str(baseline_group),
    )


def block_slices_from_dims(dims: Mapping[str, int]) -> dict[str, np.ndarray]:
    """
    Build feature index ranges for each modality.

    Parameters
    ----------
    dims : dict[str, int]
        Dictionary mapping modality names to feature counts.

        Example
        -------
        {
            "miRNA": 1500,
            "mRNA": 2000
        }

    Returns
    -------
    dict[str, ndarray]
        Dictionary mapping modality names to feature indices
        within the concatenated matrix.

    Example
    -------
    {
        "miRNA": array([0, ..., 1499]),
        "mRNA": array([1500, ..., 3499])
    }
    """
    start = 0
    idx = {}
    for name, p in dims.items():
        idx[name] = np.arange(start, start + int(p), dtype=int)
        start += int(p)
    return idx
