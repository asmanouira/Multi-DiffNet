"""
Input/output utilities for Multi-DiffNet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .operators import sym
from .metrics import ebic_single
from .data import block_slices_from_dims


def save_matrix(
    M,
    path,
    rows=None,
    cols=None,
):
    """
    Save a matrix to CSV.
    """
    pd.DataFrame(
        M,
        index=rows,
        columns=cols,
    ).to_csv(path)


def save_blocks(
    M,
    folder,
    prefix,
    group_name,
    data,
    idx,
):
    """
    Save diagonal and cross-modality blocks.
    """
    folder = Path(folder)

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe = (
        str(group_name)
        .replace("/", "_")
        .replace(" ", "_")
    )

    names = list(data.dims.keys())

    # diagonal blocks
    for a in names:

        I = idx[a]

        save_matrix(
            M[np.ix_(I, I)],
            folder / f"{prefix}_{safe}_{a}.csv",
            data.feature_names[a],
            data.feature_names[a],
        )

    # cross blocks
    for i, a in enumerate(names):

        for b in names[i + 1:]:

            I = idx[a]
            J = idx[b]

            save_matrix(
                M[np.ix_(I, J)],
                folder / f"{prefix}_{safe}_{a}__{b}.csv",
                data.feature_names[a],
                data.feature_names[b],
            )