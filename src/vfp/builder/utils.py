import numpy as np
import pandas as pd


def _prepare_record_indexes(records_config: pd.DataFrame) -> np.ndarray:

    unique_counts_without_flo = records_config.drop(columns=["FLO"]).nunique().to_list()

    grids = np.meshgrid(
        *[np.arange(1, n + 1) for n in unique_counts_without_flo], indexing="xy"
    )

    return np.stack(grids, axis=-1, dtype=int).reshape(
        -1, len(unique_counts_without_flo)
    )


def prepare_table_body(
    records_config: pd.DataFrame, reshaped_targets: np.ndarray
) -> np.ndarray:
    vfp_records_indexes = _prepare_record_indexes(records_config)
    n_idx = vfp_records_indexes.shape[1]
    n_tgt = reshaped_targets.shape[1]

    dtype = [(f"idx_{i}", np.int64) for i in range(n_idx)] + [
        (f"target_{i}", np.float64) for i in range(n_tgt)
    ]

    result = np.empty(vfp_records_indexes.shape[0], dtype=dtype)

    for i in range(n_idx):
        result[f"idx_{i}"] = vfp_records_indexes[:, i]

    for i in range(n_tgt):
        result[f"target_{i}"] = reshaped_targets[:, i]

    # return np.hstack([vfp_records_indexes, reshaped_targets], dtype=dtype)
    return result
