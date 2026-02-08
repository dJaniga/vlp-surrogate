from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def clean_dataset(
    dataset: np.ndarray,
    minimum_acceptable_flow: float,
    start_time: float,
    mapping: dict[str, int],
) -> np.ndarray:
    if dataset.size == 0:
        logger.warning("Empty dataset provided", extra={"rows": 0})
        return dataset

    flow_index = mapping["flow"]
    time_index = mapping["time"]
    thp_index = mapping["thp"]

    mask = (
        (dataset[:, flow_index] > minimum_acceptable_flow)
        & (dataset[:, time_index] >= start_time)
        & (dataset[:, thp_index] != 0)
    )
    filtered = dataset[mask]

    wgr_index = mapping.get("wgr")
    if wgr_index is not None:
        wgr_mask = filtered[:, wgr_index] != 0
        filtered = filtered[wgr_mask]

    logger.info(
        "Dataset filtered",
        extra={"rows": int(filtered.shape[0]), "columns": int(filtered.shape[1])},
    )
    return filtered
