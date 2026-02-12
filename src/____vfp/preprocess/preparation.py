from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def prepare_data_for_model(
    dataset: np.ndarray,
    mapping: dict[str, int],
) -> np.ndarray:
    if dataset.size == 0:
        logger.warning("Empty dataset provided", extra={"rows": 0})
        return dataset

    flow_index = mapping["flow"]
    thp_index = mapping["thp"]
    bhp_index = mapping["bhp"]

    if "wgr" in mapping:
        wgr_index = mapping["wgr"]
        ogr_column = np.zeros((dataset.shape[0], 1))
        inputs = np.column_stack(
            [
                dataset[:, flow_index],
                dataset[:, thp_index],
                dataset[:, wgr_index],
                ogr_column,
            ]
        )
    else:
        inputs = np.column_stack([dataset[:, flow_index], dataset[:, thp_index]])

    targets = dataset[:, bhp_index]
    model_data = np.column_stack([inputs, targets])
    logger.info(
        "Prepared model data",
        extra={"rows": int(model_data.shape[0]), "columns": int(model_data.shape[1])},
    )
    return model_data


def prepare_data_for_prediction(
    dataset: np.ndarray, n_size: int, mapping: dict[str, int]
) -> np.ndarray:
    flow_values = np.linspace(
        dataset[:, mapping["flow"]].min(),
        dataset[:, mapping["flow"]].max(),
        n_size,
    )
    thp_values = np.linspace(
        dataset[:, mapping["thp"]].min(),
        dataset[:, mapping["thp"]].max(),
        n_size,
    )

    if "wgr" in mapping:
        wgr_values = np.linspace(
            dataset[:, mapping["wgr"]].min(),
            dataset[:, mapping["wgr"]].max(),
            n_size,
        )
        ogr_values = np.array([0.0])
        prediction_rows: list[list[float]] = []
        for ogr_value in ogr_values:
            for wgr_value in wgr_values:
                for thp_value in thp_values:
                    for flow_value in flow_values:
                        prediction_rows.append(
                            [flow_value, thp_value, wgr_value, ogr_value]
                        )
        prediction_data = np.array(prediction_rows, dtype=float)
        logger.info(
            "Prepared prediction grid",
            extra={
                "rows": int(prediction_data.shape[0]),
                "columns": int(prediction_data.shape[1]),
            },
        )
        return prediction_data

    prediction_rows = [
        [flow_value, thp_value]
        for thp_value in thp_values
        for flow_value in flow_values
    ]
    prediction_data = np.array(prediction_rows, dtype=float)
    logger.info(
        "Prepared prediction grid",
        extra={
            "rows": int(prediction_data.shape[0]),
            "columns": int(prediction_data.shape[1]),
        },
    )
    return prediction_data
