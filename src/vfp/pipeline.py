import copy
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from vfp.builder import prepare_table_body, VFPTable, VFPType
from vfp.export import export_VFP_table
from vfp.modeling import VFPModel, ModelWrapper
from vfp.preprocess import (
    get_training_data,
    to_prediction_format,
    reshape_predictions,
)
from vfp.preprocess.filters import filter_valid_operating_conditions

logger = logging.getLogger(__name__)

TOL = 1e-6

@dataclass(frozen=True)
class VFPPipelineConfig:
    table_type: VFPType
    header_template: str
    required_keys: Sequence[str]
    feature_keys: Sequence[str]
    target_keys: Sequence[str]


VFPPROD_CONFIG = VFPPipelineConfig(
    table_type=VFPType.PRODUCTION,
    header_template="{table_id} {bhp_depth} 'GAS' 'WGR' 'OGR' 'THP' '' 'METRIC' 'BHP'/",
    required_keys=["FLO", "THP", "BHP"],
    feature_keys=["FLO", "THP", "WFR", "GFR"],
    target_keys=["BHP"],
)

VFPINJ_CONFIG = VFPPipelineConfig(
    table_type=VFPType.INJECTION,
    header_template="{table_id} {bhp_depth} 'GAS' 'THP' 'METRIC' 'BHP' /",
    required_keys=["FLO", "THP", "BHP"],
    feature_keys=["FLO", "THP"],
    target_keys=["BHP"],
)


def is_valid(table_type: VFPType, table: VFPTable) -> bool:
    if table_type == VFPType.INJECTION:
        for record in table.body:
            bhp = record.tolist()[1:]

            are_constants = all(
                abs(bhp[i] - bhp[i - 1]) < TOL
                for i in range(1, len(bhp))
            )
            if are_constants:
                logger.error(f"BHP VALUES MUST NOT BE CONSTANTS!: {bhp}")
                return False

            is_non_increasing = all(
                bhp[i] <= bhp[i - 1] + TOL
                for i in range(1, len(bhp))
            )
            if not is_non_increasing:
                logger.error(f"BHP VALUES MUST NOT INCREASE WITH FLOW RATE!: {bhp}")
                return False

        else:
            return True
    else:
        return True


def vfp_pipeline(
    output_file_path: Path,
    well_data: pd.DataFrame,
    reference_model: VFPModel,
    vfp_table_id: int,
    bhp_depth: float,
    config: VFPPipelineConfig,
    vfp_table_granularity: int,
    optimize_hyperparameters: bool = True,
    tuning_metric: str = "mean_squared_error",
    seed: int | None = None,
    max_seed_retries: int = 5,
) -> VFPTable | None:

    fit_results_export_path = output_file_path.with_suffix("")

    start_time = time.perf_counter()

    seeds_to_try: list[int | None] = [seed]
    rng = np.random.default_rng(seed)
    seeds_to_try += rng.integers(0, 10_000, size=max_seed_retries).tolist()

    table = None
    for attempt, current_seed in enumerate(seeds_to_try):
        if attempt > 0:
            logger.warning(
                "Retrying pipeline with seed=%s (attempt %d/%d)",
                current_seed,
                attempt,
                max_seed_retries,
            )
        table = _pipeline(
            reference_model=reference_model,
            well_data=well_data,
            required_valid_operation_condition_keys=config.required_keys,
            features_keys=config.feature_keys,
            targets_keys=config.target_keys,
            vfp_table_granularity=vfp_table_granularity,
            vfp_table_header_template=config.header_template,
            vfp_table_id=vfp_table_id,
            bhp_depth=bhp_depth,
            fit_results_export_path=fit_results_export_path,
            optimize_hyperparameters=optimize_hyperparameters,
            tuning_metric=tuning_metric,
            seed=current_seed,
        )

        if table is None:
            return None

        if is_valid(config.table_type, table):
            break

        if attempt == max_seed_retries:
            raise ValueError("BHP VALUES VALIDATION FAILED AFTER MAX RETRIES!")

        logger.error("BHP validation failed, will retry with a different seed.")

    elapsed = time.perf_counter() - start_time
    logger.info("VFP pipeline finished in %.3f s", elapsed)

    export_VFP_table(table, output_file_path, config.table_type)
    return table


def _pipeline(
    reference_model: VFPModel,
    well_data: pd.DataFrame,
    required_valid_operation_condition_keys: Sequence[str],
    features_keys: Sequence[str],
    targets_keys: Sequence[str],
    vfp_table_granularity: int,
    vfp_table_header_template: str,
    vfp_table_id: int,
    bhp_depth: float,
    fit_results_export_path: Path,
    optimize_hyperparameters: bool = True,
    tuning_metric: str = "mean_squared_error",
    seed: int | None = None,
):
    model = copy.deepcopy(reference_model)

    model_wrapper = ModelWrapper(
        model=model, export_path=fit_results_export_path, seed=seed
    )

    valid_operating_conditions = filter_valid_operating_conditions(
        well_data, required_valid_operation_condition_keys
    )

    if valid_operating_conditions.empty:
        return None

    training_data = get_training_data(
        valid_operating_conditions,
        features_keys=features_keys,
        targets_keys=targets_keys,
    )
    features_name = tuple(training_data.features.keys())

    model_wrapper.fit(
        training_data.T.to_numpy(),
        training_data.features.to_numpy(),
        training_data.target.to_numpy(),
        features_name,
        optimize_hyperparameters=optimize_hyperparameters,
        tuning_metric=tuning_metric,
    )

    prediction_content = to_prediction_format(
        training_data.features, keys=features_keys, n_size=vfp_table_granularity
    )

    predicted_targets = model_wrapper.predict(prediction_content.features)
    reshaped_targets = reshape_predictions(
        predicted_targets, n_size=vfp_table_granularity
    )
    table_body = prepare_table_body(prediction_content.records_config, reshaped_targets)

    header = vfp_table_header_template.format(
        table_id=vfp_table_id, bhp_depth=bhp_depth
    )

    vfp_table = VFPTable(
        header=header, config_records=prediction_content.records_config, body=table_body
    )

    return vfp_table
