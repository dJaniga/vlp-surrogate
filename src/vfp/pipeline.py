import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from vfp.builder import prepare_table_body, VFPTable, VFPType
from vfp.export import export_VFP_table
from vfp.modeling import VFPModel
from vfp.preprocess import (
    get_training_data,
    to_prediction_format,
    reshape_predictions,
)
from vfp.preprocess.filters import filter_valid_operating_conditions


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


def vfp_pipeline(
    output_file_path: Path,
    well_data: pd.DataFrame,
    reference_model: VFPModel,
    vfp_table_id: int,
    bhp_depth: float,
    config: VFPPipelineConfig,
    vfp_table_granularity: int,
) -> VFPTable | None:
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
    )

    if table is None:
        return None

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
    bhp_depth,
):
    model = copy.deepcopy(reference_model)

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

    model.fit(training_data.features.to_numpy(), training_data.target.to_numpy(), features_name)

    prediction_content = to_prediction_format(
        training_data.features, keys=features_keys, n_size=vfp_table_granularity
    )

    predicted_targets = model.predict(prediction_content.features)
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
