import json
import logging
from collections.abc import Set
from pathlib import Path

from readers import get_reader_by_file_suffix, WellDataFilter
from vfp.details import VFPDetails
from vfp.export import export_VFP_manifest
from vfp.modeling import VFPModel, LinearRegressionModel
from vfp.pipeline import vfp_pipeline, VFPPROD_CONFIG, VFPINJ_CONFIG

logger = logging.getLogger(__name__)


def _get_model_by_name(model_name: str) -> VFPModel:
    if model_name == "linear":
        logger.info("Using linear regression model for training and prediction")
        return LinearRegressionModel()
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def _read_vfp_details(vfp_details_file_path: Path) -> VFPDetails:
    with open(vfp_details_file_path, "r") as f:
        data = json.load(f)
        result = VFPDetails.model_validate(data)
        logger.debug(
            "Loaded VFP details",
            extra={"Path": vfp_details_file_path, "Result": result},
        )
        return result


def _read_well_data_filter(well_data_filer_path: Path | None) -> WellDataFilter | None:
    if well_data_filer_path is None:
        return None
    with open(well_data_filer_path, "r") as f:
        data = json.load(f)
        result = WellDataFilter.model_validate(data)
        logger.debug(
            "Loaded well data filter",
            extra={"Path": well_data_filer_path, "Result": result},
        )
        return result


def run_pipeline(
    source_file_path: Path,
    vfp_details_file_path: Path,
    model_name: str,
    output_folder_path: Path,
    well_data_filer_path: Path | None = None,
    table_granularity: int = 5,
) -> None:
    reader = get_reader_by_file_suffix(source_file_path)
    well_data_filter = _read_well_data_filter(well_data_filer_path)
    wells_flow_data = reader.read_wells_flow_data(source_file_path, well_data_filter)
    wells_vfpi_details = _read_vfp_details(vfp_details_file_path)
    reference_model = _get_model_by_name(model_name)
    manifest_content: list[Path] = []

    wells_name_in_flow_data = set(wells_flow_data.keys())
    wells_name_in_vfpi_details = set(wells_vfpi_details.keys())

    _check_wells_name_compliance(wells_name_in_flow_data, wells_name_in_vfpi_details)

    for well_name, flow_data in wells_flow_data.items():
        logger.info("Processing well", extra={"Well": well_name})
        vfp_details = wells_vfpi_details[well_name]
        if vfp_details is None:
            logger.warning(f"No VFP details found for well {well_name}")
            continue

        if flow_data.production is not None and not flow_data.production.empty:
            logger.debug("Processing production data", extra={"Well": well_name})
            table_id = vfp_details.VFPPROD.table_number
            bhp_depth = vfp_details.VFPPROD.bhp_depth

            output_p_file_path = Path(output_folder_path, f"{well_name}_p.vfp")

            vfp_p_table = vfp_pipeline(
                output_file_path=output_p_file_path,
                well_data=flow_data.production,
                reference_model=reference_model,
                vfp_table_id=table_id,
                bhp_depth=bhp_depth,
                config=VFPPROD_CONFIG,
                vfp_table_granularity=table_granularity,
            )

            if vfp_p_table is not None:
                manifest_content.append(output_p_file_path)

        if flow_data.injection is not None and not flow_data.injection.empty:
            logger.debug("Processing injection data", extra={"Well": well_name})
            table_id = vfp_details.VFPINJ.table_number
            bhp_depth = vfp_details.VFPINJ.bhp_depth

            output_i_file_path = Path(output_folder_path, f"{well_name}_i.vfp")

            vfp_p_table = vfp_pipeline(
                output_file_path=output_i_file_path,
                well_data=flow_data.production,
                reference_model=reference_model,
                vfp_table_id=table_id,
                bhp_depth=bhp_depth,
                config=VFPINJ_CONFIG,
                vfp_table_granularity=table_granularity,
            )

            if vfp_p_table is not None:
                manifest_content.append(output_i_file_path)

    if manifest_content:
        logger.info("Exporting VFP manifest")
        export_VFP_manifest(
            manifest_content, Path(output_folder_path, "VFP_manifest.txt")
        )

    logger.info("Pipeline completed")


def _check_wells_name_compliance(
    wells_name_in_flow_data: Set[str], wells_name_in_vfpi_details: Set[str]
) -> None:
    missing_wells = wells_name_in_flow_data - wells_name_in_vfpi_details
    if missing_wells:
        raise ValueError(
            f"Wells {missing_wells} are not found in VFP details. Check naming convention."
        )
