import json
import logging
import shutil
from collections.abc import Set
from pathlib import Path
from typing import Literal


from readers import initialize_reader_from_path, WellDataFilter
from vfp.details import VFPDetails
from vfp.export import export_VFP_manifest
from vfp.modeling import VFPModel
from vfp.modeling.gaussian_process import GaussianProcessRegressor
from vfp.modeling.sklearn_regressors import (
    LinearRegressor,
    ElasticNetRegressor,
    XGBoostRegressor,
    BayesianRidgeRegressor,
    HuberRegressor,
)
from vfp.modeling.symbolic import SymbolicRegressor

from vfp.pipeline import vfp_pipeline, VFPPROD_CONFIG, VFPINJ_CONFIG

logger = logging.getLogger(__name__)

ModelName = Literal[
    "linear", "symbolic", "xgb", "gp", "elasticnet", "bayesian_ridge", "huber_regressor"
]


# -----------------------------------------------------------------------------
# Surrogate model factory
# -----------------------------------------------------------------------------


def create_model(name: ModelName, **kwargs) -> VFPModel:
    logger.info("Creating model", extra={"model": name})
    if name == "linear":
        return LinearRegressor(**kwargs)
    if name == "elasticnet":
        return ElasticNetRegressor(**kwargs)
    if name == "symbolic":
        return SymbolicRegressor(**kwargs)
    if name == "xgb":
        return XGBoostRegressor(**kwargs)
    if name == "gp":
        return GaussianProcessRegressor(**kwargs)
    if name == "bayesian_ridge":
        return BayesianRidgeRegressor(**kwargs)
    if name == "huber_regressor":
        return HuberRegressor(**kwargs)
    raise ValueError(f"Unsupported model name: {name}")


# -----------------------------------------------------------------------------
# JSON loaders
# -----------------------------------------------------------------------------


def _read_vfp_details(vfp_details_file_path: Path) -> VFPDetails:
    if not vfp_details_file_path.exists():
        raise FileNotFoundError(f"VFP details file not found: {vfp_details_file_path}")

    with vfp_details_file_path.open("r") as f:
        data = json.load(f)

    try:
        result = VFPDetails.model_validate(data)
    except Exception as exc:
        raise ValueError(
            f"Invalid VFP details format in {vfp_details_file_path}"
        ) from exc

    logger.debug(
        "Loaded VFP details",
        extra={"Path": str(vfp_details_file_path)},
    )
    return result


def _read_well_data_filter(
    well_data_filter_path: Path | None,
) -> WellDataFilter | None:
    if well_data_filter_path is None:
        return None

    if not well_data_filter_path.exists():
        raise FileNotFoundError(
            f"Well data filter file not found: {well_data_filter_path}"
        )

    with well_data_filter_path.open("r") as f:
        data = json.load(f)

    try:
        result = WellDataFilter.model_validate(data)
    except Exception as exc:
        raise ValueError(
            f"Invalid well data filter format in {well_data_filter_path}"
        ) from exc

    logger.debug(
        "Loaded well data filter",
        extra={"Path": str(well_data_filter_path)},
    )
    return result


# -----------------------------------------------------------------------------
# Core processing
# -----------------------------------------------------------------------------


def _check_well_name_compliance(
    wells_in_flow_data: Set[str],
    wells_in_vfp_details: Set[str],
) -> None:
    missing = wells_in_flow_data - wells_in_vfp_details
    if missing:
        raise ValueError(
            f"Wells {missing} are not found in VFP details. Check naming convention."
        )

    extra = wells_in_vfp_details - wells_in_flow_data
    if extra:
        logger.warning(
            "VFP details contain wells not present in flow data",
            extra={"ExtraWells": sorted(extra)},
        )


def _process_well_stream(
    *,
    well_name: str,
    stream_name: str,
    stream_data,
    vfp_section,
    config,
    file_suffix: str,
    output_folder_path: Path,
    reference_model: VFPModel,
    table_granularity: int,
    optimize_hyperparameters: bool,
    tuning_metric: str,
) -> Path | None:
    if stream_data is None or stream_data.empty:
        return None

    if vfp_section is None:
        logger.warning(
            "Missing VFP section for stream",
            extra={"Well": well_name, "Stream": stream_name},
        )
        return None

    table_id = vfp_section.table_number
    bhp_depth = vfp_section.bhp_depth

    output_file_path = output_folder_path / f"{well_name}_{file_suffix}.vfp"

    logger.debug(
        "Running VFP pipeline",
        extra={
            "Well": well_name,
            "Stream": stream_name,
            "TableID": table_id,
        },
    )

    vfp_table = vfp_pipeline(
        output_file_path=output_file_path,
        well_data=stream_data,
        reference_model=reference_model,
        vfp_table_id=table_id,
        bhp_depth=bhp_depth,
        config=config,
        vfp_table_granularity=table_granularity,
        optimize_hyperparameters=optimize_hyperparameters,
        tuning_metric=tuning_metric,
    )

    if vfp_table is None:
        logger.warning(
            "VFP pipeline returned no table",
            extra={"Well": well_name, "Stream": stream_name},
        )
        return None

    return output_file_path


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def run_pipeline(
    source_file_path: Path,
    vfp_details_file_path: Path,
    surrogate_model: VFPModel,
    output_folder_path: Path,
    well_data_filter_path: Path | None = None,
    table_granularity: int = 5,
    optimize_hyperparameters: bool = True,
    tuning_metric: str = "mean_squared_error",
) -> list[Path]:
    if table_granularity <= 2:
        raise ValueError("table_granularity must be greater than 2")

    if output_folder_path.exists():
        shutil.rmtree(output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    reader = initialize_reader_from_path(source_file_path)

    well_data_filter = _read_well_data_filter(well_data_filter_path)
    wells_flow_data = reader.read_wells_flow_data(well_data_filter)

    wells_vfp_details = _read_vfp_details(vfp_details_file_path)
    reference_model = surrogate_model

    manifest_content: list[Path] = []

    wells_in_flow_data = set(wells_flow_data.keys())
    wells_in_vfp_details = set(wells_vfp_details.keys())

    _check_well_name_compliance(
        wells_in_flow_data,
        wells_in_vfp_details,
    )

    for well_name, flow_data in wells_flow_data.items():
        logger.info("Processing well", extra={"Well": well_name})

        vfp_details = wells_vfp_details[well_name]
        if vfp_details is None:
            logger.warning(
                "No VFP details found for well",
                extra={"Well": well_name},
            )
            continue

        # Production
        prod_path = _process_well_stream(
            well_name=well_name,
            stream_name="production",
            stream_data=flow_data.production,
            vfp_section=getattr(vfp_details, "VFPPROD", None),
            config=VFPPROD_CONFIG,
            file_suffix="p",
            output_folder_path=output_folder_path,
            reference_model=reference_model,
            table_granularity=table_granularity,
            optimize_hyperparameters=optimize_hyperparameters,
            tuning_metric=tuning_metric,
        )
        if prod_path is not None:
            manifest_content.append(prod_path)

        inj_path = _process_well_stream(
            well_name=well_name,
            stream_name="injection",
            stream_data=flow_data.injection,
            vfp_section=getattr(vfp_details, "VFPINJ", None),
            config=VFPINJ_CONFIG,
            file_suffix="i",
            output_folder_path=output_folder_path,
            reference_model=reference_model,
            table_granularity=table_granularity,
            optimize_hyperparameters=optimize_hyperparameters,
            tuning_metric=tuning_metric,
        )
        if inj_path is not None:
            manifest_content.append(inj_path)

    if manifest_content:
        manifest_path = Path(output_folder_path, "VFP_manifest.txt")
        logger.debug(
            "Exporting VFP manifest",
            extra={"ManifestPath": str(manifest_path)},
        )
        export_VFP_manifest(manifest_content, manifest_path)

    return manifest_content


def run_evaluator(
    source_file_path: Path,
    output_folder_path: Path,
    well_data_filter_path: Path | None = None,
):
    if output_folder_path.exists():
        shutil.rmtree(output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    reader = initialize_reader_from_path(source_file_path)

    well_data_filter = _read_well_data_filter(well_data_filter_path)
    wells_fit_data = reader.calculate_wells_fits(well_data_filter)

    logger.info(
        "Calculated well fits",
        extra={"WellCount": len(wells_fit_data)},
    )

    for well_name, fit_data in wells_fit_data.items():
        logger.info("Exporting well fit data", extra={"Well": well_name})

        folder_path = Path(output_folder_path, f"evaluation_{well_name}")
        folder_path.mkdir(parents=True, exist_ok=True)

        fit_data.data_frame.to_csv(Path(folder_path, "fit_data.csv"), index=False)

        with open(
            Path(folder_path, "overall_fit_results").with_suffix(".json"),
            "w",
        ) as f:
            fit_metrics = fit_data.overall
            json.dump(fit_metrics, f, indent=4)

        with open(
            Path(folder_path, "production_fit_results").with_suffix(".json"),
            "w",
        ) as f:
            fit_metrics = fit_data.production
            json.dump(fit_metrics, f, indent=4)

        with open(
            Path(folder_path, "injection_fit_results").with_suffix(".json"),
            "w",
        ) as f:
            fit_metrics = fit_data.injection
            json.dump(fit_metrics, f, indent=4)

    return wells_fit_data
