import logging
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandera.pandas as pa
from pandera import Timestamp
from resdata.summary import Summary

logger = logging.getLogger(__name__)


class VLPPData(pa.DataFrameModel):
    T: Timestamp
    FLO: float
    THP: float
    WFR: float
    GFR: float
    BHP: float


class VLPIData(pa.DataFrameModel):
    T: Timestamp
    FLO: float
    THP: float
    BHP: float


@dataclass(frozen=True, slots=True)
class VLPTrainingData:
    well_name: str
    production: VLPPData | None = field(default=None)
    injection: VLPIData | None = field(default=None)


@dataclass(frozen=True, slots=True)
class FitMetrics:
    RMSE: float


@dataclass(frozen=True, slots=True)
class FitResults:
    well_name: str
    fit_metrics: FitMetrics | None = field(default=None)


class EclSmrReader:
    PRODUCTION_HEADER = ["WGPRH", "WTHPH", "WWGRH", "WOGRH", "WBHP"]
    PRODUCTION_REQUIRED = ["WGPRH", "WTHPH", "WBHP"]
    INJECTION_HEADER = ["WGIRH", "WTHPH", "WBHP"]
    INJECTION_REQUIRED = ["WGIRH", "WTHPH", "WBHP"]
    JOIN_STRING = ":"

    PRODUCTION_RENAME_MAP = {
        "WGPRH": "FLO",
        "WTHPH": "THP",
        "WWGRH": "WFR",
        "WOGRH": "GFR",
        "WBHP": "BHP",
    }
    INJECTION_RENAME_MAP = {
        "WGIRH": "FLO",
        "WTHPH": "THP",
        "WBHP": "BHP",
    }
    FIT_COLUMNS = {"Actual": "WTHPH", "Predicted": "WTHP"}

    @classmethod
    def prepare_training_data(cls, ecl_smr_file_path: str) -> list[VLPTrainingData]:
        logger.info("Reading ECL summary file: %s", ecl_smr_file_path)
        ecl_summary = Summary(ecl_smr_file_path, join_string=cls.JOIN_STRING)

        wells = list(ecl_summary.wells())
        logger.info("Found %d wells", len(wells))

        results: list[VLPTrainingData] = []
        for well in wells:
            logger.debug("Preparing training data for well %s", well)
            production = cls._prepare_vlpp_data(ecl_summary, well)
            injection = cls._prepare_vlpi_data(ecl_summary, well)

            if production is None and injection is None:
                logger.debug(
                    "Well %s: no production/injection data after filtering", well
                )

            results.append(
                VLPTrainingData(
                    well_name=well, production=production, injection=injection
                )
            )

        logger.info("Prepared training data for %d wells", len(results))
        return results

    @classmethod
    def prepare_fit_results(cls, ecl_smr_file_path: str) -> list[FitResults]:
        logger.info("Reading ECL summary file: %s", ecl_smr_file_path)
        ecl_summary = Summary(ecl_smr_file_path, join_string=cls.JOIN_STRING)

        wells = list(ecl_summary.wells())
        logger.info("Found %d wells", len(wells))

        results: list[FitResults] = []
        for well in wells:
            logger.debug("Preparing fit results for well %s", well)
            fit_metrics = cls._calculate_fit_results(ecl_summary, well)
            logger.debug("Fit metrics for well %s: %s", well, fit_metrics)

            results.append(FitResults(well_name=well, fit_metrics=fit_metrics))

        return results

    @classmethod
    def _prepare_vlpp_data(cls, summary: Summary, well_name: str) -> VLPPData | None:
        column_keys = [
            f"{h}{cls.JOIN_STRING}{well_name}" for h in cls.PRODUCTION_HEADER
        ]
        required = [f"{h}{cls.JOIN_STRING}{well_name}" for h in cls.PRODUCTION_REQUIRED]

        df = summary.pandas_frame(column_keys=column_keys)

        missing_required = [col for col in required if col not in df.columns]
        if missing_required:
            logger.debug(
                "Well %s (production): missing required columns: %s",
                well_name,
                missing_required,
            )
            return None

        mask = (df[required] != 0).all(axis=1)
        df_required = df.loc[mask].reset_index(names="T")
        if df_required.empty:
            logger.debug(
                "Well %s (production): no rows after non-zero filter", well_name
            )
            return None

        rename_map = {
            f"{k}{cls.JOIN_STRING}{well_name}": v
            for (k, v) in cls.PRODUCTION_RENAME_MAP.items()
        }
        ordered_cols = ["T", *cls.PRODUCTION_RENAME_MAP.values()]
        vlpp_df = df_required.rename(columns=rename_map)[ordered_cols]

        return VLPPData.validate(vlpp_df)

    @classmethod
    def _prepare_vlpi_data(cls, summary: Summary, well_name: str) -> VLPIData | None:
        column_keys = [f"{h}{cls.JOIN_STRING}{well_name}" for h in cls.INJECTION_HEADER]
        required = [f"{h}{cls.JOIN_STRING}{well_name}" for h in cls.INJECTION_REQUIRED]

        df = summary.pandas_frame(column_keys=column_keys)

        missing_required = [col for col in required if col not in df.columns]
        if missing_required:
            logger.debug(
                "Well %s (injection): missing required columns: %s",
                well_name,
                missing_required,
            )
            return None

        mask = (df[required] != 0).all(axis=1)
        df_required = df.loc[mask].reset_index(names="T")
        if df_required.empty:
            logger.debug(
                "Well %s (injection): no rows after non-zero filter", well_name
            )
            return None

        rename_map = {
            f"{k}{cls.JOIN_STRING}{well_name}": v
            for (k, v) in cls.INJECTION_RENAME_MAP.items()
        }
        ordered_cols = ["T", *cls.INJECTION_RENAME_MAP.values()]
        vlpi_df = df_required.rename(columns=rename_map)[ordered_cols]

        return VLPIData.validate(vlpi_df)

    @classmethod
    def _calculate_fit_results(
        cls, summary: Summary, well_name: str
    ) -> FitMetrics | None:
        well_fit_columns = {
            k: f"{v}{cls.JOIN_STRING}{well_name}" for k, v in cls.FIT_COLUMNS.items()
        }
        df = summary.pandas_frame(column_keys=[v for v in well_fit_columns.values()])
        y_actual = df[well_fit_columns["Actual"]].to_numpy()
        y_pred = df[well_fit_columns["Predicted"]].to_numpy()
        if y_actual.size == 0 or y_pred.size == 0:
            raise ValueError("Input arrays must not be empty.")
        if y_actual.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_actual {y_actual.shape} != y_pred {y_pred.shape}"
            )
        if not np.any(y_actual) and not np.any(y_pred):
            logger.debug("Well %s: no rows after non-zero filter", well_name)
            return None

        rmse = cls._calculate_rmse(y_actual, y_pred)
        return FitMetrics(RMSE=rmse)

    @classmethod
    def _calculate_rmse(
        cls, y_actual: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
    ) -> float:
        diff = y_actual - y_pred
        mse = np.mean(np.square(diff))
        rmse = float(np.sqrt(mse))

        return rmse
