import logging
from datetime import datetime
from pathlib import Path
from typing import Type
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd
import pandera.pandas as pa
from resdata.summary import Summary

from readers.base import ReaderInterface
from readers.models import (
    WellsFlowData,
    FlowData,
    ProductionFlowData,
    InjectionFlowData,
    WellsFitResults,
    FitResults,
    WellDataFilter,
)
from toolbox.fit_metrics import all_fit_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EclipseConfig:
    production_header: tuple[str, ...] = ("WGPRH", "WTHPH", "WWGRH", "WOGRH", "WBHP")
    injection_header: tuple[str, ...] = ("WGIRH", "WTHPH", "WBHP")

    production_rename: Mapping[str, str] = field(
        default_factory=lambda: {
            "FLO": "WGPRH",
            "THP": "WTHPH",
            "WFR": "WWGRH",
            "GFR": "WOGRH",
            "BHP": "WBHP",
        }
    )

    injection_rename: Mapping[str, str] = field(
        default_factory=lambda: {
            "FLO": "WGIRH",
            "THP": "WTHPH",
            "BHP": "WBHP",
        }
    )

    fit_columns: Mapping[str, str] = field(
        default_factory=lambda: {"actual": "WTHPH", "predicted": "WTHP"}
    )

    join_string: str = ":"


class EclipseReader(ReaderInterface):
    def __init__(self, summary: Summary, config: EclipseConfig | None = None):
        self.summary = summary
        self.config = config or EclipseConfig()
        self._wells = list(self.summary.wells())
        logger.debug(
            "EclipseReader initialized",
            extra={
                "wells_count": len(self._wells),
                "join_string": self.config.join_string,
            },
        )

    @classmethod
    def from_file(cls, ecl_smr_file_path: str | Path) -> "EclipseReader":
        path = str(ecl_smr_file_path)
        logger.info("Loading summary file", extra={"path": path})
        try:
            summary = Summary(path, join_string=":", lazy_load=True)
        except Exception:
            logger.exception("Failed to load summary file", extra={"path": path})
            raise
        logger.debug("Summary file loaded", extra={"path": path})
        return cls(summary)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_wells_flow_data(
        self, well_data_filter: WellDataFilter | None = None
    ) -> WellsFlowData:
        wells_to_read, time_from, time_to = self._resolve_filter(well_data_filter)
        logger.info(
            "Reading wells flow data",
            extra={
                "wells_count": len(wells_to_read),
                "time_from": time_from,
                "time_to": time_to,
            },
        )
        self._validate_wells_exist(wells_to_read)

        results: WellsFlowData = {}

        for well in wells_to_read:
            logger.debug("Reading well", extra={"well": well})

            production = self._read_flow_data(
                well,
                header=self.config.production_header,
                rename_map=self.config.production_rename,
                model=ProductionFlowData,
                label="production",
                time_from=time_from,
                time_to=time_to,
            )

            injection = self._read_flow_data(
                well,
                header=self.config.injection_header,
                rename_map=self.config.injection_rename,
                model=InjectionFlowData,
                label="injection",
                time_from=time_from,
                time_to=time_to,
            )

            if production is None and injection is None:
                logger.warning(
                    "No flow data after filtering",
                    extra={"well": well, "time_from": time_from, "time_to": time_to},
                )

            results[well] = FlowData(production=production, injection=injection)

        logger.info(
            "Completed reading wells flow data", extra={"wells_count": len(results)}
        )
        return results

    def calculate_wells_fits(
        self, well_data_filter: WellDataFilter | None = None
    ) -> WellsFitResults:
        wells_to_read, time_from, time_to = self._resolve_filter(well_data_filter)
        logger.info(
            "Calculating wells fits",
            extra={
                "wells_count": len(wells_to_read),
                "time_from": time_from,
                "time_to": time_to,
            },
        )
        self._validate_wells_exist(wells_to_read)

        results: WellsFitResults = {}

        for well in wells_to_read:
            logger.debug("Calculating fit for well", extra={"well": well})
            fit = self._calculate_fit_results(well, time_from, time_to)
            if fit is not None:
                results[well] = fit
            else:
                logger.debug("No fit results for well", extra={"well": well})

        logger.info(
            "Completed calculating wells fits",
            extra={"wells_with_results": len(results)},
        )
        return results

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _read_flow_data(
        self,
        well: str,
        *,
        header: tuple[str, ...],
        rename_map: Mapping[str, str],
        model: Type[pa.DataFrameModel],
        label: str,
        time_from: datetime | None,
        time_to: datetime | None,
    ) -> pd.DataFrame | None:
        column_keys = self._column_keys(header, well)
        logger.debug(
            "Reading flow data frame",
            extra={"well": well, "label": label, "columns_count": len(column_keys)},
        )

        try:
            df = self.summary.pandas_frame(column_keys=column_keys)
        except Exception:
            logger.exception(
                "Failed to read pandas frame from summary",
                extra={"well": well, "label": label, "column_keys": column_keys},
            )
            raise

        self._assert_datetime_index(df)

        before_rows = len(df)
        df = self._filter_non_negative(df, column_keys)
        logger.debug(
            "Applied non-negative filter",
            extra={
                "well": well,
                "label": label,
                "rows_before": before_rows,
                "rows_after": len(df),
            },
        )
        if df.empty:
            logger.info(
                "No rows left after non-negative filter",
                extra={"well": well, "label": label},
            )
            return None

        before_rows = len(df)
        df = self._apply_time_filter(df, time_from, time_to)
        logger.debug(
            "Applied time filter",
            extra={
                "well": well,
                "label": label,
                "time_from": time_from,
                "time_to": time_to,
                "rows_before": before_rows,
                "rows_after": len(df),
            },
        )
        if df.empty:
            logger.info(
                "No rows left after time filter",
                extra={
                    "well": well,
                    "label": label,
                    "time_from": time_from,
                    "time_to": time_to,
                },
            )
            return None

        df = df.reset_index(names="T")
        df = self._rename_columns(df, rename_map, well)

        logger.debug(
            "Validating flow data frame",
            extra={
                "well": well,
                "label": label,
                "rows": len(df),
                "columns": list(df.columns),
            },
        )
        try:
            validated = model.validate(df)
        except Exception:
            logger.exception(
                "Validation failed",
                extra={"well": well, "label": label, "rows": len(df)},
            )
            raise

        logger.info(
            "Flow data read successfully",
            extra={"well": well, "label": label, "rows": len(validated)},
        )
        return validated

    def _calculate_fit_results(
        self,
        well: str,
        time_from: datetime | None,
        time_to: datetime | None,
    ) -> FitResults | None:
        fit_cols = {
            k: f"{v}{self.config.join_string}{well}"
            for k, v in self.config.fit_columns.items()
        }

        prod_key = (
            f"{self.config.production_rename['FLO']}{self.config.join_string}{well}"
        )
        inj_key = (
            f"{self.config.injection_rename['FLO']}{self.config.join_string}{well}"
        )

        column_keys = [*fit_cols.values(), prod_key, inj_key]
        df = self.summary.pandas_frame(column_keys=column_keys)

        missing = [c for c in column_keys if c not in df.columns]
        if missing:
            logger.warning(
                "Missing columns for fit calculation",
                extra={"well": well, "missing": missing, "requested": column_keys},
            )
            return None

        self._assert_datetime_index(df)
        before_rows = len(df)
        df = self._apply_time_filter(df, time_from, time_to)
        if df.empty:
            logger.info(
                "No rows left after time filter for fit calculation",
                extra={
                    "well": well,
                    "time_from": time_from,
                    "time_to": time_to,
                    "rows_before": before_rows,
                },
            )
            return None

        y_actual = df[fit_cols["actual"]].to_numpy(dtype=np.float64, copy=False)
        y_pred = df[fit_cols["predicted"]].to_numpy(dtype=np.float64, copy=False)

        if y_actual.size == 0 or y_actual.shape != y_pred.shape:
            logger.warning(
                "Fit arrays have invalid shape/size",
                extra={
                    "well": well,
                    "y_actual_shape": getattr(y_actual, "shape", None),
                    "y_pred_shape": getattr(y_pred, "shape", None),
                },
            )
            return None

        finite_mask = np.isfinite(y_actual) & np.isfinite(y_pred)
        if not np.any(finite_mask):
            logger.info("No finite points for fit calculation", extra={"well": well})
            return None

        mask_overall = finite_mask
        mask_prod = finite_mask & (df[prod_key].to_numpy() > 0)
        mask_inj = finite_mask & (df[inj_key].to_numpy() > 0)

        overall = all_fit_metrics(y_actual, y_pred, mask_overall)
        production = all_fit_metrics(y_actual, y_pred, mask_prod)
        injection = all_fit_metrics(y_actual, y_pred, mask_inj)

        if _all_nan(overall) and _all_nan(production) and _all_nan(injection):
            logger.info("All fit metrics are NaN; skipping", extra={"well": well})
            return None

        logger.debug(
            "Fit metrics computed",
            extra={
                "well": well,
                "overall": overall,
                "production": production,
                "injection": injection,
                "n_points_overall": int(np.sum(mask_overall)),
                "n_points_prod": int(np.sum(mask_prod)),
                "n_points_inj": int(np.sum(mask_inj)),
            },
        )

        out = pd.DataFrame(
            {
                "T": df.index,
                "measured": y_actual,
                "simulated": y_pred,
                "production_mask": mask_prod,
                "injection_mask": mask_inj,
            },
        )

        return FitResults(
            overall=overall,
            production=production,
            injection=injection,
            data_frame=out,
        )

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _resolve_filter(self, well_data_filter: WellDataFilter | None):
        wells = (
            well_data_filter.wells
            if well_data_filter and well_data_filter.wells
            else self._wells
        )
        time_from = (
            well_data_filter.time.from_T
            if well_data_filter and well_data_filter.time
            else None
        )
        time_to = (
            well_data_filter.time.to_T
            if well_data_filter and well_data_filter.time
            else None
        )
        logger.debug(
            "Resolved well filter",
            extra={
                "wells_count": len(wells),
                "time_from": time_from,
                "time_to": time_to,
            },
        )
        return wells, time_from, time_to

    def _validate_wells_exist(self, wells: list[str]) -> None:
        unknown = set(wells) - set(self._wells)
        if unknown:
            logger.error(
                "Unknown wells requested",
                extra={
                    "unknown": sorted(unknown),
                    "known_wells_count": len(self._wells),
                },
            )
            raise ValueError(f"Unknown wells requested: {sorted(unknown)}")

    def _column_keys(self, headers: tuple[str, ...], well: str) -> list[str]:
        return [f"{h}{self.config.join_string}{well}" for h in headers]

    @staticmethod
    def _assert_datetime_index(df: pd.DataFrame) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error(
                "Expected DataFrame with DatetimeIndex",
                extra={"index_type": type(df.index).__name__},
            )
            raise TypeError("Expected DataFrame with DatetimeIndex")

    @staticmethod
    def _filter_non_negative(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        mask = (df[columns] >= 0).all(axis=1)
        return df.loc[mask]

    @staticmethod
    def _apply_time_filter(
        df: pd.DataFrame,
        time_from: datetime | None,
        time_to: datetime | None,
    ) -> pd.DataFrame:
        if time_from is None and time_to is None:
            return df

        mask = pd.Series(True, index=df.index)

        if time_from is not None:
            mask &= df.index >= time_from

        if time_to is not None:
            mask &= df.index <= time_to

        return df.loc[mask]

    def _rename_columns(
        self,
        df: pd.DataFrame,
        rename_map: Mapping[str, str],
        well: str,
    ) -> pd.DataFrame:
        reverse = {
            f"{v}{self.config.join_string}{well}": k for k, v in rename_map.items()
        }
        ordered = ["T", *rename_map.keys()]
        return df.rename(columns=reverse)[ordered]


def _all_nan(metrics: Mapping[str, float]) -> bool:
    if not metrics:
        return True
    return all(not np.isfinite(v) for v in metrics.values())
