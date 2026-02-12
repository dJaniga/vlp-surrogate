import logging
from pathlib import Path
from time import perf_counter
from typing import ClassVar, Mapping, Type

import numpy as np
import numpy.typing as npt
import pandera.pandas as pa
from resdata.summary import Summary

from readers.interfaces import MetricFn
from readers.models import VLPTrainingData, FitResults, VLPPData, VLPIData

logger = logging.getLogger(__name__)


class EclipseSummaryReader:
    PRODUCTION_HEADER = ["WGPRH", "WTHPH", "WWGRH", "WOGRH", "WBHP"]
    PRODUCTION_REQUIRED = ["WGPRH", "WTHPH", "WBHP"]

    INJECTION_HEADER = ["WGIRH", "WTHPH", "WBHP"]
    INJECTION_REQUIRED = ["WGIRH", "WTHPH", "WBHP"]

    JOIN_STRING = ":"

    # Canonical column names -> ECL keys
    PRODUCTION_RENAME_MAP = {
        "FLO": "WGPRH",
        "THP": "WTHPH",
        "WFR": "WWGRH",
        "GFR": "WOGRH",
        "BHP": "WBHP",
    }
    INJECTION_RENAME_MAP = {
        "FLO": "WGIRH",
        "THP": "WTHPH",
        "BHP": "WBHP",
    }

    FIT_COLUMNS = {"Actual": "WTHPH", "Predicted": "WTHP"}

    _METRICS: ClassVar[dict[str, MetricFn]] = {}

    # ---------------------------
    # Public API
    # ---------------------------
    @classmethod
    def prepare_training_data(
        cls, ecl_smr_file_path: str | Path
    ) -> list[VLPTrainingData]:
        """Read an ECL summary file and return per-well training datasets for VLP."""
        t0 = perf_counter()
        summary, wells = cls._load_summary_and_wells(ecl_smr_file_path)

        n_prod = 0
        n_inj = 0
        n_empty = 0

        results: list[VLPTrainingData] = []
        for well in wells:
            logger.debug("Preparing training data for well=%s", well)
            production = cls._prepare_vlpp_data(summary, well)
            injection = cls._prepare_vlpi_data(summary, well)

            if production is None:
                logger.debug("Well=%s: production training data unavailable", well)
            else:
                n_prod += 1

            if injection is None:
                logger.debug("Well=%s: injection training data unavailable", well)
            else:
                n_inj += 1

            if production is None and injection is None:
                n_empty += 1
                logger.info(
                    "Well=%s: no production/injection data after filtering", well
                )

            results.append(
                VLPTrainingData(
                    well_name=well, production=production, injection=injection
                )
            )

        elapsed = perf_counter() - t0
        logger.info(
            "Prepared training data: wells=%d, prod=%d, inj=%d, empty=%d, elapsed=%.3fs",
            len(wells),
            n_prod,
            n_inj,
            n_empty,
            elapsed,
        )
        return results

    @classmethod
    def prepare_fit_results(cls, ecl_smr_file_path: str | Path) -> list[FitResults]:
        """Read an ECL summary file and compute per-well fit metrics for registered metrics."""
        t0 = perf_counter()
        summary, wells = cls._load_summary_and_wells(ecl_smr_file_path)

        if not cls._METRICS:
            logger.warning(
                "No metrics registered; fit results will be empty. "
                "Register at least one metric via EclSmrReader.register_metric()."
            )
        else:
            logger.info(
                "Computing fit results using metrics=%s", ", ".join(cls.list_metrics())
            )

        n_ok = 0
        n_skipped = 0

        results: list[FitResults] = []
        for well in wells:
            logger.debug("Preparing fit results for well=%s", well)
            fit_results = cls._calculate_fit_results(summary, well)
            if fit_results is None:
                n_skipped += 1
                logger.info("Well=%s: fit results skipped (missing/empty data)", well)
                continue
            n_ok += 1
            results.append(fit_results)

        elapsed = perf_counter() - t0
        logger.info(
            "Prepared fit results: wells=%d, ok=%d, skipped=%d, elapsed=%.3fs",
            len(wells),
            n_ok,
            n_skipped,
            elapsed,
        )
        return results

    # ---------------------------
    # Metric registry
    # ---------------------------
    @classmethod
    def register_metric(
        cls, name: str, fn: MetricFn, *, overwrite: bool = False
    ) -> None:
        """Register a new metric function available to fit calculations."""
        if not name or not name.strip():
            raise ValueError("Metric name must be a non-empty string.")
        if (name in cls._METRICS) and not overwrite:
            raise ValueError(
                f"Metric '{name}' is already registered. Use overwrite=True."
            )

        cls._METRICS[name] = fn
        logger.info(
            "Registered metric name=%s overwrite=%s (total=%d)",
            name,
            overwrite,
            len(cls._METRICS),
        )

    @classmethod
    def unregister_metric(cls, name: str) -> None:
        removed = cls._METRICS.pop(name, None) is not None
        logger.info(
            "Unregistered metric name=%s removed=%s (total=%d)",
            name,
            removed,
            len(cls._METRICS),
        )

    @classmethod
    def list_metrics(cls) -> tuple[str, ...]:
        return tuple(cls._METRICS.keys())

    # ---------------------------
    # Summary loading
    # ---------------------------
    @classmethod
    def _load_summary_and_wells(
        cls, ecl_smr_file_path: str | Path
    ) -> tuple[Summary, list[str]]:
        path = str(ecl_smr_file_path)
        logger.info(
            "Reading ECL summary file path=%s join_string=%s", path, cls.JOIN_STRING
        )
        try:
            summary = Summary(path, join_string=cls.JOIN_STRING)
            wells = list(summary.wells())
        except Exception:
            logger.exception("Failed to read ECL summary file path=%s", path)
            raise

        logger.info("Summary loaded successfully: wells=%d", len(wells))
        logger.debug("Wells=%s", wells)
        return summary, wells

    # ---------------------------
    # Training data preparation
    # ---------------------------
    @classmethod
    def _prepare_vlpp_data(cls, summary: Summary, well_name: str) -> VLPPData | None:
        return cls._prepare_vlp_data(
            summary=summary,
            well_name=well_name,
            header=cls.PRODUCTION_HEADER,
            required=cls.PRODUCTION_REQUIRED,
            rename_map=cls.PRODUCTION_RENAME_MAP,
            model=VLPPData,
            label="production",
        )

    @classmethod
    def _prepare_vlpi_data(cls, summary: Summary, well_name: str) -> VLPIData | None:
        return cls._prepare_vlp_data(
            summary=summary,
            well_name=well_name,
            header=cls.INJECTION_HEADER,
            required=cls.INJECTION_REQUIRED,
            rename_map=cls.INJECTION_RENAME_MAP,
            model=VLPIData,
            label="injection",
        )

    @classmethod
    def _prepare_vlp_data(
        cls,
        *,
        summary: Summary,
        well_name: str,
        header: list[str],
        required: list[str],
        rename_map: dict[str, str],
        model: Type[pa.DataFrameModel],
        label: str,
    ) -> pa.DataFrameModel | None:
        column_keys = [f"{h}{cls.JOIN_STRING}{well_name}" for h in header]
        required_cols = [f"{h}{cls.JOIN_STRING}{well_name}" for h in required]

        df = summary.pandas_frame(column_keys=column_keys)
        logger.debug(
            "Well=%s (%s): loaded frame rows=%d cols=%d",
            well_name,
            label,
            len(df),
            len(df.columns),
        )

        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            logger.info(
                "Well=%s (%s): missing required columns=%s",
                well_name,
                label,
                missing_required,
            )
            return None

        mask = (df[required_cols] != 0).all(axis=1)
        df_required = df.loc[mask].reset_index(names="T")
        logger.debug(
            "Well=%s (%s): non-zero filter kept=%d/%d",
            well_name,
            label,
            len(df_required),
            len(df),
        )
        if df_required.empty:
            logger.info("Well=%s (%s): no rows after non-zero filter", well_name, label)
            return None

        # Convert "ECLKEY:WELL" -> "CANONICAL"
        reverse_rename = {
            f"{v}{cls.JOIN_STRING}{well_name}": k for (k, v) in rename_map.items()
        }
        ordered_cols = ["T", *rename_map.keys()]
        vlp_df = df_required.rename(columns=reverse_rename)[ordered_cols]

        try:
            validated = model.validate(vlp_df)
        except Exception:
            logger.exception("Well=%s (%s): validation failed", well_name, label)
            raise

        logger.debug(
            "Well=%s (%s): validation ok rows=%d", well_name, label, len(vlp_df)
        )
        return validated

    # ---------------------------
    # Fit metric calculations
    # ---------------------------
    @classmethod
    def _calculate_fit_results(
        cls, summary: Summary, well_name: str
    ) -> FitResults | None:
        well_fit_columns = {
            k: f"{v}{cls.JOIN_STRING}{well_name}" for k, v in cls.FIT_COLUMNS.items()
        }

        production = f"{cls.PRODUCTION_RENAME_MAP['FLO']}{cls.JOIN_STRING}{well_name}"
        injection = f"{cls.INJECTION_RENAME_MAP['FLO']}{cls.JOIN_STRING}{well_name}"

        column_keys = [*well_fit_columns.values(), production, injection]
        df = summary.pandas_frame(column_keys=column_keys)

        required_cols = [*well_fit_columns.values(), production, injection]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.info("Well=%s: missing fit columns=%s", well_name, missing)
            return None

        y_actual = df[well_fit_columns["Actual"]].to_numpy(dtype=np.float64, copy=False)
        y_pred = df[well_fit_columns["Predicted"]].to_numpy(
            dtype=np.float64, copy=False
        )

        if y_actual.size == 0 or y_pred.size == 0:
            logger.info("Well=%s: empty actual/pred arrays", well_name)
            return None
        if y_actual.shape != y_pred.shape:
            logger.info(
                "Well=%s: shape mismatch y_actual=%s y_pred=%s",
                well_name,
                y_actual.shape,
                y_pred.shape,
            )
            return None

        if not np.any(y_actual) and not np.any(y_pred):
            logger.debug("Well %s: no rows after non-zero filter", well_name)
            return None

        # Masks for slices
        mask_overall = np.ones_like(y_actual, dtype=bool)
        mask_production = (df[production] > 0).to_numpy()
        mask_injection = (df[injection] > 0).to_numpy()

        logger.debug(
            "Well=%s: samples total=%d production=%d injection=%d",
            well_name,
            y_actual.size,
            int(np.sum(mask_production)),
            int(np.sum(mask_injection)),
        )

        overall = cls._compute_metrics(
            y_actual, y_pred, mask_overall, context=f"well={well_name} slice=overall"
        )
        production_metrics = cls._compute_metrics(
            y_actual,
            y_pred,
            mask_production,
            context=f"well={well_name} slice=production",
        )
        injection_metrics = cls._compute_metrics(
            y_actual,
            y_pred,
            mask_injection,
            context=f"well={well_name} slice=injection",
        )

        if (
            _all_nan(overall)
            and _all_nan(production_metrics)
            and _all_nan(injection_metrics)
        ):
            logger.info("Well=%s: all metrics are NaN (no usable rows)", well_name)
            return None

        logger.debug(
            "Well=%s: metrics computed overall=%s production=%s injection=%s",
            well_name,
            overall,
            production_metrics,
            injection_metrics,
        )

        return FitResults(
            well_name=well_name,
            overall=overall,
            production=production_metrics,
            injection=injection_metrics,
        )

    @classmethod
    def _compute_metrics(
        cls,
        y_actual: npt.NDArray[np.float64],
        y_pred: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool_],
        *,
        context: str,
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        if not cls._METRICS:
            logger.debug("No metrics registered (%s)", context)
            return out

        for name, fn in cls._METRICS.items():
            t0 = perf_counter()
            try:
                value = float(fn(y_actual, y_pred, mask))
            except Exception:
                logger.exception(
                    "Metric failed name=%s (%s); recording NaN", name, context
                )
                value = float("nan")

            out[name] = value
            elapsed = perf_counter() - t0
            logger.debug(
                "Metric computed name=%s value=%s elapsed=%.4fs (%s)",
                name,
                value,
                elapsed,
                context,
            )

        return out


def _all_nan(metrics: Mapping[str, float]) -> bool:
    if not metrics:
        return True
    return all(not np.isfinite(v) for v in metrics.values())
