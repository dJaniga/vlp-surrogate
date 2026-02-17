import logging
from pathlib import Path

from modeling.linear_regressor import LinearRegressionModel
from readers import EclipseReader
from vfp import run_vfp_pipeline, VFPPROD_CONFIG, VFPINJ_CONFIG

logging.basicConfig(level=logging.INFO)


def main():
    p = "/home/octopus/Projects/vlp-surrogate/example_unsmry/SIM.UNSMRY"
    t = "/home/octopus/Projects/vlp-surrogate/vfp_test"

    r = EclipseReader.read_wells_flow_data(p)

    reference_model = LinearRegressionModel()

    for well_name, data in r.items():
        if (
            not data.production is None and not data.production.empty
        ):  # production pipeline
            tp = run_vfp_pipeline(
                Path(t, f"{well_name}_p.vfp"),
                data.production,
                reference_model,
                1,
                900,
                VFPPROD_CONFIG,
            )
        if not data.injection is None and not data.injection.empty:
            ti = run_vfp_pipeline(
                Path(t, f"{well_name}_i.vfp"),
                data.injection,
                reference_model,
                1,
                900,
                VFPINJ_CONFIG,
            )


if __name__ == "__main__":
    main()
