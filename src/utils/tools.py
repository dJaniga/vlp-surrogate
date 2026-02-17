import glob
import json
import logging
import os

from pydantic import Json

logger = logging.getLogger(__name__)


def _parse_vfp_file(filepath: str) -> tuple[int, float] | None:
    with open(filepath, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith(("VFPPROD", "VFPINJ")):
            # Search forward for first non-empty line
            for next_line in lines[i + 1 :]:
                header_line = next_line.strip()

                if not header_line:
                    continue  # skip blank lines

                header_line = header_line.replace("/", "").strip()
                parts = header_line.split()

                if len(parts) < 2:
                    return None  # malformed header

                try:
                    first_value = int(parts[0])
                    second_value = float(parts[1])
                except ValueError:
                    return None  # invalid numeric conversion

                return first_value, second_value

            return None  # no header line found after keyword

    return None  # no VFPPROD / VFPINJ found


def _collect_vfp_data(folder_path: str) -> Json:
    results: dict[str, dict[str, dict[str, float]]] = {}

    for filepath in glob.glob(os.path.join(folder_path, "*.vfp")):
        filename = os.path.basename(filepath)
        logger.info("Processing %s", filename)
        well_name = filename.replace(".vfp", "")
        is_inj = well_name.endswith("inj")
        base_name = well_name.replace("inj", "")

        v = _parse_vfp_file(filepath)

        if v is None:
            continue

        first_value, second_value = v

        if base_name not in results:
            results[base_name] = {}

        key = "VFPINJ" if is_inj else "VFPPROD"
        results[base_name][key] = {
            "table number": first_value,
            "bhp depth": second_value,
        }

    return results


def vfp_well_data_collector(folder_path: str, output_json_path: str) -> None:
    logger.info("Collecting VFP data from %s", folder_path)
    data = _collect_vfp_data(folder_path)

    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info("Collected VFP data to %s", output_json_path)
