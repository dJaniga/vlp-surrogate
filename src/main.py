import logging

import pandas as pd

from ecl import EclSmrReader

pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 1000)

logging.basicConfig(level=logging.DEBUG)


def main():
    summary_file_path = r"xxx"  # path to SLB Eclipse summary file
    r = EclSmrReader.prepare_training_data(summary_file_path)


if __name__ == "__main__":
    main()
