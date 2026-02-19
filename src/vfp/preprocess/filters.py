from collections.abc import Sequence

import pandas as pd


def filter_valid_operating_conditions(
    df: pd.DataFrame, keys: Sequence[str]
) -> pd.DataFrame:
    mask = (df[keys] > 0).all(axis=1)
    return df.loc[mask]
