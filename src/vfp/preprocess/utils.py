from collections.abc import Sequence
from typing import NamedTuple

import pandas as pd


class TrainingData(NamedTuple):
    features: pd.DataFrame
    target: pd.DataFrame


def select_columns_by_keys(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    return df[keys]


def get_training_data(
    df: pd.DataFrame, features_keys: Sequence[str], targets_keys: Sequence[str]
) -> TrainingData:
    return TrainingData(
        select_columns_by_keys(df, features_keys),
        select_columns_by_keys(df, targets_keys),
    )
