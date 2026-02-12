from collections.abc import Sequence
from typing import Type

import numpy as np
import pandas as pd


def to_features(df: Type[pd.DataFrame], keys: Sequence[str]) -> np.ndarray:
    return df[keys].to_numpy()

def to_targets(df: Type[pd.DataFrame], keys: Sequence[str]) -> np.ndarray:
    return df[keys].to_numpy()