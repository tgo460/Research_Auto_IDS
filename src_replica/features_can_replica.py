from typing import List

import pandas as pd

from src_replica.preprocessing_standard import (
    BYTE_COLS,
    STANDARD_CAN_ENGINEERING_WINDOW,
    add_standard_can_features,
)


def add_can_engineered_features(
    df_raw: pd.DataFrame,
    window: int = STANDARD_CAN_ENGINEERING_WINDOW,
) -> pd.DataFrame:
    return add_standard_can_features(df_raw, window=window)


__all__: List[str] = ["BYTE_COLS", "add_can_engineered_features"]
