from __future__ import annotations

import pandas as pd


def pct_return(series: pd.Series, periods: int) -> pd.Series:
    return series.pct_change(periods=periods)


def rolling_volatility(series: pd.Series, window: int) -> pd.Series:
    return series.pct_change().rolling(window=window).std()


def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()

