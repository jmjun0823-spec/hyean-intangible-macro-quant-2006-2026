from typing import Iterable, Optional
import numpy as np
import pandas as pd


def get_business_calendar(start_date: str, end_date: str, market: str = "US") -> pd.DatetimeIndex:
    """
    시장(US / KR_KOSPI)별 평일 캘린더를 생성합니다.
    주말 및 주요 고정 공휴일(US: New Year/Independence/Christmas, KR: New Year/Samiljeol/Children/Hangeul/Christmas)을 제외합니다.
    """
    idx = pd.bdate_range(start=start_date, end=end_date)
    if len(idx) == 0:
        return idx

    common_holidays = {(1, 1), (12, 25)}
    if market.upper().startswith("KR"):
        market_holidays = common_holidays | {(3, 1), (5, 5), (10, 9)}
    else:
        market_holidays = common_holidays | {(7, 4)}

    mask = [((dt.month, dt.day) not in market_holidays) for dt in idx]
    return idx[mask]


def align_market_time_series(
    df: Optional[pd.DataFrame],
    dates: Iterable[pd.Timestamp],
    value_cols: Iterable[str],
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """
    빈 DataFrame/휴장일/결측치를 포함한 시계열을 기준 날짜 인덱스에 정렬합니다.
    빈 응답이 들어와도 예외 없이 NaN/0으로 채워진 정렬 프레임을 반환합니다.
    """
    target_idx = pd.DatetimeIndex(pd.to_datetime(list(dates))).sort_values().unique()
    cols = list(value_cols)

    if df is None or df.empty:
        out = pd.DataFrame(index=target_idx, columns=cols, dtype=float)
        out.index.name = "date"
        return out.fillna(0.0)

    work = df.copy()
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"])
        work = work.sort_values("date").drop_duplicates(subset=["date"], keep="last").set_index("date")
    else:
        work.index = pd.to_datetime(work.index)
        work = work.sort_index()

    existing_cols = [c for c in cols if c in work.columns]
    aligned = work[existing_cols].reindex(target_idx)

    for c in cols:
        if c not in aligned.columns:
            aligned[c] = np.nan

    if fill_method == "ffill":
        aligned[cols] = aligned[cols].ffill()
    elif fill_method == "zero":
        aligned[cols] = aligned[cols].fillna(0.0)

    aligned.index.name = "date"
    return aligned[cols]


def apply_signal_lag(series: pd.Series, lag_periods: int = 1) -> pd.Series:
    """
    시점 t에 관측·산출된 시그널이 미래 정보를 선반영하지 않도록 lag_periods만큼 뒤로 미룹니다.
    lag_periods <= 0 이면 원본을 반환합니다.
    """
    if lag_periods <= 0:
        return series.copy()
    return series.shift(lag_periods)
