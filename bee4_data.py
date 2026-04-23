"""
bee4_data.py
=============
OHLCV loading and WaveTrend indicator preparation for bee4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bee4_params import (
    WT_CHANNEL_LEN,
    WT_AVG_LEN,
    WT_SIGNAL_LEN,
    WT_CHANNEL_LEN_GRID,
    WT_AVG_LEN_GRID,
    WT_SIGNAL_LEN_GRID,
    WT_EMA_FILTER_LEN,
    WT_EMA_FILTER_LEN_OPTIONS,
)


def wt1_column(channel_len: int, avg_len: int) -> str:
    return f"wt1_c{int(channel_len)}_a{int(avg_len)}"


def wt2_column(channel_len: int, avg_len: int, signal_len: int) -> str:
    return f"wt2_c{int(channel_len)}_a{int(avg_len)}_s{int(signal_len)}"


def wt_columns(channel_len: int, avg_len: int, signal_len: int) -> tuple[str, str]:
    return wt1_column(channel_len, avg_len), wt2_column(channel_len, avg_len, signal_len)


def _bars_since_flag(flag: pd.Series) -> pd.Series:
    values: list[float] = []
    last_idx: int | None = None
    for idx, is_true in enumerate(flag.fillna(False).astype(bool).to_list()):
        if is_true:
            last_idx = idx
        values.append(np.nan if last_idx is None else float(idx - last_idx))
    return pd.Series(values, index=flag.index, dtype="float64")


def compute_wave_trend(
    df: pd.DataFrame,
    channel_len: int,
    avg_len: int,
    signal_len: int,
) -> tuple[pd.Series, pd.Series]:
    """
    Classic WaveTrend oscillator:
      ap  = HLC3
      esa = EMA(ap, channel_len)
      d   = EMA(abs(ap - esa), channel_len)
      ci  = (ap - esa) / (0.015 * d)
      wt1 = EMA(ci, avg_len)
      wt2 = SMA(wt1, signal_len)
    """
    ap = (df["high"] + df["low"] + df["close"]) / 3.0
    esa = ap.ewm(span=int(channel_len), adjust=False).mean()
    d = (ap - esa).abs().ewm(span=int(channel_len), adjust=False).mean()
    d = d.replace(0.0, np.nan)
    ci = (ap - esa) / (0.015 * d)
    wt1 = ci.ewm(span=int(avg_len), adjust=False).mean()
    wt2 = wt1.rolling(int(signal_len)).mean()
    return wt1, wt2


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add precomputed WaveTrend columns for all WFO combinations plus default aliases:
      - hlc3
      - wt1_cX_aY
      - wt2_cX_aY_sZ
      - ema_X filter columns
      - wt1, wt2, wt_delta
      - wt_green_dot, wt_red_dot
      - bars_since_wt_green_dot, bars_since_wt_red_dot
    """
    df = df.copy()
    df["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3.0

    channel_lens = sorted(set([WT_CHANNEL_LEN] + list(WT_CHANNEL_LEN_GRID)))
    avg_lens = sorted(set([WT_AVG_LEN] + list(WT_AVG_LEN_GRID)))
    signal_lens = sorted(set([WT_SIGNAL_LEN] + list(WT_SIGNAL_LEN_GRID)))

    for channel_len in channel_lens:
        ap = df["hlc3"]
        esa = ap.ewm(span=int(channel_len), adjust=False).mean()
        d = (ap - esa).abs().ewm(span=int(channel_len), adjust=False).mean().replace(0.0, np.nan)
        ci = (ap - esa) / (0.015 * d)

        for avg_len in avg_lens:
            wt1_col = wt1_column(channel_len, avg_len)
            wt1 = ci.ewm(span=int(avg_len), adjust=False).mean()
            df[wt1_col] = wt1

            for signal_len in signal_lens:
                df[wt2_column(channel_len, avg_len, signal_len)] = wt1.rolling(int(signal_len)).mean()

    default_wt1_col, default_wt2_col = wt_columns(WT_CHANNEL_LEN, WT_AVG_LEN, WT_SIGNAL_LEN)
    ema_lens = sorted(set([WT_EMA_FILTER_LEN] + list(WT_EMA_FILTER_LEN_OPTIONS)))
    for ema_len in ema_lens:
        df[f"ema_{int(ema_len)}"] = df["close"].ewm(span=int(ema_len), adjust=False).mean()
    df["ema20"] = df["ema_20"]
    df["wt1"] = df[default_wt1_col]
    df["wt2"] = df[default_wt2_col]
    df["wt_delta"] = df["wt1"] - df["wt2"]
    df["wt_green_dot"] = (df["wt1"].shift(1) <= df["wt2"].shift(1)) & (df["wt1"] > df["wt2"])
    df["wt_red_dot"] = (df["wt1"].shift(1) >= df["wt2"].shift(1)) & (df["wt1"] < df["wt2"])
    df["bars_since_wt_green_dot"] = _bars_since_flag(df["wt_green_dot"])
    df["bars_since_wt_red_dot"] = _bars_since_flag(df["wt_red_dot"])

    return df


def load_klines(csv_path: str) -> pd.DataFrame:
    """
    Load candle CSV.
    Supports open_time as milliseconds, seconds or string timestamps.
    """
    df = pd.read_csv(csv_path)
    df.rename(columns={c: c.lower() for c in df.columns}, inplace=True)

    if "open_time" in df.columns:
        col = df["open_time"]
        try:
            is_numeric = np.issubdtype(col.dtype, np.number)
        except TypeError:
            is_numeric = False
        if is_numeric:
            unit = "ms" if col.max() > 1e12 else "s"
            df["time"] = pd.to_datetime(col, unit=unit, utc=True)
        else:
            df["time"] = pd.to_datetime(col, utc=True, errors="coerce")
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    else:
        raise ValueError("Missing open_time / time column in CSV.")

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing '{col}' column in CSV.")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def format_ts(ts) -> str:
    """Pretty UTC timestamp string."""
    if pd.isna(ts):
        return "NaT"
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%d %H:%M")

