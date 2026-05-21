"""
test_bee4.py
============
Pytest suite for the BEE4_4 WaveTrend H1 + closed H4 branch.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bee4_binance
from bee4_binance import interval_to_ms, wfo_bars
from bee4_data import (
    compute_wave_trend,
    htf_prev_wt1_column,
    htf_prev_wt2_column,
    htf_wt1_column,
    htf_wt2_column,
    load_klines,
    prepare_indicators,
    wt_columns,
)
from bee4_engine import (
    BarData,
    PositionState,
    apply_slippage,
    bar_from_row,
    generate_entry_signal,
    generate_exit_signal,
    generate_partial_exit_signal,
)
from bee4_strategy import Bee4Strategy
from bee4_wfo import get_latest_best_params, walk_forward_optimization
from bee4_wfo_scoring import score_params


BASE_PARAMS = {
    "wt_channel_len": 10,
    "wt_avg_len": 21,
    "wt_signal_len": 3,
    "wt_min_signal_level": 0.0,
    "wt_zero_line": 0.0,
    "trade_direction": "long",
    "allow_longs": True,
    "allow_shorts": False,
    "short_trading_enabled": False,
    "wt_long_entry_window_bars": 3,
    "wt_long_entry_max_above_zero": -30.0,
    "wt_long_close_min_level": 40.0,
    "wt_long_exit_min_level": 40.0,
    "wt_long_require_ema20_reclaim": False,
    "wt_long_require_htf_trend": False,
    "wt_short_entry_window_bars": 0,
    "wt_short_entry_min_below_zero": 30.0,
    "wt_short_exit_max_level": 0.0,
    "wt_short_require_ema20_reject": False,
    "wt_short_require_htf_trend": False,
    "wt_ema_filter_len": 20,
    "wt_h4_filter_interval": "4h",
    "wt_h4_long_filter_max": 20.0,
    "wt_h4_long_close_min": 10.0,
    "wt_h4_short_filter_min": 50.0,
    "wt_long_tp1_enabled": True,
    "wt_long_tp1_pct": 0.01,
    "wt_long_tp1_fraction": 1.0 / 3.0,
    "wt_long_tp1_breakeven_enabled": True,
    "wt_long_tp2_enabled": True,
    "wt_long_tp2_pct": 0.02,
    "wt_long_tp2_fraction": 1.0 / 3.0,
    "wt_long_tp1_timeout_hours": 72.0,
    "wt_long_emergency_sl_enabled": False,
    "wt_long_emergency_sl_capital_pct": 0.0,
    "wt_short_tp1_enabled": True,
    "wt_short_tp1_pct": 0.01,
    "wt_short_tp1_fraction": 1.0 / 3.0,
    "atr_stop_enabled": False,
    "atr_stop_multiplier": 2.0,
    "breakeven_trigger_atr": 0.0,
    "trailing_trigger_atr": 0.0,
    "trailing_distance_atr": 1.5,
    "wt_exhaustion_exit_enabled": False,
    "wt_exhaustion_min_level": 50.0,
    "fee_rate": 0.00035,
    "slippage_bps": 0.0,
    "spread_bps": 0.0,
    "max_bars_in_trade": 0,
}

LONG_ONLY_PARAMS = {
    **BASE_PARAMS,
    "trade_direction": "long",
    "allow_longs": True,
    "allow_shorts": False,
}

SHORT_ONLY_PARAMS = {
    **BASE_PARAMS,
    "trade_direction": "short",
    "allow_longs": False,
    "allow_shorts": True,
    "short_trading_enabled": True,
}

SHORTS_ENABLED_PARAMS = {
    **BASE_PARAMS,
    "trade_direction": "both",
    "allow_longs": True,
    "allow_shorts": True,
    "short_trading_enabled": True,
}


def _make_bar(
    time=pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
    close=1800.0,
    wt1=-40.0,
    wt2=-45.0,
    h4_wt1=-28.0,
    h4_wt2=-22.0,
    h4_prev_wt1=-34.0,
    h4_prev_wt2=-22.0,
):
    return BarData(
        time=time,
        open=close,
        high=close + 5.0,
        low=close - 5.0,
        close=close,
        wt1=wt1,
        wt2=wt2,
        wt_delta=wt1 - wt2,
        h4_wt1=h4_wt1,
        h4_wt2=h4_wt2,
        h4_wt_delta=h4_wt1 - h4_wt2,
        h4_prev_wt1=h4_prev_wt1,
        h4_prev_wt2=h4_prev_wt2,
        h4_prev_wt_delta=h4_prev_wt1 - h4_prev_wt2,
        ema20=np.nan,
        ema_filter_len=20,
        htf_ema200=np.nan,
        atr=25.0,
        wt_green_dot=False,
        wt_red_dot=False,
        bars_since_wt_green_dot=np.nan,
        bars_since_wt_red_dot=np.nan,
    )


def _signal_df() -> pd.DataFrame:
    wt1_col, wt2_col = wt_columns(
        BASE_PARAMS["wt_channel_len"],
        BASE_PARAMS["wt_avg_len"],
        BASE_PARAMS["wt_signal_len"],
    )
    h4_wt1_col = htf_wt1_column(
        BASE_PARAMS["wt_channel_len"],
        BASE_PARAMS["wt_avg_len"],
        BASE_PARAMS["wt_h4_filter_interval"],
    )
    h4_wt2_col = htf_wt2_column(
        BASE_PARAMS["wt_channel_len"],
        BASE_PARAMS["wt_avg_len"],
        BASE_PARAMS["wt_signal_len"],
        BASE_PARAMS["wt_h4_filter_interval"],
    )
    times = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    closes = [1800.0, 1810.0, 1830.0, 1820.0]
    wt1_vals = [-48.0, -34.0, 44.0, 50.0]
    wt2_vals = [-42.0, -44.0, 36.0, 56.0]
    h4_wt1_vals = [-34.0, -28.0, 62.0, 58.0]
    h4_wt2_vals = [-22.0, -22.0, 50.0, 52.0]

    df = pd.DataFrame(
        {
            "time": times,
            "open": closes,
            "high": [c + 6.0 for c in closes],
            "low": [c - 6.0 for c in closes],
            "close": closes,
            "volume": [1000.0] * len(closes),
            wt1_col: wt1_vals,
            wt2_col: wt2_vals,
            h4_wt1_col: h4_wt1_vals,
            h4_wt2_col: h4_wt2_vals,
            "wt1": wt1_vals,
            "wt2": wt2_vals,
            "wt_delta": np.array(wt1_vals) - np.array(wt2_vals),
            "h4_wt1": h4_wt1_vals,
            "h4_wt2": h4_wt2_vals,
        }
    )
    df["h4_wt_delta"] = df["h4_wt1"] - df["h4_wt2"]
    df["h4_prev_wt1"] = df["h4_wt1"].shift(1)
    df["h4_prev_wt2"] = df["h4_wt2"].shift(1)
    df["h4_prev_wt_delta"] = df["h4_wt_delta"].shift(1)
    df["ema20"] = pd.Series(closes).ewm(span=20, adjust=False).mean()
    df["ema_20"] = df["ema20"]
    df["htf_ema200"] = pd.Series(closes).ewm(span=200, adjust=False).mean()
    df["atr"] = [20.0] * len(df)
    df["wt_green_dot"] = (df["wt1"].shift(1) <= df["wt2"].shift(1)) & (df["wt1"] > df["wt2"])
    df["wt_red_dot"] = (df["wt1"].shift(1) >= df["wt2"].shift(1)) & (df["wt1"] < df["wt2"])
    df["bars_since_wt_green_dot"] = [np.nan, 0.0, 1.0, 2.0]
    df["bars_since_wt_red_dot"] = [np.nan, np.nan, np.nan, 0.0]
    return df


def _write_csv(path: Path, column: str, values: list[str]) -> Path:
    df = pd.DataFrame(
        {
            column: values,
            "open": [100.0] * len(values),
            "high": [101.0] * len(values),
            "low": [99.0] * len(values),
            "close": [100.0] * len(values),
            "volume": [1.0] * len(values),
        }
    )
    df.to_csv(path, index=False)
    return path


class TestWaveTrendPreparation:
    def test_prepare_indicators_adds_h4_columns(self):
        times = pd.date_range("2024-01-01", periods=240, freq="1h", tz="UTC")
        close = 1800.0 + np.sin(np.linspace(0, 30, len(times))) * 60.0
        df = pd.DataFrame(
            {
                "time": times,
                "open": close,
                "high": close + 4.0,
                "low": close - 4.0,
                "close": close,
                "volume": 1000.0,
            }
        )

        out = prepare_indicators(df)
        wt1_col, wt2_col = wt_columns(10, 21, BASE_PARAMS["wt_signal_len"])
        h4_wt1_col = htf_wt1_column(10, 21, "4h")
        h4_wt2_col = htf_wt2_column(10, 21, BASE_PARAMS["wt_signal_len"], "4h")

        for col in [
            "wt1",
            "wt2",
            "wt_delta",
            "h4_wt1",
            "h4_wt2",
            "h4_wt_delta",
            "h4_prev_wt_delta",
            wt1_col,
            wt2_col,
            h4_wt1_col,
            h4_wt2_col,
        ]:
            assert col in out.columns

        assert out["wt1"].dropna().shape[0] > 0
        assert out["h4_wt1"].dropna().shape[0] > 0

    def test_h4_values_use_last_closed_h4_candle(self):
        times = pd.date_range("2024-01-01", periods=160, freq="1h", tz="UTC")
        close = 1800.0 + np.linspace(0.0, 120.0, len(times)) + np.sin(np.linspace(0, 12, len(times))) * 25.0
        df = pd.DataFrame(
            {
                "time": times,
                "open": close,
                "high": close + 5.0,
                "low": close - 5.0,
                "close": close,
                "volume": 1000.0,
            }
        )

        out = prepare_indicators(df)
        h4_df = (
            df.set_index("time")
            .resample("4h")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
            .reset_index()
        )
        wt1_h4, wt2_h4 = compute_wave_trend(h4_df, 10, 21, BASE_PARAMS["wt_signal_len"])
        manual_wt1 = pd.Series(wt1_h4.to_numpy(), index=pd.to_datetime(h4_df["time"], utc=True), dtype="float64")
        manual_wt2 = pd.Series(wt2_h4.to_numpy(), index=pd.to_datetime(h4_df["time"], utc=True), dtype="float64")
        closed_start = manual_wt2.dropna().index[8]
        base_time = closed_start + pd.Timedelta("4h")

        window = out[(out["time"] >= base_time) & (out["time"] < base_time + pd.Timedelta("4h"))]
        assert len(window) == 4
        assert window["h4_wt1"].nunique() == 1
        assert window["h4_wt2"].nunique() == 1
        assert window.iloc[0]["h4_wt1"] == pytest.approx(manual_wt1.loc[closed_start])
        assert window.iloc[0]["h4_wt2"] == pytest.approx(manual_wt2.loc[closed_start])
        assert window.iloc[0]["h4_prev_wt1"] == pytest.approx(manual_wt1.shift(1).loc[closed_start])
        assert window.iloc[0]["h4_prev_wt2"] == pytest.approx(manual_wt2.shift(1).loc[closed_start])


class TestEntrySignals:
    def test_open_long_on_h1_green_dot_with_h4_filter(self):
        prev = _make_bar(wt1=-48.0, wt2=-42.0)
        bar = _make_bar(wt1=-34.0, wt2=-44.0)

        sig = generate_entry_signal(bar, prev, BASE_PARAMS, None)

        assert sig.action == "open_long"
        assert sig.reason == "WT_H1_GREEN_WINDOW_H4_SOFT_FILTER"

    def test_open_long_in_recent_h1_green_window_with_h4_soft_filter(self):
        prev = _make_bar(wt1=-36.0, wt2=-44.0)
        bar = _make_bar(
            wt1=-38.0,
            wt2=-42.0,
            h4_wt1=2.0,
            h4_wt2=-2.0,
            h4_prev_wt1=0.0,
            h4_prev_wt2=-2.0,
        )
        bar.bars_since_wt_green_dot = 2.0

        sig = generate_entry_signal(bar, prev, BASE_PARAMS, None)

        assert sig.action == "open_long"
        assert sig.reason == "WT_H1_GREEN_WINDOW_H4_SOFT_FILTER"

    def test_no_long_without_h4_convergence(self):
        prev = _make_bar(wt1=-48.0, wt2=-42.0)
        bar = _make_bar(
            wt1=-34.0,
            wt2=-44.0,
            h4_wt1=-34.0,
            h4_wt2=-22.0,
            h4_prev_wt1=-28.0,
            h4_prev_wt2=-22.0,
        )

        sig = generate_entry_signal(bar, prev, BASE_PARAMS, None)

        assert sig.action == "none"

    def test_no_long_when_h1_zone_is_too_shallow(self):
        prev = _make_bar(wt1=-20.0, wt2=-18.0)
        bar = _make_bar(wt1=-24.0, wt2=-28.0)

        sig = generate_entry_signal(bar, prev, BASE_PARAMS, None)

        assert sig.action == "none"

    def test_short_entry_is_disabled_by_default(self):
        prev = _make_bar(
            wt1=48.0,
            wt2=42.0,
            h4_wt1=66.0,
            h4_wt2=52.0,
            h4_prev_wt1=74.0,
            h4_prev_wt2=52.0,
        )
        bar = _make_bar(
            wt1=54.0,
            wt2=64.0,
            h4_wt1=58.0,
            h4_wt2=52.0,
            h4_prev_wt1=66.0,
            h4_prev_wt2=52.0,
        )

        sig = generate_entry_signal(bar, prev, BASE_PARAMS, None)

        assert sig.action == "none"

    def test_open_short_on_h1_red_dot_with_h4_filter_when_explicitly_enabled(self):
        prev = _make_bar(
            wt1=48.0,
            wt2=42.0,
            h4_wt1=66.0,
            h4_wt2=52.0,
            h4_prev_wt1=74.0,
            h4_prev_wt2=52.0,
        )
        bar = _make_bar(
            wt1=54.0,
            wt2=64.0,
            h4_wt1=58.0,
            h4_wt2=52.0,
            h4_prev_wt1=66.0,
            h4_prev_wt2=52.0,
        )

        sig = generate_entry_signal(bar, prev, SHORTS_ENABLED_PARAMS, None)

        assert sig.action == "open_short"
        assert sig.reason == "WT_H1_RED_DOT_H4_FILTER"

    def test_direction_filter_blocks_short_in_long_only_mode(self):
        prev = _make_bar(
            wt1=48.0,
            wt2=42.0,
            h4_wt1=66.0,
            h4_wt2=52.0,
            h4_prev_wt1=74.0,
            h4_prev_wt2=52.0,
        )
        bar = _make_bar(
            wt1=54.0,
            wt2=64.0,
            h4_wt1=58.0,
            h4_wt2=52.0,
            h4_prev_wt1=66.0,
            h4_prev_wt2=52.0,
        )

        sig = generate_entry_signal(bar, prev, LONG_ONLY_PARAMS, None)

        assert sig.action == "none"


class TestExitSignals:
    def test_long_tp1_closes_one_third_when_price_reaches_one_percent(self):
        bar = _make_bar(close=1800.0)
        bar.high = 1820.0
        pos = PositionState(side="long", entry_price=1800.0, entry_time=bar.time)

        sig = generate_partial_exit_signal(bar, BASE_PARAMS, pos)

        assert sig.action == "close_partial"
        assert sig.reason == "LONG_TP1_PARTIAL"
        assert sig.exit_price == pytest.approx(1818.0)
        assert sig.meta["close_fraction"] == pytest.approx(1.0 / 3.0)

    def test_long_tp1_only_once(self):
        bar = _make_bar(close=1800.0)
        bar.high = 1820.0
        pos = PositionState(side="long", entry_price=1800.0, entry_time=bar.time, tp1_taken=True)

        sig = generate_partial_exit_signal(bar, BASE_PARAMS, pos)

        assert sig.action == "none"

    def test_long_tp2_closes_one_third_after_tp1_when_price_reaches_two_percent(self):
        bar = _make_bar(close=1800.0)
        bar.high = 1840.0
        pos = PositionState(
            side="long",
            entry_price=1800.0,
            entry_time=bar.time,
            remaining_fraction=2.0 / 3.0,
            tp1_taken=True,
        )

        sig = generate_partial_exit_signal(bar, BASE_PARAMS, pos)

        assert sig.action == "close_partial"
        assert sig.reason == "LONG_TP2_PARTIAL"
        assert sig.exit_price == pytest.approx(1836.0)
        assert sig.meta["close_fraction"] == pytest.approx(1.0 / 3.0)

    def test_long_tp2_only_once(self):
        bar = _make_bar(close=1800.0)
        bar.high = 1840.0
        pos = PositionState(
            side="long",
            entry_price=1800.0,
            entry_time=bar.time,
            remaining_fraction=1.0 / 3.0,
            tp1_taken=True,
            tp2_taken=True,
        )

        sig = generate_partial_exit_signal(bar, BASE_PARAMS, pos)

        assert sig.action == "none"

    def test_long_emergency_stop_is_disabled_by_default(self):
        prev = _make_bar(wt1=-35.0, wt2=-40.0)
        bar = _make_bar(wt1=-34.0, wt2=-39.0)
        bar.low = 1760.0
        pos = PositionState(side="long", entry_price=1800.0, entry_time=bar.time)

        sig = generate_exit_signal(bar, prev, BASE_PARAMS, pos)

        assert sig.action == "none"

    def test_long_emergency_stop_closes_remaining_position_when_enabled(self):
        prev = _make_bar(wt1=-35.0, wt2=-40.0)
        bar = _make_bar(wt1=-34.0, wt2=-39.0)
        bar.low = 1760.0
        pos = PositionState(side="long", entry_price=1800.0, entry_time=bar.time)
        params = dict(
            BASE_PARAMS,
            wt_long_emergency_sl_enabled=True,
            wt_long_emergency_sl_capital_pct=0.02,
        )

        sig = generate_exit_signal(bar, prev, params, pos)

        assert sig.action == "close_force"
        assert sig.reason == "LONG_EMERGENCY_SL_CAPITAL"
        assert sig.exit_price == pytest.approx(1764.0)

    def test_long_emergency_stop_closes_only_remaining_fraction_after_tps(self):
        prev = _make_bar(wt1=-35.0, wt2=-40.0)
        bar = _make_bar(wt1=-34.0, wt2=-39.0)
        bar.low = 1680.0
        pos = PositionState(
            side="long",
            entry_price=1800.0,
            entry_time=bar.time,
            remaining_fraction=1.0 / 3.0,
            tp1_taken=True,
            tp2_taken=True,
        )
        params = dict(
            BASE_PARAMS,
            wt_long_emergency_sl_enabled=True,
            wt_long_emergency_sl_capital_pct=0.02,
        )

        sig = generate_exit_signal(bar, prev, params, pos)

        assert sig.action == "close_force"
        assert sig.reason == "LONG_EMERGENCY_SL_CAPITAL"
        assert sig.exit_price == pytest.approx(1692.0)
        assert sig.meta["emergency_sl_capital_pct"] == pytest.approx(0.02)
        assert sig.meta["emergency_sl_price_pct"] == pytest.approx(0.06)

    def test_long_emergency_stop_on_remaining_third_does_not_trigger_at_one_percent_price_drop(self):
        prev = _make_bar(wt1=-35.0, wt2=-40.0)
        bar = _make_bar(wt1=-34.0, wt2=-39.0)
        bar.low = 1780.0
        pos = PositionState(
            side="long",
            entry_price=1800.0,
            entry_time=bar.time,
            remaining_fraction=1.0 / 3.0,
            tp1_taken=True,
            tp2_taken=True,
        )
        params = dict(BASE_PARAMS, wt_long_tp1_breakeven_enabled=False)

        sig = generate_exit_signal(bar, prev, params, pos)

        assert sig.action == "none"

    def test_long_tp1_breakeven_does_not_trigger_on_tp1_bar(self):
        prev = _make_bar(wt1=-34.0, wt2=-39.0)
        bar = _make_bar(wt1=-33.0, wt2=-38.0)
        bar.low = 1798.0
        pos = PositionState(
            side="long",
            entry_price=1800.0,
            entry_time=bar.time,
            remaining_fraction=2.0 / 3.0,
            tp1_taken=True,
            tp1_protection_after_bars=1,
        )

        sig = generate_exit_signal(bar, prev, BASE_PARAMS, pos)

        assert sig.action == "none"

    def test_long_tp1_breakeven_closes_remaining_before_tp2(self):
        prev = _make_bar(wt1=-34.0, wt2=-39.0)
        bar = _make_bar(wt1=-33.0, wt2=-38.0)
        bar.low = 1798.0
        pos = PositionState(
            side="long",
            entry_price=1800.0,
            entry_time=bar.time,
            bars_in_position=1,
            remaining_fraction=2.0 / 3.0,
            tp1_taken=True,
            tp1_protection_after_bars=1,
        )

        sig = generate_exit_signal(bar, prev, BASE_PARAMS, pos)

        assert sig.action == "close_force"
        assert sig.reason == "LONG_TP1_BREAKEVEN_EXIT"
        assert sig.exit_price == pytest.approx(1800.0)
        assert sig.meta["remaining_fraction_before"] == pytest.approx(2.0 / 3.0)

    def test_long_tp2_breakeven_does_not_trigger_on_tp2_bar(self):
        prev = _make_bar(wt1=-34.0, wt2=-39.0)
        bar = _make_bar(wt1=-33.0, wt2=-38.0)
        bar.low = 1798.0
        pos = PositionState(
            side="long",
            entry_price=1800.0,
            entry_time=bar.time,
            bars_in_position=1,
            remaining_fraction=1.0 / 3.0,
            tp1_taken=True,
            tp2_taken=True,
            tp1_protection_after_bars=2,
        )

        sig = generate_exit_signal(bar, prev, BASE_PARAMS, pos)

        assert sig.action == "none"

    def test_long_tp2_breakeven_closes_remaining_after_tp2(self):
        prev = _make_bar(wt1=-34.0, wt2=-39.0)
        bar = _make_bar(wt1=-33.0, wt2=-38.0)
        bar.low = 1798.0
        pos = PositionState(
            side="long",
            entry_price=1800.0,
            entry_time=bar.time,
            bars_in_position=2,
            remaining_fraction=1.0 / 3.0,
            tp1_taken=True,
            tp2_taken=True,
            tp1_protection_after_bars=2,
        )

        sig = generate_exit_signal(bar, prev, BASE_PARAMS, pos)

        assert sig.action == "close_force"
        assert sig.reason == "LONG_TP2_BREAKEVEN_EXIT"
        assert sig.exit_price == pytest.approx(1800.0)
        assert sig.meta["remaining_fraction_before"] == pytest.approx(1.0 / 3.0)

    def test_long_exits_after_72h_without_tp1_when_under_entry_and_h1_not_rising(self):
        entry_time = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
        prev = _make_bar(time=entry_time + pd.Timedelta(hours=72), close=1790.0, wt1=-20.0, wt2=-25.0)
        bar = _make_bar(time=entry_time + pd.Timedelta(hours=73), close=1785.0, wt1=-22.0, wt2=-26.0)
        pos = PositionState(side="long", entry_price=1800.0, entry_time=entry_time)

        sig = generate_exit_signal(bar, prev, BASE_PARAMS, pos)

        assert sig.action == "close_force"
        assert sig.reason == "LONG_TP1_TIMEOUT_MOMENTUM_EXIT"
        assert sig.meta["hours_since_entry"] == pytest.approx(73.0)
        assert sig.meta["tp1_timeout_hours"] == pytest.approx(72.0)

    def test_long_timeout_before_tp1_does_not_exit_when_h1_lines_rise(self):
        entry_time = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
        prev = _make_bar(time=entry_time + pd.Timedelta(hours=72), close=1790.0, wt1=-24.0, wt2=-28.0)
        bar = _make_bar(time=entry_time + pd.Timedelta(hours=73), close=1785.0, wt1=-22.0, wt2=-26.0)
        pos = PositionState(side="long", entry_price=1800.0, entry_time=entry_time)

        sig = generate_exit_signal(bar, prev, BASE_PARAMS, pos)

        assert sig.action == "none"

    def test_short_tp1_closes_one_third_when_price_drops_one_percent(self):
        bar = _make_bar(close=1800.0)
        bar.low = 1780.0
        pos = PositionState(side="short", entry_price=1800.0, entry_time=bar.time)

        sig = generate_partial_exit_signal(bar, BASE_PARAMS, pos)

        assert sig.action == "close_partial"
        assert sig.reason == "SHORT_TP1_PARTIAL"
        assert sig.exit_price == pytest.approx(1782.0)
        assert sig.meta["close_fraction"] == pytest.approx(1.0 / 3.0)

    def test_h1_red_dot_can_close_long_without_opening_short(self):
        prev = _make_bar(
            wt1=48.0,
            wt2=42.0,
            h4_wt1=66.0,
            h4_wt2=52.0,
            h4_prev_wt1=74.0,
            h4_prev_wt2=52.0,
        )
        bar = _make_bar(
            wt1=54.0,
            wt2=64.0,
            h4_wt1=58.0,
            h4_wt2=52.0,
            h4_prev_wt1=66.0,
            h4_prev_wt2=52.0,
        )
        pos = PositionState(side="long", entry_price=1800.0, entry_time=bar.time)

        sig = generate_exit_signal(bar, prev, BASE_PARAMS, pos)

        assert sig.action == "close_force"
        assert sig.reason == "WT_HIGH_ZONE_H1_MOMENTUM_EXIT_LONG"

    def test_h1_red_dot_exit_still_works_in_long_only_mode(self):
        prev = _make_bar(
            wt1=48.0,
            wt2=42.0,
            h4_wt1=66.0,
            h4_wt2=52.0,
            h4_prev_wt1=74.0,
            h4_prev_wt2=52.0,
        )
        bar = _make_bar(
            wt1=54.0,
            wt2=64.0,
            h4_wt1=58.0,
            h4_wt2=52.0,
            h4_prev_wt1=66.0,
            h4_prev_wt2=52.0,
        )
        pos = PositionState(side="long", entry_price=1800.0, entry_time=bar.time)

        sig = generate_exit_signal(bar, prev, LONG_ONLY_PARAMS, pos)

        assert sig.action == "close_force"
        assert sig.reason == "WT_HIGH_ZONE_H1_MOMENTUM_EXIT_LONG"

    def test_h4_red_dot_does_not_close_long_without_h1_red_dot(self):
        prev = _make_bar(
            wt1=-5.0,
            wt2=-8.0,
            h4_wt1=54.0,
            h4_wt2=50.0,
            h4_prev_wt1=48.0,
            h4_prev_wt2=50.0,
        )
        bar = _make_bar(
            wt1=-4.0,
            wt2=-7.0,
            h4_wt1=48.0,
            h4_wt2=52.0,
            h4_prev_wt1=54.0,
            h4_prev_wt2=50.0,
        )
        pos = PositionState(side="long", entry_price=1800.0, entry_time=bar.time)

        sig = generate_exit_signal(bar, prev, BASE_PARAMS, pos)

        assert sig.action == "none"

    def test_first_h1_red_dot_closes_long_when_h4_lines_converge(self):
        prev = _make_bar(
            wt1=52.0,
            wt2=46.0,
            h4_wt1=58.0,
            h4_wt2=50.0,
            h4_prev_wt1=64.0,
            h4_prev_wt2=50.0,
        )
        bar = _make_bar(
            wt1=50.0,
            wt2=56.0,
            h4_wt1=56.0,
            h4_wt2=50.0,
            h4_prev_wt1=64.0,
            h4_prev_wt2=50.0,
        )
        params = dict(LONG_ONLY_PARAMS)
        pos = PositionState(side="long", entry_price=1800.0, entry_time=bar.time)

        sig = generate_exit_signal(bar, prev, params, pos)

        assert sig.action == "close_force"
        assert sig.reason == "WT_HIGH_ZONE_H1_MOMENTUM_EXIT_LONG"
        assert sig.meta["h1_red_close_count"] == 1

    def test_h1_red_dot_below_long_close_level_does_not_close_long(self):
        prev = _make_bar(
            wt1=-48.0,
            wt2=-52.0,
            h4_wt1=58.0,
            h4_wt2=50.0,
            h4_prev_wt1=64.0,
            h4_prev_wt2=50.0,
        )
        bar = _make_bar(
            wt1=-54.0,
            wt2=-50.0,
            h4_wt1=56.0,
            h4_wt2=50.0,
            h4_prev_wt1=64.0,
            h4_prev_wt2=50.0,
        )
        params = dict(LONG_ONLY_PARAMS, wt_long_close_min_level=0.0, wt_h4_long_close_min=0.0)
        pos = PositionState(side="long", entry_price=1800.0, entry_time=bar.time)

        sig = generate_exit_signal(bar, prev, params, pos)

        assert sig.action == "none"
        assert pos.h1_red_close_count == 1

    def test_h1_red_dot_below_h4_close_level_does_not_close_long(self):
        prev = _make_bar(
            wt1=52.0,
            wt2=46.0,
            h4_wt1=-10.0,
            h4_wt2=-18.0,
            h4_prev_wt1=-6.0,
            h4_prev_wt2=-20.0,
        )
        bar = _make_bar(
            wt1=50.0,
            wt2=56.0,
            h4_wt1=-12.0,
            h4_wt2=-18.0,
            h4_prev_wt1=-6.0,
            h4_prev_wt2=-20.0,
        )
        params = dict(LONG_ONLY_PARAMS, wt_long_close_min_level=0.0, wt_h4_long_close_min=0.0)
        pos = PositionState(side="long", entry_price=1800.0, entry_time=bar.time)

        sig = generate_exit_signal(bar, prev, params, pos)

        assert sig.action == "none"
        assert pos.h1_red_close_count == 1

    def test_h1_weakening_closes_long_even_when_h4_gap_widens(self):
        prev = _make_bar(
            wt1=52.0,
            wt2=46.0,
            h4_wt1=44.0,
            h4_wt2=40.0,
            h4_prev_wt1=42.0,
            h4_prev_wt2=40.0,
        )
        bar = _make_bar(
            wt1=50.0,
            wt2=56.0,
            h4_wt1=48.0,
            h4_wt2=40.0,
            h4_prev_wt1=42.0,
            h4_prev_wt2=40.0,
        )
        params = dict(LONG_ONLY_PARAMS)
        pos = PositionState(side="long", entry_price=1800.0, entry_time=bar.time)

        sig = generate_exit_signal(bar, prev, params, pos)

        assert sig.action == "close_force"
        assert sig.reason == "WT_HIGH_ZONE_H1_MOMENTUM_EXIT_LONG"
        assert pos.h1_red_close_count == 1

    def test_h4_gap_widening_does_not_close_long(self):
        prev = _make_bar(
            wt1=-35.0,
            wt2=-42.0,
            h4_wt1=-28.0,
            h4_wt2=-22.0,
            h4_prev_wt1=-24.0,
            h4_prev_wt2=-22.0,
        )
        bar = _make_bar(
            wt1=-32.0,
            wt2=-40.0,
            h4_wt1=-34.0,
            h4_wt2=-22.0,
            h4_prev_wt1=-28.0,
            h4_prev_wt2=-22.0,
        )
        pos = PositionState(side="long", entry_price=1800.0, entry_time=bar.time)

        sig = generate_exit_signal(bar, prev, BASE_PARAMS, pos)

        assert sig.action == "none"

    def test_h1_positive_zone_without_exit_signal_does_not_close_long(self):
        prev = _make_bar(
            wt1=5.0,
            wt2=2.0,
            h4_wt1=-28.0,
            h4_wt2=-22.0,
            h4_prev_wt1=-24.0,
            h4_prev_wt2=-22.0,
        )
        bar = _make_bar(
            wt1=8.0,
            wt2=3.0,
            h4_wt1=-34.0,
            h4_wt2=-22.0,
            h4_prev_wt1=-28.0,
            h4_prev_wt2=-22.0,
        )
        pos = PositionState(side="long", entry_price=1800.0, entry_time=bar.time)

        sig = generate_exit_signal(bar, prev, BASE_PARAMS, pos)

        assert sig.action == "none"

    def test_h4_gap_widening_does_not_close_short(self):
        prev = _make_bar(
            wt1=64.0,
            wt2=54.0,
            h4_wt1=58.0,
            h4_wt2=52.0,
            h4_prev_wt1=54.0,
            h4_prev_wt2=52.0,
        )
        bar = _make_bar(
            wt1=60.0,
            wt2=55.0,
            h4_wt1=66.0,
            h4_wt2=52.0,
            h4_prev_wt1=58.0,
            h4_prev_wt2=52.0,
        )
        pos = PositionState(side="short", entry_price=1800.0, entry_time=bar.time)

        sig = generate_exit_signal(bar, prev, BASE_PARAMS, pos)

        assert sig.action == "none"

    def test_h4_green_dot_closes_short(self):
        prev = _make_bar(
            wt1=5.0,
            wt2=8.0,
            h4_wt1=48.0,
            h4_wt2=52.0,
            h4_prev_wt1=54.0,
            h4_prev_wt2=52.0,
        )
        bar = _make_bar(
            wt1=4.0,
            wt2=7.0,
            h4_wt1=56.0,
            h4_wt2=50.0,
            h4_prev_wt1=48.0,
            h4_prev_wt2=52.0,
        )
        pos = PositionState(side="short", entry_price=1800.0, entry_time=bar.time)

        sig = generate_exit_signal(bar, prev, BASE_PARAMS, pos)

        assert sig.action == "close_force"
        assert sig.reason == "WT_H4_GREEN_DOT_EXIT_SHORT"

    def test_third_h1_green_dot_closes_short_when_h4_lines_converge(self):
        prev = _make_bar(
            wt1=-56.0,
            wt2=-52.0,
            h4_wt1=-58.0,
            h4_wt2=-50.0,
            h4_prev_wt1=-64.0,
            h4_prev_wt2=-50.0,
        )
        bar = _make_bar(
            wt1=-46.0,
            wt2=-50.0,
            h4_wt1=-56.0,
            h4_wt2=-50.0,
            h4_prev_wt1=-64.0,
            h4_prev_wt2=-50.0,
        )
        pos = PositionState(side="short", entry_price=1800.0, entry_time=bar.time, h1_green_close_count=2)

        sig = generate_exit_signal(bar, prev, BASE_PARAMS, pos)

        assert sig.action == "close_force"
        assert sig.reason == "WT_H1_THIRD_GREEN_DOT_H4_CONVERGENCE_EXIT_SHORT"
        assert sig.meta["h1_green_close_count"] == 3


class TestBarHelpers:
    def test_bar_from_row_reads_h1_h4_and_ema_columns(self):
        wt1_col, wt2_col = wt_columns(10, 21, BASE_PARAMS["wt_signal_len"])
        h4_wt1_col = htf_wt1_column(10, 21, "4h")
        h4_wt2_col = htf_wt2_column(10, 21, BASE_PARAMS["wt_signal_len"], "4h")
        h4_prev_wt1_col = htf_prev_wt1_column(10, 21, "4h")
        h4_prev_wt2_col = htf_prev_wt2_column(10, 21, BASE_PARAMS["wt_signal_len"], "4h")
        row = pd.Series(
            {
                "time": pd.Timestamp("2024-01-01", tz="UTC"),
                "open": 100.0,
                "high": 102.0,
                "low": 98.0,
                "close": 100.0,
                wt1_col: -34.0,
                wt2_col: -44.0,
                h4_wt1_col: -28.0,
                h4_wt2_col: -22.0,
                h4_prev_wt1_col: -36.0,
                h4_prev_wt2_col: -26.0,
                "h4_prev_wt1": -34.0,
                "h4_prev_wt2": -22.0,
                "h4_prev_wt_delta": -12.0,
                "ema20": 90.0,
                "ema_10": 95.0,
                "htf_ema200": 85.0,
                "atr": 4.0,
            }
        )

        bar = bar_from_row(row, dict(BASE_PARAMS, wt_ema_filter_len=10))

        assert bar.wt1 == pytest.approx(-34.0)
        assert bar.wt2 == pytest.approx(-44.0)
        assert bar.h4_wt1 == pytest.approx(-28.0)
        assert bar.h4_wt2 == pytest.approx(-22.0)
        assert bar.h4_prev_wt1 == pytest.approx(-36.0)
        assert bar.h4_prev_wt2 == pytest.approx(-26.0)
        assert bar.h4_prev_wt_delta == pytest.approx(-10.0)
        assert bar.ema20 == pytest.approx(95.0)
        assert bar.ema_filter_len == 10
        assert bar.htf_ema200 == pytest.approx(85.0)
        assert bar.atr == pytest.approx(4.0)

    def test_load_klines_string(self):
        path = Path(".test_load_klines_string.csv")
        try:
            times_str = [f"2024-01-0{i + 1} 00:00:00+00:00" for i in range(5)]
            csv_path = _write_csv(path, "open_time", times_str)
            df = load_klines(str(csv_path))
            assert len(df) == 5
        finally:
            path.unlink(missing_ok=True)


class TestSlippage:
    def test_zero_slippage_returns_price(self):
        assert apply_slippage(1800.0, "long", "open", 0.0, 0.0) == 1800.0

    def test_long_open_increases_price(self):
        assert apply_slippage(1800.0, "long", "open", 5.0, 2.0) > 1800.0

    def test_short_open_decreases_price(self):
        assert apply_slippage(1800.0, "short", "open", 5.0, 2.0) < 1800.0


class TestBacktestAccounting:
    def test_partial_long_tp1_keeps_two_thirds_until_signal_exit(self):
        df = _signal_df()
        params = dict(LONG_ONLY_PARAMS, fee_rate=0.0, slippage_bps=0.0, spread_bps=0.0)
        strat = Bee4Strategy(params, fee_rate=0.0)

        trades, equity, final_cap = strat.run(df, 9_000.0)

        assert list(trades["reason"])[:2] == [
            "LONG_TP1_PARTIAL",
            "WT_HIGH_ZONE_H1_MOMENTUM_EXIT_LONG",
        ]
        assert trades.iloc[0]["close_fraction"] == pytest.approx(1.0 / 3.0)
        assert trades.iloc[0]["remaining_fraction_after"] == pytest.approx(2.0 / 3.0)
        assert trades.iloc[0]["position_notional"] == pytest.approx(3_000.0)
        assert trades.iloc[1]["position_notional"] == pytest.approx(6_000.0)
        assert trades.iloc[0]["logical_trade_no"] == trades.iloc[1]["logical_trade_no"]
        assert list(trades["trade_event"])[:2] == ["TP1", "EXIT"]
        assert trades.iloc[0]["time_to_tp1_hours"] == pytest.approx(1.0)
        assert np.isnan(trades.iloc[1]["time_to_tp1_hours"])
        assert trades.iloc[1]["holding_hours"] == pytest.approx(2.0)
        assert len(equity) >= 3
        assert final_cap > 9_000.0

    def test_partial_long_tp1_and_tp2_can_fill_on_same_bar_before_signal_exit(self):
        df = _signal_df()
        df.loc[2, "high"] = 1850.0
        params = dict(LONG_ONLY_PARAMS, fee_rate=0.0, slippage_bps=0.0, spread_bps=0.0)
        strat = Bee4Strategy(params, fee_rate=0.0)

        trades, _equity, final_cap = strat.run(df, 9_000.0)

        assert list(trades["reason"])[:3] == [
            "LONG_TP1_PARTIAL",
            "LONG_TP2_PARTIAL",
            "WT_HIGH_ZONE_H1_MOMENTUM_EXIT_LONG",
        ]
        assert list(trades["trade_event"])[:3] == ["TP1", "TP2", "EXIT"]
        assert list(trades["close_fraction"])[:3] == pytest.approx([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        assert list(trades["position_notional"])[:3] == pytest.approx([3_000.0, 3_000.0, 3_000.0])
        assert trades.iloc[0]["time_to_tp1_hours"] == pytest.approx(1.0)
        assert np.isnan(trades.iloc[1]["time_to_tp1_hours"])
        assert trades["logical_trade_no"].nunique() == 1
        assert final_cap > 9_000.0

    def test_tp1_breakeven_closes_remaining_when_price_returns_to_entry(self):
        df = _signal_df()
        df.loc[3, "low"] = 1805.0
        params = dict(
            LONG_ONLY_PARAMS,
            fee_rate=0.0,
            slippage_bps=0.0,
            spread_bps=0.0,
            wt_long_tp2_enabled=False,
            wt_long_close_min_level=999.0,
            wt_h4_long_close_min=999.0,
        )
        strat = Bee4Strategy(params, fee_rate=0.0)

        trades, _equity, final_cap = strat.run(df, 9_000.0)

        assert list(trades["reason"]) == [
            "LONG_TP1_PARTIAL",
            "LONG_TP1_BREAKEVEN_EXIT",
        ]
        assert list(trades["trade_event"]) == ["TP1", "EXIT"]
        assert trades.iloc[1]["close_fraction"] == pytest.approx(2.0 / 3.0)
        assert trades.iloc[1]["exit_price"] == pytest.approx(trades.iloc[1]["entry_price"])
        assert final_cap > 9_000.0

    def test_tp2_breakeven_closes_remaining_when_price_returns_to_entry(self):
        df = _signal_df()
        df.loc[2, "high"] = 1850.0
        df.loc[3, "low"] = 1805.0
        params = dict(
            LONG_ONLY_PARAMS,
            fee_rate=0.0,
            slippage_bps=0.0,
            spread_bps=0.0,
            wt_long_close_min_level=999.0,
            wt_h4_long_close_min=999.0,
        )
        strat = Bee4Strategy(params, fee_rate=0.0)

        trades, _equity, final_cap = strat.run(df, 9_000.0)

        assert list(trades["reason"]) == [
            "LONG_TP1_PARTIAL",
            "LONG_TP2_PARTIAL",
            "LONG_TP2_BREAKEVEN_EXIT",
        ]
        assert list(trades["trade_event"]) == ["TP1", "TP2", "EXIT"]
        assert trades.iloc[2]["close_fraction"] == pytest.approx(1.0 / 3.0)
        assert trades.iloc[2]["exit_price"] == pytest.approx(trades.iloc[2]["entry_price"])
        assert final_cap > 9_000.0

    def test_emergency_stop_has_priority_over_tp_on_same_bar(self):
        df = _signal_df()
        df.loc[2, "high"] = 1850.0
        df.loc[2, "low"] = 1760.0
        params = dict(
            LONG_ONLY_PARAMS,
            fee_rate=0.0,
            slippage_bps=0.0,
            spread_bps=0.0,
            wt_long_emergency_sl_enabled=True,
            wt_long_emergency_sl_capital_pct=0.02,
        )
        strat = Bee4Strategy(params, fee_rate=0.0)

        trades, _equity, final_cap = strat.run(df, 9_000.0)

        assert trades.iloc[0]["reason"] == "LONG_EMERGENCY_SL_CAPITAL"
        assert trades.iloc[0]["close_fraction"] == pytest.approx(1.0)
        assert len(trades) == 1
        assert final_cap < 9_000.0

    def test_open_position_at_data_end_is_not_force_closed(self):
        df = _signal_df()
        params = dict(
            LONG_ONLY_PARAMS,
            fee_rate=0.0,
            slippage_bps=0.0,
            spread_bps=0.0,
            wt_long_tp1_enabled=False,
            wt_long_tp2_enabled=False,
            wt_long_close_min_level=999.0,
            wt_h4_long_close_min=999.0,
            wt_long_emergency_sl_enabled=False,
        )
        strat = Bee4Strategy(params, fee_rate=0.0)

        trades, equity, final_cap = strat.run(df, 9_000.0)

        assert trades.empty
        assert final_cap == pytest.approx(9_000.0)
        assert equity.iloc[-1]["time"] == df.iloc[-1]["time"]

    def test_open_position_can_continue_with_new_window_params(self):
        df = _signal_df()
        first_window = df.iloc[:2].copy()
        next_window = df.iloc[2:3].copy()
        next_window.loc[next_window.index[0], "high"] = 1820.0

        params_entry_window = dict(
            LONG_ONLY_PARAMS,
            fee_rate=0.0,
            slippage_bps=0.0,
            spread_bps=0.0,
            wt_long_tp1_pct=0.01,
            wt_long_tp2_enabled=False,
            wt_long_close_min_level=999.0,
            wt_h4_long_close_min=999.0,
            wt_long_emergency_sl_enabled=False,
        )
        strat_entry = Bee4Strategy(params_entry_window, fee_rate=0.0)

        trades1, _equity1, cap1, position, cap_at_open, next_trade_id = strat_entry.run(
            first_window,
            9_000.0,
            keep_open_position=True,
            return_state=True,
        )

        assert trades1.empty
        assert position is not None
        assert position.trade_id == 1
        assert cap_at_open == pytest.approx(9_000.0)
        assert next_trade_id == 2

        params_next_window = dict(params_entry_window, wt_long_tp1_pct=0.005)
        strat_next = Bee4Strategy(params_next_window, fee_rate=0.0)

        trades2, _equity2, cap2, position2, cap_at_open2, next_trade_id2 = strat_next.run(
            next_window,
            cap1,
            initial_position=position,
            initial_capital_at_open=cap_at_open,
            initial_next_trade_id=next_trade_id,
            previous_row=first_window.iloc[-1],
            keep_open_position=True,
            return_state=True,
        )

        assert len(trades2) == 1
        assert trades2.iloc[0]["reason"] == "LONG_TP1_PARTIAL"
        assert trades2.iloc[0]["logical_trade_no"] == 1
        assert trades2.iloc[0]["exit_price"] == pytest.approx(1810.0 * 1.005)
        assert position2 is not None
        assert cap_at_open2 == pytest.approx(9_000.0)
        assert next_trade_id2 == 2
        assert cap2 > cap1


class TestWFOHelpers:
    def test_wfo_bars_1h(self):
        ob, lb = wfo_bars("1h", 90, 14)
        assert ob == 2160
        assert lb == 336

    def test_interval_to_ms(self):
        assert interval_to_ms("1h") == 3_600_000

    def test_update_csv_cache_backfills_requested_start(self, monkeypatch):
        path = Path(".test_cache_backfill.csv")
        first_ms = int(pd.Timestamp("2025-01-01 00:00", tz="UTC").timestamp() * 1000)
        future_ms = int(pd.Timestamp("2099-01-01 00:00", tz="UTC").timestamp() * 1000)
        requested_ms = int(pd.Timestamp("2023-01-01 00:00", tz="UTC").timestamp() * 1000)
        try:
            _write_csv(path, "open_time", [first_ms, future_ms])
            calls = []

            def fake_fetch(symbol, interval, start_time_ms=None, end_time_ms=None, market="spot", verbose=True):
                calls.append((start_time_ms, end_time_ms))
                return pd.DataFrame(
                    {
                        "open_time": [start_time_ms, end_time_ms],
                        "open": [100.0, 101.0],
                        "high": [101.0, 102.0],
                        "low": [99.0, 100.0],
                        "close": [100.5, 101.5],
                        "volume": [1.0, 1.0],
                    }
                )

            monkeypatch.setattr(bee4_binance, "fetch_klines_binance", fake_fetch)
            out = bee4_binance.update_csv_cache(
                str(path),
                symbol="ETHUSDT",
                interval="1h",
                start_date="2023-01-01",
                verbose=False,
            )

            assert calls == [(requested_ms, first_ms - interval_to_ms("1h"))]
            assert int(out["open_time"].min()) == requested_ms
            assert int(out["open_time"].iloc[0]) == requested_ms
        finally:
            path.unlink(missing_ok=True)

    def test_save_and_load_params_json(self):
        from bee4_params import save_params

        path = Path(".test_params.json")
        try:
            save_params(dict(BASE_PARAMS), str(path))
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded["wt_channel_len"] == BASE_PARAMS["wt_channel_len"]
            assert loaded["wt_h4_long_filter_max"] == BASE_PARAMS["wt_h4_long_filter_max"]
        finally:
            path.unlink(missing_ok=True)

    def test_get_latest_best_params_includes_h4_filters_and_direction(self):
        windows_df = pd.DataFrame(
            {
                "best_wt_channel_len": [10, 10, 12],
                "best_wt_avg_len": [21, 21, 28],
                "best_wt_signal_len": [3, 3, 4],
                "best_wt_min_signal_level": [0.0, 0.0, 0.0],
                "best_wt_reentry_window_bars": [0, 0, 0],
                "best_wt_use_ema_filter": [False, False, False],
                "best_wt_use_htf_filter": [False, False, False],
                "best_wt_ema_filter_len": [20, 20, 20],
                "best_wt_long_entry_max_above_zero": [-30.0, -30.0, -40.0],
                "best_wt_long_close_min_level": [0.0, 10.0, 10.0],
                "best_wt_short_entry_min_below_zero": [30.0, 30.0, 40.0],
                "best_wt_h4_long_filter_max": [-20.0, -20.0, -30.0],
                "best_wt_h4_long_close_min": [0.0, 20.0, 20.0],
                "best_wt_h4_short_filter_min": [50.0, 50.0, 60.0],
                "best_wt_long_emergency_sl_capital_pct": [0.0, 0.05, 0.05],
                "best_wt_long_tp1_pct": [0.008, 0.01, 0.01],
                "best_wt_long_tp2_pct": [0.015, 0.02, 0.02],
                "best_wt_long_tp1_fraction": [0.25, 1.0 / 3.0, 1.0 / 3.0],
                "best_wt_long_tp2_fraction": [0.25, 0.5, 0.5],
                "best_wt_long_tp1_timeout_hours": [24.0, 48.0, 48.0],
                "allow_longs": [True, True, True],
                "allow_shorts": [False, False, False],
                "n_trades_live": [2, 1, 1],
            }
        )

        best = get_latest_best_params(windows_df)

        assert best["trade_direction"] == "long"
        assert best["allow_longs"] is True
        assert best["allow_shorts"] is False
        assert best["short_trading_enabled"] is False
        assert best["wt_channel_len"] == 10
        assert best["wt_avg_len"] == 21
        assert best["wt_signal_len"] == 3
        assert best["wt_long_entry_max_above_zero"] == pytest.approx(-30.0)
        assert best["wt_long_close_min_level"] == pytest.approx(10.0)
        assert best["wt_long_exit_min_level"] == pytest.approx(10.0)
        assert best["wt_short_entry_min_below_zero"] == pytest.approx(30.0)
        assert best["wt_h4_long_filter_max"] == pytest.approx(-20.0)
        assert best["wt_h4_long_close_min"] == pytest.approx(20.0)
        assert best["wt_h4_short_filter_min"] == pytest.approx(50.0)
        assert best["wt_long_emergency_sl_enabled"] is True
        assert best["wt_long_emergency_sl_capital_pct"] == pytest.approx(0.05)
        assert best["wt_long_tp1_pct"] == pytest.approx(0.01)
        assert best["wt_long_tp2_pct"] == pytest.approx(0.02)
        assert best["wt_long_tp1_fraction"] == pytest.approx(1.0 / 3.0)
        assert best["wt_long_tp2_fraction"] == pytest.approx(0.5)
        assert best["wt_long_tp1_timeout_hours"] == pytest.approx(48.0)

    def test_growth_scoring_prefers_higher_return_with_acceptable_risk(self):
        low_return = pd.DataFrame({"pnl": [100.0, -40.0, 50.0, -30.0, 20.0]})
        higher_return = pd.DataFrame({"pnl": [500.0, -100.0, 200.0, -50.0, 100.0]})

        low_score = score_params(low_return, 10_100.0, 10_000.0, mode="growth")
        higher_score = score_params(higher_return, 10_650.0, 10_000.0, mode="growth")

        assert higher_score > low_score
        assert np.isfinite(higher_score)

    def test_wfo_accepts_bee4_2_grid_overrides(self):
        times = pd.date_range("2024-01-01", periods=160, freq="1h", tz="UTC")
        wave = np.sin(np.linspace(0, 18, len(times)))
        close = 1800.0 + wave * 90.0 + np.linspace(0.0, 60.0, len(times))
        df = pd.DataFrame(
            {
                "time": times,
                "open": close,
                "high": close + 6.0,
                "low": close - 6.0,
                "close": close,
                "volume": 1000.0,
            }
        )
        df = prepare_indicators(df)

        grid_overrides = {
            "wt_channel_len": [10],
            "wt_avg_len": [21],
            "wt_signal_len": [3],
            "wt_min_signal_level": [0.0],
            "wt_reentry_window_bars": [0],
            "wt_use_ema_filter": [False],
            "wt_use_htf_filter": [False],
            "wt_ema_filter_len": [20],
            "wt_long_entry_max_above_zero": [-30.0],
            "wt_long_close_min_level": [0.0],
            "wt_short_entry_min_below_zero": [30.0],
            "wt_h4_long_filter_max": [-20.0],
            "wt_h4_long_close_min": [0.0],
            "wt_h4_short_filter_min": [50.0],
            "wt_long_emergency_sl_capital_pct": [0.05],
            "wt_long_tp1_pct": [0.008],
            "wt_long_tp2_pct": [0.015],
            "wt_long_tp1_fraction": [0.25],
            "wt_long_tp2_fraction": [0.5],
            "wt_long_tp1_timeout_hours": [48.0],
        }

        _trades, _equity, windows_df, _final_cap, stopped = walk_forward_optimization(
            df,
            interval="1h",
            verbose=False,
            fee_rate=0.0,
            opt_days=2,
            live_days=1,
            initial_capital=10_000.0,
            base_params=dict(BASE_PARAMS, fee_rate=0.0, slippage_bps=0.0, spread_bps=0.0),
            grid_overrides=grid_overrides,
        )

        assert stopped is False
        assert not windows_df.empty
        assert set(windows_df["best_wt_channel_len"]) == {10}
        assert set(windows_df["best_wt_avg_len"]) == {21}
        assert set(windows_df["best_wt_signal_len"]) == {3}
        assert set(windows_df["best_wt_long_entry_max_above_zero"]) == {-30.0}
        assert set(windows_df["best_wt_long_close_min_level"]) == {0.0}
        assert set(windows_df["best_wt_short_entry_min_below_zero"]) == {30.0}
        assert set(windows_df["best_wt_h4_long_filter_max"]) == {-20.0}
        assert set(windows_df["best_wt_h4_long_close_min"]) == {0.0}
        assert set(windows_df["best_wt_h4_short_filter_min"]) == {50.0}
        assert set(windows_df["best_wt_long_emergency_sl_capital_pct"]) == {0.05}
        assert set(windows_df["best_wt_long_emergency_sl_enabled"]) == {True}
        assert set(windows_df["best_wt_long_tp1_pct"]) == {0.008}
        assert set(windows_df["best_wt_long_tp2_pct"]) == {0.015}
        assert set(windows_df["best_wt_long_tp1_fraction"]) == {0.25}
        assert set(windows_df["best_wt_long_tp2_fraction"]) == {0.5}
        assert set(windows_df["best_wt_long_tp1_timeout_hours"]) == {48.0}

    def test_wfo_carries_open_position_to_next_live_window(self):
        times = pd.date_range("2024-01-01", periods=72, freq="1h", tz="UTC")
        close = np.full(len(times), 100.0)
        high = close + 0.2
        low = close - 0.2

        wt1 = np.full(len(times), -60.0)
        wt2 = np.full(len(times), -60.0)
        wt1[46], wt2[46] = -50.0, -40.0
        wt1[47], wt2[47] = -40.0, -50.0
        wt1[48], wt2[48] = -35.0, -45.0
        close[48] = 101.1
        high[48] = 101.2
        low[48] = 100.9

        df = pd.DataFrame(
            {
                "time": times,
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000.0,
                "wt1": wt1,
                "wt2": wt2,
                "h4_wt1": np.full(len(times), -50.0),
                "h4_wt2": np.full(len(times), -60.0),
                "h4_prev_wt1": np.full(len(times), -60.0),
                "h4_prev_wt2": np.full(len(times), -50.0),
                "ema20": close,
                "ema_20": close,
                "htf_ema200": close,
                "atr": np.full(len(times), 10.0),
                "wt_green_dot": [False] * len(times),
                "wt_red_dot": [False] * len(times),
                "bars_since_wt_green_dot": [np.nan] * len(times),
                "bars_since_wt_red_dot": [np.nan] * len(times),
            }
        )
        df["wt_green_dot"] = (df["wt1"].shift(1) <= df["wt2"].shift(1)) & (df["wt1"] > df["wt2"])
        df["bars_since_wt_green_dot"] = np.where(df["wt_green_dot"], 0.0, np.nan)
        df["h4_wt_delta"] = df["h4_wt1"] - df["h4_wt2"]
        df["h4_prev_wt_delta"] = df["h4_prev_wt1"] - df["h4_prev_wt2"]

        grid_overrides = {
            "wt_channel_len": [10],
            "wt_avg_len": [21],
            "wt_signal_len": [3],
            "wt_min_signal_level": [0.0],
            "wt_reentry_window_bars": [0],
            "wt_use_ema_filter": [False],
            "wt_use_htf_filter": [False],
            "wt_ema_filter_len": [20],
            "wt_long_entry_max_above_zero": [-30.0],
            "wt_long_close_min_level": [999.0],
            "wt_short_entry_min_below_zero": [30.0],
            "wt_h4_long_filter_max": [-20.0],
            "wt_h4_long_close_min": [999.0],
            "wt_h4_short_filter_min": [50.0],
            "wt_long_emergency_sl_capital_pct": [0.0],
            "wt_long_tp1_pct": [0.01],
            "wt_long_tp2_pct": [0.02],
            "wt_long_tp1_fraction": [1.0],
            "wt_long_tp2_fraction": [1.0 / 3.0],
            "wt_long_tp1_timeout_hours": [72.0],
        }

        trades, _equity, windows_df, final_cap, stopped = walk_forward_optimization(
            df,
            interval="1h",
            verbose=False,
            fee_rate=0.0,
            opt_days=1,
            live_days=1,
            initial_capital=10_000.0,
            base_params=dict(
                BASE_PARAMS,
                fee_rate=0.0,
                slippage_bps=0.0,
                spread_bps=0.0,
                wt_long_tp1_fraction=1.0,
                wt_long_tp2_enabled=False,
                wt_long_emergency_sl_enabled=False,
            ),
            grid_overrides=grid_overrides,
        )

        assert stopped is False
        assert len(windows_df) == 2
        assert bool(windows_df.iloc[0]["open_position_carried"]) is True
        assert bool(windows_df.iloc[1]["open_position_carried"]) is False
        assert len(trades) == 1
        assert trades.iloc[0]["entry_time"] == times[47]
        assert trades.iloc[0]["exit_time"] == times[48]
        assert trades.iloc[0]["window_id"] == 1
        assert trades.iloc[0]["entry_window_id"] == 0
        assert trades.iloc[0]["exit_window_id"] == 1
        assert trades.iloc[0]["entry_params_wt_long_entry_max_above_zero"] == pytest.approx(-30.0)
        assert trades.iloc[0]["exit_params_wt_long_entry_max_above_zero"] == pytest.approx(-30.0)
        assert trades.iloc[0]["entry_params_wt_long_tp1_timeout_hours"] == pytest.approx(72.0)
        assert trades.iloc[0]["exit_params_wt_long_tp1_timeout_hours"] == pytest.approx(72.0)
        assert trades.iloc[0]["trade_event"] == "TP1"
        assert final_cap > 10_000.0

    def test_wfo_can_stop_during_first_window(self):
        times = pd.date_range("2024-01-01", periods=160, freq="1h", tz="UTC")
        wave = np.sin(np.linspace(0, 18, len(times)))
        close = 1800.0 + wave * 90.0 + np.linspace(0.0, 60.0, len(times))
        df = pd.DataFrame(
            {
                "time": times,
                "open": close,
                "high": close + 6.0,
                "low": close - 6.0,
                "close": close,
                "volume": 1000.0,
            }
        )
        df = prepare_indicators(df)

        checks = {"count": 0}

        def should_stop():
            checks["count"] += 1
            return checks["count"] >= 5

        _trades, _equity, windows_df, _final_cap, stopped = walk_forward_optimization(
            df,
            interval="1h",
            verbose=False,
            fee_rate=0.0,
            opt_days=2,
            live_days=1,
            initial_capital=10_000.0,
            base_params=dict(BASE_PARAMS, fee_rate=0.0, slippage_bps=0.0, spread_bps=0.0),
            should_stop=should_stop,
        )

        assert stopped is True
        assert windows_df.empty


class TestClosedCandleOnly:
    def test_runner_uses_last_closed_candle(self):
        import bee4_live_runner as runner_module

        source = inspect.getsource(runner_module.main_loop)
        assert "df.iloc[-2]" in source
        assert "df.iloc[-3]" in source
        assert "len(df) < 3" in source


class TestBacktestRunnerParity:
    def _simulate_runner(self, df: pd.DataFrame, params: dict) -> list[dict]:
        import copy
        import bee4_live_runner as runner_module
        from bee4_live_runner import process_bar

        state = {
            "position": None,
            "entry_price": None,
            "entry_time": None,
            "bars_in_position": 0,
            "capital": 10_000.0,
            "capital_at_open": None,
            "stop_price": None,
            "entry_atr": None,
            "last_bar_time": None,
            "daily_loss_usd": 0.0,
            "daily_date": None,
            "paused": False,
            "mode": "paper",
        }

        collected = []
        original_log_trade = runner_module.log_trade
        original_save_state = runner_module.save_state
        runner_module.log_trade = lambda t: collected.append(copy.deepcopy(t))
        runner_module.save_state = lambda s: None

        try:
            for i in range(1, len(df)):
                bar = bar_from_row(df.iloc[i], params)
                prev = bar_from_row(df.iloc[i - 1], params)
                process_bar(bar, prev, params, state, mode="paper")
        finally:
            runner_module.log_trade = original_log_trade
            runner_module.save_state = original_save_state

        return collected

    def test_same_closed_trades_as_runner(self):
        df = _signal_df()
        strat = Bee4Strategy(LONG_ONLY_PARAMS)
        bt_trades, _equity, _final_cap = strat.run(df, 10_000.0)
        bt_real = bt_trades[bt_trades["reason"] != "FORCE_EXIT_END"].reset_index(drop=True)

        runner_trades = self._simulate_runner(df, LONG_ONLY_PARAMS)

        assert len(bt_real) == len(runner_trades)
        assert list(bt_real["side"]) == [t["side"] for t in runner_trades]
        assert list(bt_real["reason"]) == [t["reason"] for t in runner_trades]


class TestDashboardWFOGridPersistence:
    def test_saved_wfo_grid_restores_checkbox_values_in_dashboard_order(self):
        from bee4_dashboard import WFO_GRID_CONTROL_ORDER, _wfo_grid_values_from_result

        result = {
            "wfo_grid_overrides": {
                "wt_channel_len": [10],
                "wt_avg_len": [21],
                "wt_signal_len": [3],
                "wt_min_signal_level": [0.0],
                "wt_reentry_window_bars": [0, 3],
                "wt_use_ema_filter": [False],
                "wt_use_htf_filter": [False],
                "wt_ema_filter_len": [20],
                "wt_long_entry_max_above_zero": [-50.0, -40.0],
                "wt_short_entry_min_below_zero": [30.0],
                "wt_h4_long_filter_max": [-60.0, -50.0],
                "wt_h4_short_filter_min": [50.0],
                "wt_long_close_min_level": [50.0, 60.0],
                "wt_h4_long_close_min": [50.0],
                "wt_long_emergency_sl_capital_pct": [0.0],
                "wt_long_tp1_pct": [0.01],
                "wt_long_tp2_pct": [0.03],
                "wt_long_tp1_fraction": [0.25],
                "wt_long_tp2_fraction": [0.25],
                "wt_long_tp1_timeout_hours": [24.0, 72.0],
            }
        }

        restored = _wfo_grid_values_from_result(result)
        by_component = {
            component_id: value
            for (component_id, _grid_key), value in zip(WFO_GRID_CONTROL_ORDER, restored)
        }

        assert by_component["chk-grid-long-zone"] == [-50.0, -40.0]
        assert by_component["chk-grid-h4-long"] == [-60.0, -50.0]
        assert by_component["chk-grid-long-tp1-timeout"] == [24.0, 72.0]
        assert by_component["chk-grid-reentry"] == [0, 3]


class TestPineExport:
    def test_repeated_wfo_logical_ids_get_unique_pine_open_markers(self):
        from bee4_dashboard import _build_pine_trades_overlay

        result = {
            "symbol": "ETHUSDT",
            "tf": "1h",
            "mode": "wfo",
            "trades": [
                {
                    "side": "long",
                    "entry_time": "2025-01-01 00:00:00",
                    "exit_time": "2025-01-01 03:00:00",
                    "entry_price": 1000.0,
                    "exit_price": 1010.0,
                    "logical_trade_no": 1,
                    "trade_event": "EXIT",
                    "pnl": 100.0,
                    "entry_window_id": 2,
                    "exit_window_id": 3,
                    "entry_params_wt_long_entry_max_above_zero": -50.0,
                    "entry_params_wt_h4_long_filter_max": -50.0,
                    "entry_params_wt_long_close_min_level": 60.0,
                    "entry_params_wt_h4_long_close_min": 60.0,
                    "entry_params_wt_long_tp1_pct": 0.01,
                    "entry_params_wt_long_tp2_pct": 0.03,
                    "entry_params_wt_long_tp1_fraction": 0.25,
                    "entry_params_wt_long_tp2_fraction": 0.25,
                    "entry_params_wt_long_tp1_timeout_hours": 72.0,
                    "exit_params_wt_long_entry_max_above_zero": -40.0,
                    "exit_params_wt_h4_long_filter_max": -40.0,
                    "exit_params_wt_long_close_min_level": 50.0,
                    "exit_params_wt_h4_long_close_min": 50.0,
                    "exit_params_wt_long_tp1_pct": 0.02,
                    "exit_params_wt_long_tp2_pct": 0.05,
                    "exit_params_wt_long_tp1_fraction": 0.5,
                    "exit_params_wt_long_tp2_fraction": 0.5,
                    "exit_params_wt_long_tp1_timeout_hours": 48.0,
                },
                {
                    "side": "long",
                    "entry_time": "2025-02-01 00:00:00",
                    "exit_time": "2025-02-01 03:00:00",
                    "entry_price": 1100.0,
                    "exit_price": 1110.0,
                    "logical_trade_no": 1,
                    "trade_event": "EXIT",
                    "pnl": 100.0,
                },
            ],
        }

        pine = _build_pine_trades_overlay(result)

        assert "OPEN LONG" in pine
        assert 'f_add(timestamp("UTC",2025,1,1,0,0),"OPEN_LONG",1000,1,"T1 OPEN"' in pine
        assert 'f_add(timestamp("UTC",2025,2,1,0,0),"OPEN_LONG",1100,2,"T2 OPEN"' in pine
        assert "entry_window=2 | open H1/H4=-50/-50 | close H1/H4=60/60" in pine
        assert "TP1=1%/25% | TP2=3%/25% | TP1 timeout=72h" in pine
        assert "exit_window=3 | open H1/H4=-40/-40 | close H1/H4=50/50" in pine
        assert "TP1=2%/50% | TP2=5%/50% | TP1 timeout=48h" in pine
        assert 'text="LONG"' in pine
        assert "snapToBar" in pine
        assert "array.set(matched, i, true)" in pine
        assert "yloc=yloc.price" in pine


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
