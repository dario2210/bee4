"""
test_bee4.py
=============
Pytest suite for the first bee4 WaveTrend strategy.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bee4_binance import interval_to_ms, wfo_bars
from bee4_data import load_klines, prepare_indicators, wt_columns
from bee4_engine import (
    BarData,
    PositionState,
    apply_slippage,
    bar_from_row,
    generate_entry_signal,
    generate_exit_signal,
)
from bee4_strategy import Bee4Strategy
from bee4_wfo import get_latest_best_params, walk_forward_optimization


BASE_PARAMS = {
    "wt_channel_len": 10,
    "wt_avg_len": 21,
    "wt_signal_len": 4,
    "wt_min_signal_level": 5.0,
    "wt_zero_line": 0.0,
    "allow_shorts": True,
    "wt_long_entry_window_bars": 3,
    "wt_long_entry_max_above_zero": 0.0,
    "wt_long_exit_min_level": 0.0,
    "wt_long_require_ema20_reclaim": True,
    "wt_short_entry_window_bars": 3,
    "wt_short_entry_min_below_zero": 0.0,
    "wt_short_exit_max_level": 0.0,
    "wt_short_require_ema20_reject": True,
    "fee_rate": 0.0007,
    "slippage_bps": 0.0,
    "spread_bps": 0.0,
    "max_bars_in_trade": 0,
}

LONG_ONLY_PARAMS = {
    **BASE_PARAMS,
    "allow_shorts": False,
}

REVERSAL_TEST_PARAMS = {
    **BASE_PARAMS,
    "wt_long_entry_window_bars": 0,
    "wt_short_entry_window_bars": 0,
    "wt_long_require_ema20_reclaim": False,
    "wt_short_require_ema20_reject": False,
}


def _make_bar(
    close=1800.0,
    wt1=-20.0,
    wt2=-25.0,
    ema20=1700.0,
    wt_green_dot=False,
    wt_red_dot=False,
    bars_since_wt_green_dot=np.nan,
    bars_since_wt_red_dot=np.nan,
):
    return BarData(
        time=pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
        close=close,
        wt1=wt1,
        wt2=wt2,
        wt_delta=wt1 - wt2,
        ema20=ema20,
        wt_green_dot=wt_green_dot,
        wt_red_dot=wt_red_dot,
        bars_since_wt_green_dot=bars_since_wt_green_dot,
        bars_since_wt_red_dot=bars_since_wt_red_dot,
    )


def _signal_df() -> pd.DataFrame:
    wt1_col, wt2_col = wt_columns(
        BASE_PARAMS["wt_channel_len"],
        BASE_PARAMS["wt_avg_len"],
        BASE_PARAMS["wt_signal_len"],
    )
    times = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    closes = [1800.0, 1810.0, 1825.0, 1815.0, 1795.0, 1785.0]
    wt1_vals = [-40.0, -24.0, -18.0, 22.0, 18.0, -16.0]
    wt2_vals = [-36.0, -28.0, -20.0, 26.0, 20.0, -20.0]

    df = pd.DataFrame(
        {
            "time": times,
            "open": closes,
            "high": [c + 5 for c in closes],
            "low": [c - 5 for c in closes],
            "close": closes,
            "volume": [1000.0] * len(closes),
            wt1_col: wt1_vals,
            wt2_col: wt2_vals,
            "wt1": wt1_vals,
            "wt2": wt2_vals,
            "wt_delta": np.array(wt1_vals) - np.array(wt2_vals),
            "ema20": pd.Series(closes).ewm(span=20, adjust=False).mean(),
        }
    )
    df["wt_green_dot"] = (df["wt1"].shift(1) <= df["wt2"].shift(1)) & (df["wt1"] > df["wt2"])
    df["wt_red_dot"] = (df["wt1"].shift(1) >= df["wt2"].shift(1)) & (df["wt1"] < df["wt2"])
    bars_since_green = []
    bars_since_red = []
    last_green = None
    last_red = None
    for idx, row in df.iterrows():
        if bool(row["wt_green_dot"]):
            last_green = idx
        if bool(row["wt_red_dot"]):
            last_red = idx
        bars_since_green.append(np.nan if last_green is None else float(idx - last_green))
        bars_since_red.append(np.nan if last_red is None else float(idx - last_red))
    df["bars_since_wt_green_dot"] = bars_since_green
    df["bars_since_wt_red_dot"] = bars_since_red
    return df


class TestWaveTrendPreparation:
    def test_prepare_indicators_adds_default_columns(self):
        times = pd.date_range("2024-01-01", periods=120, freq="1h", tz="UTC")
        close = np.linspace(1500.0, 1700.0, len(times))
        df = pd.DataFrame(
            {
                "time": times,
                "open": close,
                "high": close + 4,
                "low": close - 4,
                "close": close,
                "volume": 1000.0,
            }
        )

        out = prepare_indicators(df)
        wt1_col, wt2_col = wt_columns(10, 21, 4)

        for col in [
            "hlc3",
            "ema20",
            "wt1",
            "wt2",
            "wt_delta",
            "wt_green_dot",
            "wt_red_dot",
            "bars_since_wt_green_dot",
            "bars_since_wt_red_dot",
            wt1_col,
            wt2_col,
        ]:
            assert col in out.columns

        assert out["wt1"].dropna().shape[0] > 0
        assert out["wt2"].dropna().shape[0] > 0


class TestEntrySignals:
    def test_open_long_on_green_dot_below_zero(self):
        prev = _make_bar(wt1=-35.0, wt2=-32.0)
        bar = _make_bar(wt1=-24.0, wt2=-28.0)
        sig = generate_entry_signal(bar, prev, BASE_PARAMS, None)
        assert sig.action == "open_long"
        assert sig.reason == "WT_GREEN_DOT_BELOW_ZERO"

    def test_open_long_in_reentry_window_after_green_dot(self):
        prev = _make_bar(wt1=-20.0, wt2=-18.0)
        bar = _make_bar(
            wt1=-12.0,
            wt2=-10.0,
            wt_green_dot=False,
            bars_since_wt_green_dot=2.0,
        )
        sig = generate_entry_signal(bar, prev, BASE_PARAMS, None)
        assert sig.action == "open_long"
        assert sig.reason == "WT_LONG_REENTRY_WINDOW"

    def test_no_long_without_ema20_reclaim(self):
        prev = _make_bar(wt1=-35.0, wt2=-32.0, ema20=1900.0)
        bar = _make_bar(wt1=-24.0, wt2=-28.0, ema20=1900.0)
        sig = generate_entry_signal(bar, prev, BASE_PARAMS, None)
        assert sig.action == "none"

    def test_open_short_on_red_dot_above_zero(self):
        prev = _make_bar(wt1=34.0, wt2=30.0, ema20=1900.0)
        bar = _make_bar(wt1=22.0, wt2=27.0, ema20=1900.0)
        sig = generate_entry_signal(bar, prev, BASE_PARAMS, None)
        assert sig.action == "open_short"
        assert sig.reason == "WT_RED_DOT_ABOVE_ZERO"

    def test_open_short_in_reentry_window_after_red_dot(self):
        prev = _make_bar(wt1=20.0, wt2=18.0, ema20=1900.0)
        bar = _make_bar(
            close=1790.0,
            wt1=12.0,
            wt2=10.0,
            ema20=1850.0,
            wt_red_dot=False,
            bars_since_wt_red_dot=2.0,
        )
        sig = generate_entry_signal(bar, prev, BASE_PARAMS, None)
        assert sig.action == "open_short"
        assert sig.reason == "WT_SHORT_REENTRY_WINDOW"

    def test_no_short_without_ema20_reject(self):
        prev = _make_bar(wt1=34.0, wt2=30.0, close=1950.0, ema20=1900.0)
        bar = _make_bar(wt1=22.0, wt2=27.0, close=1950.0, ema20=1900.0)
        sig = generate_entry_signal(bar, prev, BASE_PARAMS, None)
        assert sig.action == "none"

    def test_no_long_when_cross_is_above_zero(self):
        prev = _make_bar(wt1=3.0, wt2=5.0)
        bar = _make_bar(wt1=8.0, wt2=6.0)
        sig = generate_entry_signal(bar, prev, BASE_PARAMS, None)
        assert sig.action == "none"

    def test_no_signal_when_already_in_position(self):
        prev = _make_bar(wt1=-35.0, wt2=-32.0)
        bar = _make_bar(wt1=-24.0, wt2=-28.0)
        pos = PositionState("long", 1800.0, pd.Timestamp("2024-01-01", tz="UTC"))
        sig = generate_entry_signal(bar, prev, BASE_PARAMS, pos)
        assert sig.action == "none"


class TestExitSignals:
    def test_long_closes_on_red_dot_above_zero_in_long_only_mode(self):
        pos = PositionState("long", 1800.0, pd.Timestamp("2024-01-01", tz="UTC"))
        prev = _make_bar(wt1=30.0, wt2=26.0)
        bar = _make_bar(wt1=22.0, wt2=27.0, wt_red_dot=True)
        sig = generate_exit_signal(bar, prev, LONG_ONLY_PARAMS, pos)
        assert sig.action == "close_force"
        assert sig.reason == "WT_RED_DOT_EXIT_LONG"

    def test_long_closes_only_on_reverse_short_signal_when_shorts_enabled(self):
        pos = PositionState("long", 1800.0, pd.Timestamp("2024-01-01", tz="UTC"))
        prev = _make_bar(wt1=30.0, wt2=26.0)
        bar = _make_bar(wt1=22.0, wt2=27.0)
        sig = generate_exit_signal(bar, prev, REVERSAL_TEST_PARAMS, pos)
        assert sig.action == "close_reverse"
        assert sig.reason == "REVERSE_TO_SHORT"

    def test_short_closes_only_on_reverse_long_signal(self):
        pos = PositionState("short", 1800.0, pd.Timestamp("2024-01-01", tz="UTC"))
        prev = _make_bar(wt1=-34.0, wt2=-30.0)
        bar = _make_bar(wt1=-24.0, wt2=-28.0)
        sig = generate_exit_signal(bar, prev, REVERSAL_TEST_PARAMS, pos)
        assert sig.action == "close_reverse"
        assert sig.reason == "REVERSE_TO_LONG"

    def test_short_closes_on_green_dot_below_zero_when_reversal_is_disabled(self):
        pos = PositionState("short", 1800.0, pd.Timestamp("2024-01-01", tz="UTC"))
        prev = _make_bar(wt1=-30.0, wt2=-34.0)
        bar = _make_bar(wt1=-24.0, wt2=-28.0, wt_green_dot=True)
        sig = generate_exit_signal(bar, prev, LONG_ONLY_PARAMS, pos)
        assert sig.action == "close_force"
        assert sig.reason == "WT_GREEN_DOT_EXIT_SHORT"

    def test_time_stop_remains_optional_safety(self):
        params = dict(BASE_PARAMS, max_bars_in_trade=2)
        pos = PositionState("long", 1800.0, pd.Timestamp("2024-01-01", tz="UTC"), bars_in_position=1)
        prev = _make_bar(wt1=-20.0, wt2=-24.0)
        bar = _make_bar(wt1=-19.0, wt2=-18.0)
        sig = generate_exit_signal(bar, prev, params, pos)
        assert sig.action == "close_force"
        assert sig.reason == "TIME_STOP"


class TestStrategy:
    def test_strategy_reverses_on_opposite_signal_when_shorts_enabled(self):
        df = _signal_df()
        strat = Bee4Strategy(REVERSAL_TEST_PARAMS)
        trades, equity, final_cap = strat.run(df, 10_000.0)

        assert len(trades) == 3
        assert list(trades["side"]) == ["long", "short", "long"]
        assert trades.iloc[0]["reason"] == "REVERSE_TO_SHORT"
        assert trades.iloc[1]["reason"] == "REVERSE_TO_LONG"
        assert trades.iloc[2]["reason"] == "FORCE_EXIT_END"
        assert final_cap > 0
        assert not equity.empty

    def test_capital_after_matches_final_cap(self):
        df = _signal_df()
        strat = Bee4Strategy(REVERSAL_TEST_PARAMS)
        trades, _equity, final_cap = strat.run(df, 10_000.0)
        assert trades.iloc[-1]["capital_after"] == pytest.approx(final_cap, abs=1e-8)

    def test_long_only_profile_exits_without_opening_shorts(self):
        df = _signal_df()
        strat = Bee4Strategy(LONG_ONLY_PARAMS)
        trades, _equity, _final_cap = strat.run(df, 10_000.0)

        assert len(trades) == 1
        assert set(trades["side"]) == {"long"}
        assert list(trades["reason"]) == ["WT_RED_DOT_EXIT_LONG"]


class TestDataLoading:
    def _write_csv(self, path, open_time_col, open_time_values):
        df = pd.DataFrame(
            {
                open_time_col: open_time_values,
                "open": [1800.0] * len(open_time_values),
                "high": [1810.0] * len(open_time_values),
                "low": [1790.0] * len(open_time_values),
                "close": [1805.0] * len(open_time_values),
                "volume": [1000.0] * len(open_time_values),
            }
        )
        df.to_csv(path, index=False)
        return path

    def test_load_klines_ms(self):
        path = Path(".test_load_klines_ms.csv")
        try:
            times_ms = [
                int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000) + i * 3600000
                for i in range(5)
            ]
            csv_path = self._write_csv(str(path), "open_time", times_ms)
            df = load_klines(csv_path)
            assert len(df) == 5
            assert "time" in df.columns
        finally:
            path.unlink(missing_ok=True)

    def test_load_klines_string(self):
        path = Path(".test_load_klines_string.csv")
        try:
            times_str = [f"2024-01-0{i+1} 00:00:00+00:00" for i in range(5)]
            csv_path = self._write_csv(str(path), "open_time", times_str)
            df = load_klines(csv_path)
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


class TestWFOHelpers:
    def test_wfo_bars_1h(self):
        ob, lb = wfo_bars("1h", 90, 14)
        assert ob == 2160
        assert lb == 336

    def test_interval_to_ms(self):
        assert interval_to_ms("1h") == 3_600_000

    def test_save_and_load_params_json(self):
        from bee4_params import save_params

        path = Path(".test_params.json")
        try:
            save_params(dict(BASE_PARAMS), str(path))
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded["wt_channel_len"] == BASE_PARAMS["wt_channel_len"]
            assert loaded["wt_signal_len"] == BASE_PARAMS["wt_signal_len"]
        finally:
            path.unlink(missing_ok=True)

    def test_get_latest_best_params_includes_new_strategy_filters(self):
        windows_df = pd.DataFrame(
            {
                "best_wt_channel_len": [8, 10, 10, 10, 12],
                "best_wt_avg_len": [14, 21, 21, 21, 28],
                "best_wt_signal_len": [3, 4, 4, 4, 5],
                "best_wt_min_signal_level": [0.0, 10.0, 10.0, 10.0, 20.0],
                "best_wt_reentry_window_bars": [1, 2, 2, 2, 3],
                "best_wt_use_ema_filter": [False, True, True, True, False],
                "best_wt_long_entry_max_above_zero": [0.0, 5.0, 5.0, 5.0, 10.0],
                "best_wt_short_entry_min_below_zero": [-10.0, -5.0, -5.0, -5.0, 0.0],
                "n_trades_live": [1, 2, 2, 1, 0],
            }
        )

        best = get_latest_best_params(windows_df)

        assert best["wt_channel_len"] == 10
        assert best["wt_avg_len"] == 21
        assert best["wt_signal_len"] == 4
        assert best["wt_min_signal_level"] == pytest.approx(10.0)
        assert best["wt_long_entry_window_bars"] == 2
        assert best["wt_short_entry_window_bars"] == 2
        assert best["wt_long_require_ema20_reclaim"] is True
        assert best["wt_short_require_ema20_reject"] is True
        assert best["wt_long_entry_max_above_zero"] == pytest.approx(5.0)
        assert best["wt_short_entry_min_below_zero"] == pytest.approx(-5.0)

    def test_wfo_accepts_new_grid_overrides(self):
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
            "wt_channel_len": [8],
            "wt_avg_len": [14],
            "wt_signal_len": [3],
            "wt_min_signal_level": [0.0],
            "wt_reentry_window_bars": [2],
            "wt_use_ema_filter": [True],
            "wt_long_entry_max_above_zero": [5.0],
            "wt_short_entry_min_below_zero": [-5.0],
        }

        _trades, _equity, windows_df, _final_cap = walk_forward_optimization(
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

        assert not windows_df.empty
        assert set(windows_df["best_wt_channel_len"]) == {8}
        assert set(windows_df["best_wt_avg_len"]) == {14}
        assert set(windows_df["best_wt_signal_len"]) == {3}
        assert set(windows_df["best_wt_min_signal_level"]) == {0.0}
        assert set(windows_df["best_wt_reentry_window_bars"]) == {2}
        assert set(windows_df["best_wt_use_ema_filter"]) == {True}
        assert set(windows_df["best_wt_long_entry_max_above_zero"]) == {5.0}
        assert set(windows_df["best_wt_short_entry_min_below_zero"]) == {-5.0}


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

