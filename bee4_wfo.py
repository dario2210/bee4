"""
bee4_wfo.py
============
Walk-forward optimization for the first bee4 WaveTrend strategy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from bee4_binance import wfo_bars
from bee4_params import (
    BINANCE_INTERVAL,
    FEE_RATE,
    INITIAL_CAPITAL,
    LIVE_DAYS,
    OPT_DAYS,
    WT_AVG_LEN,
    WT_AVG_LEN_GRID,
    WT_CHANNEL_LEN,
    WT_CHANNEL_LEN_GRID,
    WT_MIN_SIGNAL_LEVEL,
    WT_MIN_SIGNAL_LEVEL_GRID,
    WT_SIGNAL_LEN,
    WT_SIGNAL_LEN_GRID,
)
from bee4_strategy import Bee4Strategy
from bee4_wfo_scoring import score_params


def walk_forward_optimization(
    df: pd.DataFrame,
    interval: str = BINANCE_INTERVAL,
    score_mode: str = "balanced",
    verbose: bool = True,
    on_window_done=None,
    fee_rate: float = FEE_RATE,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, float]:
    """Run WFO over precomputed WaveTrend indicator data."""
    opt_bars, live_bars = wfo_bars(interval, OPT_DAYS, LIVE_DAYS)
    opt_capital = 10_000.0

    n = len(df)
    start = 0
    window_id = 0
    current_capital = INITIAL_CAPITAL

    all_live_trades: list[pd.DataFrame] = []
    global_equity: Optional[pd.DataFrame] = None
    window_stats: list[dict] = []

    total_windows = max(0, (n - opt_bars) // live_bars)
    if verbose:
        print(
            f"[WFO] candles={n} | windows~={total_windows} | "
            f"opt={OPT_DAYS}d ({opt_bars} bars) | live={LIVE_DAYS}d ({live_bars} bars) | "
            f"score_mode={score_mode}"
        )
        print("-" * 70)

    while start + opt_bars + live_bars <= n:
        opt_slice = df.iloc[start : start + opt_bars]
        live_slice = df.iloc[start + opt_bars : start + opt_bars + live_bars]

        best_score = -1e9
        best_params = None
        best_opt_trades = None
        best_opt_cap = opt_capital

        for wt_channel_len in WT_CHANNEL_LEN_GRID:
            for wt_avg_len in WT_AVG_LEN_GRID:
                for wt_signal_len in WT_SIGNAL_LEN_GRID:
                    for wt_min_signal_level in WT_MIN_SIGNAL_LEVEL_GRID:
                        params = {
                            "wt_channel_len": wt_channel_len,
                            "wt_avg_len": wt_avg_len,
                            "wt_signal_len": wt_signal_len,
                            "wt_min_signal_level": wt_min_signal_level,
                        }
                        strat = Bee4Strategy(params, fee_rate=fee_rate)
                        trades_opt, _, final_cap_opt = strat.run(opt_slice, opt_capital)
                        score = score_params(trades_opt, final_cap_opt, opt_capital, mode=score_mode)
                        if score > best_score:
                            best_score = score
                            best_params = params
                            best_opt_trades = trades_opt
                            best_opt_cap = final_cap_opt

        if best_params is None:
            if verbose:
                print(f"[WFO] Window {window_id}: no usable params, stopping.")
            break

        opt_n_trades = 0 if best_opt_trades is None or best_opt_trades.empty else len(best_opt_trades)
        opt_ret_pct = (best_opt_cap / opt_capital - 1.0) * 100.0
        opt_pf = 0.0
        opt_max_dd = 0.0
        if best_opt_trades is not None and not best_opt_trades.empty:
            wins = best_opt_trades[best_opt_trades["pnl"] > 0]["pnl"].sum()
            losses = best_opt_trades[best_opt_trades["pnl"] <= 0]["pnl"].sum()
            opt_pf = wins / abs(losses) if losses < 0 else 0.0
            equity = np.array([opt_capital] + list(opt_capital + best_opt_trades["pnl"].cumsum().values))
            running_max = np.maximum.accumulate(equity)
            dd_arr = (equity - running_max) / running_max
            opt_max_dd = dd_arr.min() * 100.0

        strat = Bee4Strategy(best_params, fee_rate=fee_rate)
        trades_live, equity_live, final_cap_live = strat.run(live_slice, current_capital)

        if not trades_live.empty:
            trades_live = trades_live.copy()
            trades_live["window_id"] = window_id
            trades_live["wt_channel_len"] = best_params["wt_channel_len"]
            trades_live["wt_avg_len"] = best_params["wt_avg_len"]
            trades_live["wt_signal_len"] = best_params["wt_signal_len"]
            trades_live["wt_min_signal_level"] = best_params["wt_min_signal_level"]
            all_live_trades.append(trades_live)

        if equity_live is not None and not equity_live.empty:
            global_equity = (
                equity_live.copy()
                if global_equity is None
                else pd.concat([global_equity, equity_live.iloc[1:]], ignore_index=True)
            )

        live_ret_pct = (final_cap_live / current_capital - 1.0) * 100.0 if current_capital > 0 else 0.0
        n_trades_live = 0 if trades_live.empty else len(trades_live)

        window_stats.append(
            {
                "window_id": window_id,
                "live_start": live_slice["time"].iloc[0],
                "live_end": live_slice["time"].iloc[-1],
                "best_wt_channel_len": best_params["wt_channel_len"],
                "best_wt_avg_len": best_params["wt_avg_len"],
                "best_wt_signal_len": best_params["wt_signal_len"],
                "best_wt_min_signal_level": best_params["wt_min_signal_level"],
                "opt_score": best_score,
                "opt_return_pct": opt_ret_pct,
                "opt_pf": opt_pf,
                "opt_max_dd_pct": opt_max_dd,
                "opt_n_trades": opt_n_trades,
                "live_return_pct": live_ret_pct,
                "live_final_cap": final_cap_live,
                "n_trades_live": n_trades_live,
            }
        )

        if verbose:
            print(
                f"[WFO] {window_id:3d} | "
                f"{live_slice['time'].iloc[0].strftime('%Y-%m-%d')} -> "
                f"{live_slice['time'].iloc[-1].strftime('%Y-%m-%d')} | "
                f"ret={live_ret_pct:+.2f}% tr={n_trades_live} "
                f"ch={best_params['wt_channel_len']} avg={best_params['wt_avg_len']} "
                f"sig={best_params['wt_signal_len']} minlvl={best_params['wt_min_signal_level']:.1f}"
            )

        if on_window_done is not None:
            on_window_done(
                window_id,
                total_windows,
                list(window_stats),
                list(all_live_trades),
                global_equity.copy() if global_equity is not None else None,
                current_capital,
            )

        current_capital = final_cap_live
        start += live_bars
        window_id += 1

    all_trades_df = pd.concat(all_live_trades, ignore_index=True) if all_live_trades else pd.DataFrame()
    windows_df = pd.DataFrame(window_stats)
    return all_trades_df, global_equity, windows_df, current_capital


def get_latest_best_params(windows_df: pd.DataFrame) -> dict:
    """
    Return a stable parameter set from the last few WFO windows.
    We prefer windows with live activity and use mode/median aggregation.
    """
    if windows_df is None or windows_df.empty:
        return {}

    recent = windows_df.tail(5).copy()
    if "n_trades_live" in recent.columns:
        active = recent[recent["n_trades_live"] >= 1]
        if len(active) >= 2:
            recent = active

    channel_len = int(recent["best_wt_channel_len"].mode().iloc[0]) if "best_wt_channel_len" in recent.columns else WT_CHANNEL_LEN
    avg_len = int(recent["best_wt_avg_len"].mode().iloc[0]) if "best_wt_avg_len" in recent.columns else WT_AVG_LEN
    signal_len = int(recent["best_wt_signal_len"].mode().iloc[0]) if "best_wt_signal_len" in recent.columns else WT_SIGNAL_LEN
    min_signal_level = (
        float(recent["best_wt_min_signal_level"].median())
        if "best_wt_min_signal_level" in recent.columns
        else WT_MIN_SIGNAL_LEVEL
    )

    return {
        "wt_channel_len": channel_len,
        "wt_avg_len": avg_len,
        "wt_signal_len": signal_len,
        "wt_min_signal_level": min_signal_level,
    }

