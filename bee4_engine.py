"""
bee4_engine.py
===============
Shared decision layer for the first bee4 WaveTrend strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from bee4_data import wt_columns
from bee4_params import WT_ZERO_LINE

Side = Literal["long", "short"]
Action = Literal["none", "open_long", "open_short", "close_reverse", "close_force"]


@dataclass
class BarData:
    """Single-candle input for the strategy engine."""

    time: object
    close: float
    wt1: float
    wt2: float
    wt_delta: float
    ema20: float = np.nan
    wt_green_dot: bool = False
    wt_red_dot: bool = False
    bars_since_wt_green_dot: float = np.nan
    bars_since_wt_red_dot: float = np.nan


@dataclass
class Signal:
    action: Action
    reason: str = ""
    exit_price: Optional[float] = None
    meta: dict = field(default_factory=dict)


@dataclass
class PositionState:
    side: Side
    entry_price: float
    entry_time: object
    bars_in_position: int = 0
    entry_meta: dict = field(default_factory=dict)


def compute_trade_close(
    entry_price: float,
    exit_price: float,
    side: Side,
    fee_rate: float,
    capital_at_open: float,
) -> dict:
    """Shared trade close accounting used by backtest and live runner."""
    if side == "long":
        gross_ret = (exit_price / entry_price) - 1.0
    else:
        gross_ret = (entry_price / exit_price) - 1.0

    net_ret = (1.0 + gross_ret) * (1.0 - fee_rate) ** 2 - 1.0
    fee_ret = net_ret - gross_ret
    fee_usd = abs(fee_ret) * capital_at_open
    pnl = capital_at_open * net_ret

    return {
        "gross_ret": gross_ret,
        "net_ret": net_ret,
        "fee_ret": fee_ret,
        "fee_usd": fee_usd,
        "pnl": pnl,
    }


def _zone(wt1: float, wt2: float, zero_line: float) -> str:
    if wt1 < zero_line and wt2 < zero_line:
        return "below_zero"
    if wt1 > zero_line and wt2 > zero_line:
        return "above_zero"
    return "mixed"


def _cross_up(bar: BarData, prev_bar: BarData) -> bool:
    return prev_bar.wt_delta <= 0.0 and bar.wt_delta > 0.0


def _cross_down(bar: BarData, prev_bar: BarData) -> bool:
    return prev_bar.wt_delta >= 0.0 and bar.wt_delta < 0.0


def _signal_level(bar: BarData) -> float:
    return min(abs(bar.wt1), abs(bar.wt2))


def _has_recent_signal(value: float, max_bars: int) -> bool:
    return not np.isnan(value) and value <= float(max_bars)


def _flag_value(value) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    return bool(value)


def _float_or_nan(value) -> float:
    if value is None:
        return np.nan
    try:
        return np.nan if np.isnan(value) else float(value)
    except TypeError:
        return float(value)


def generate_entry_signal(
    bar: BarData,
    prev_bar: BarData,
    params: dict,
    position: Optional[PositionState],
) -> Signal:
    """
    Entry logic:
      - long on bullish WaveTrend dot/cross when both lines are below zero
      - short on bearish WaveTrend dot/cross when both lines are above zero
      - optional minimum distance from zero via wt_min_signal_level
    """
    if position is not None:
        return Signal(action="none")

    if any(np.isnan(v) for v in [bar.wt1, bar.wt2, prev_bar.wt1, prev_bar.wt2]):
        return Signal(action="none")

    zero_line = float(params.get("wt_zero_line", WT_ZERO_LINE))
    min_level = float(params.get("wt_min_signal_level", 0.0))
    level_now = _signal_level(bar)

    allow_shorts = bool(params.get("allow_shorts", True))
    entry_window_bars = int(params.get("wt_long_entry_window_bars", 0) or 0)
    entry_max_above_zero = float(params.get("wt_long_entry_max_above_zero", 0.0))
    require_ema20_reclaim = bool(params.get("wt_long_require_ema20_reclaim", False))
    ema20_ok = not require_ema20_reclaim or (not np.isnan(bar.ema20) and bar.close > bar.ema20)

    fresh_long_cross = (
        _cross_up(bar, prev_bar)
        and bar.wt1 < zero_line
        and bar.wt2 < zero_line
        and level_now >= min_level
    )
    long_reentry_window = (
        entry_window_bars > 0
        and _has_recent_signal(bar.bars_since_wt_green_dot, entry_window_bars)
        and bar.wt1 <= entry_max_above_zero
        and bar.wt2 <= entry_max_above_zero
        and level_now >= min_level
        and ema20_ok
    )
    long_cond = (fresh_long_cross and ema20_ok) or long_reentry_window
    short_cond = (
        allow_shorts
        and _cross_down(bar, prev_bar)
        and bar.wt1 > zero_line
        and bar.wt2 > zero_line
        and level_now >= min_level
    )

    meta = {
        "entry_wt1": round(bar.wt1, 4),
        "entry_wt2": round(bar.wt2, 4),
        "entry_delta": round(bar.wt_delta, 4),
        "entry_zone": _zone(bar.wt1, bar.wt2, zero_line),
        "entry_signal_level": round(level_now, 4),
        "prev_wt1": round(prev_bar.wt1, 4),
        "prev_wt2": round(prev_bar.wt2, 4),
        "entry_ema20": round(bar.ema20, 4) if not np.isnan(bar.ema20) else np.nan,
    }

    if long_cond:
        long_reason = "WT_GREEN_DOT_BELOW_ZERO" if fresh_long_cross else "WT_LONG_REENTRY_WINDOW"
        return Signal(
            action="open_long",
            reason=long_reason,
            meta={
                **meta,
                "cross_type": "bullish" if fresh_long_cross else "bullish_reentry",
                "bars_since_green_dot": round(bar.bars_since_wt_green_dot, 4)
                if not np.isnan(bar.bars_since_wt_green_dot)
                else np.nan,
            },
        )
    if short_cond:
        return Signal(
            action="open_short",
            reason="WT_RED_DOT_ABOVE_ZERO",
            meta={**meta, "cross_type": "bearish"},
        )
    return Signal(action="none")


def generate_exit_signal(
    bar: BarData,
    prev_bar: BarData,
    params: dict,
    position: PositionState,
) -> Signal:
    """
    Exit logic:
      - close long only when the opposite short signal appears
      - close short only when the opposite long signal appears
      - optional time stop remains as a safety override when max_bars_in_trade > 0
    """
    position.bars_in_position += 1

    max_bars = int(params.get("max_bars_in_trade", 0) or 0)
    zero_line = float(params.get("wt_zero_line", WT_ZERO_LINE))

    def _meta(trigger: str) -> dict:
        return {
            "exit_wt1": round(bar.wt1, 4),
            "exit_wt2": round(bar.wt2, 4),
            "exit_delta": round(bar.wt_delta, 4),
            "exit_zone": _zone(bar.wt1, bar.wt2, zero_line),
            "exit_signal_level": round(_signal_level(bar), 4),
            "bars_in_position": position.bars_in_position,
            "exit_trigger": trigger,
        }

    if max_bars > 0 and position.bars_in_position >= max_bars:
        return Signal(action="close_force", reason="TIME_STOP", meta=_meta("TIME_STOP"))

    allow_shorts = bool(params.get("allow_shorts", True))
    long_exit_min_level = float(params.get("wt_long_exit_min_level", zero_line))

    opposite_signal = generate_entry_signal(bar, prev_bar, params, None)
    if position.side == "long":
        if allow_shorts and opposite_signal.action == "open_short":
            return Signal(
                action="close_reverse",
                reason="REVERSE_TO_SHORT",
                meta=_meta("REVERSE_TO_SHORT"),
            )
        if bar.wt_red_dot and bar.wt1 > long_exit_min_level and bar.wt2 > long_exit_min_level:
            return Signal(
                action="close_force",
                reason="WT_RED_DOT_EXIT_LONG",
                meta=_meta("WT_RED_DOT_EXIT_LONG"),
            )

    if position.side == "short":
        if opposite_signal.action == "open_long":
            return Signal(
                action="close_reverse",
                reason="REVERSE_TO_LONG",
                meta=_meta("REVERSE_TO_LONG"),
            )
        if not allow_shorts and bar.wt_green_dot and bar.wt1 < zero_line and bar.wt2 < zero_line:
            return Signal(
                action="close_force",
                reason="WT_GREEN_DOT_EXIT_SHORT",
                meta=_meta("WT_GREEN_DOT_EXIT_SHORT"),
            )

    return Signal(action="none")


def apply_slippage(
    price: float,
    side: Side,
    action: str,
    slippage_bps: float = 0.0,
    spread_bps: float = 0.0,
) -> float:
    """Adjust execution price by slippage and half-spread."""
    if slippage_bps == 0.0 and spread_bps == 0.0:
        return price

    total_bps = slippage_bps + spread_bps / 2.0
    factor = total_bps / 10_000.0
    opening = action == "open"

    if (side == "long" and opening) or (side == "short" and not opening):
        return price * (1.0 + factor)
    return price * (1.0 - factor)


def bar_from_row(row, params: dict) -> BarData:
    """Build BarData for the parameter-specific WaveTrend columns."""
    wt1_col, wt2_col = wt_columns(
        int(params["wt_channel_len"]),
        int(params["wt_avg_len"]),
        int(params["wt_signal_len"]),
    )

    wt1 = row[wt1_col] if wt1_col in row.index else row.get("wt1", np.nan)
    wt2 = row[wt2_col] if wt2_col in row.index else row.get("wt2", np.nan)
    wt1 = float(wt1) if not np.isnan(wt1) else np.nan
    wt2 = float(wt2) if not np.isnan(wt2) else np.nan
    bars_since_wt_green_dot = row.get("bars_since_wt_green_dot", np.nan)
    bars_since_wt_red_dot = row.get("bars_since_wt_red_dot", np.nan)

    return BarData(
        time=row["time"],
        close=float(row["close"]),
        wt1=wt1,
        wt2=wt2,
        wt_delta=float(wt1 - wt2) if not np.isnan(wt1) and not np.isnan(wt2) else np.nan,
        ema20=_float_or_nan(row.get("ema20", np.nan)),
        wt_green_dot=_flag_value(row.get("wt_green_dot", False)),
        wt_red_dot=_flag_value(row.get("wt_red_dot", False)),
        bars_since_wt_green_dot=(
            _float_or_nan(bars_since_wt_green_dot)
        ),
        bars_since_wt_red_dot=(
            _float_or_nan(bars_since_wt_red_dot)
        ),
    )

