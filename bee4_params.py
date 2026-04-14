"""
bee4_params.py
===============
Central configuration for the first bee4 WaveTrend strategy.

Project structure stays aligned with bee1 on purpose, but the trading logic
and WFO grids are now built around WaveTrend crossover signals.
"""

from __future__ import annotations

import json
import os

# Input data
CSV_PATH = "ethusdt_1h.csv"
INITIAL_CAPITAL = 10_000.0

# Binance data source
BINANCE_SYMBOL = "ETHUSDT"
BINANCE_INTERVAL = "1h"
BINANCE_MARKET = "spot"
BINANCE_START_DATE = "2021-01-01"
BINANCE_CSV_CACHE = None

# Default WaveTrend setup
WT_CHANNEL_LEN = 10
WT_AVG_LEN = 21
WT_SIGNAL_LEN = 4
WT_MIN_SIGNAL_LEVEL = 0.0
WT_ZERO_LINE = 0.0

# Fees and execution friction
FEE_RATE = 0.0007

# WFO windows
OPT_DAYS = 120
LIVE_DAYS = 14

# WFO parameter grid
WT_CHANNEL_LEN_GRID = [8, 10, 12, 14]
WT_AVG_LEN_GRID = [14, 21, 28, 35]
WT_SIGNAL_LEN_GRID = [3, 4, 5]
WT_MIN_SIGNAL_LEVEL_GRID = [0.0, 5.0, 10.0, 20.0]

# Compatibility aliases kept so the bee1 dashboard structure can stay intact
TP_GRID = WT_CHANNEL_LEN_GRID
SL_GRID = WT_AVG_LEN_GRID
TRAIL_DROP_GRID = WT_SIGNAL_LEN_GRID
EMA_LEN_GRID = WT_MIN_SIGNAL_LEVEL_GRID

# Visual guide bands for the dashboard signal panel
TMA_LOW_MIN = -60.0
TMA_LOW_MAX = -53.0
TMA_HIGH_MIN = 53.0
TMA_HIGH_MAX = 60.0

# Fallback params used when WFO JSON is missing
DEFAULT_PARAMS = {
    "wt_channel_len": WT_CHANNEL_LEN,
    "wt_avg_len": WT_AVG_LEN,
    "wt_signal_len": WT_SIGNAL_LEN,
    "wt_min_signal_level": WT_MIN_SIGNAL_LEVEL,
    "wt_zero_line": WT_ZERO_LINE,
    "fee_rate": FEE_RATE,
    "slippage_bps": 2.0,
    "spread_bps": 1.0,
    "max_bars_in_trade": 0,
}

WFO_BEST_PARAMS_PATH = "bee4_wfo_best_params.json"


def load_params() -> dict:
    """
    Return strategy params:
    1) start from DEFAULT_PARAMS,
    2) override with WFO_BEST_PARAMS_PATH when present.
    """
    params = dict(DEFAULT_PARAMS)

    json_path = os.path.join(os.path.dirname(__file__), WFO_BEST_PARAMS_PATH)
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key in DEFAULT_PARAMS:
                    if key in data:
                        params[key] = data[key]
            print(f"[params] Loaded WFO params from {WFO_BEST_PARAMS_PATH}")
        except Exception as exc:
            print(f"[params] Could not load {WFO_BEST_PARAMS_PATH}: {exc!r}")
    else:
        print(f"[params] Missing {WFO_BEST_PARAMS_PATH} - using DEFAULT_PARAMS")

    return params


def save_params(params: dict, path: str = WFO_BEST_PARAMS_PATH) -> None:
    """Persist strategy params to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"[params] Saved params -> {path}")

