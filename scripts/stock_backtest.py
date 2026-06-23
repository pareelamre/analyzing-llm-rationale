#!/usr/bin/env python3
"""Vectorised backtesting engine for cross-sectional and time-series strategies.

Importable module used by stock_sweep.py and stock_ml.py.

Strategies supported:
  momentum      -- cross-sectional momentum (rank by N-day return)
  vol_adj_mom   -- momentum / realized vol (Sharpe-scaled)
  ma_cross      -- time-series MA crossover (fast > slow → long)
  rsi_reversal  -- RSI mean reversion (RSI < lo → long, RSI > hi → short)
  bb_reversion  -- Bollinger Band position mean reversion

Transaction costs: flat bps per side (default 5bps = 0.05%).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

StrategyType = Literal["momentum", "vol_adj_mom", "ma_cross", "rsi_reversal", "bb_reversion"]
DirectionType = Literal["long_only", "short_only", "long_short"]


@dataclass
class BacktestParams:
    strategy: StrategyType = "momentum"
    lookback: int = 126        # days for signal computation
    holding: int = 21          # days to hold before rebalancing
    skip_recent: int = 21      # days to skip at end of lookback (avoid reversal)
    top_pct: float = 0.2       # fraction of universe to go long/short
    direction: DirectionType = "long_short"
    cost_bps: float = 5.0      # one-way transaction cost in basis points
    # MA cross params
    ma_fast: int = 50
    ma_slow: int = 200
    # RSI params
    rsi_window: int = 14
    rsi_lo: float = 30.0
    rsi_hi: float = 70.0
    # BB params
    bb_window: int = 20
    bb_std: float = 2.0
    # vol-adj mom params
    vol_window: int = 63


@dataclass
class BacktestResult:
    sharpe: float
    annual_return: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    equity_curve: pd.Series = field(repr=False, default_factory=pd.Series)
    params: BacktestParams = field(repr=False, default_factory=BacktestParams)


def _compute_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _compute_bb_position(close: pd.Series, window: int, n_std: float) -> pd.Series:
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return (close - lower) / (upper - lower + 1e-9) - 0.5  # centered at 0


def _signal_matrix(prices: pd.DataFrame, p: BacktestParams) -> pd.DataFrame:
    """
    Returns a DataFrame of raw signals (tickers as columns, dates as index).
    Higher = more bullish. Thresholding / ranking is done in run_backtest.
    """
    adj = prices  # already adjusted via yfinance auto_adjust

    if p.strategy == "momentum":
        end_ret = adj.pct_change(p.lookback - p.skip_recent)
        if p.skip_recent > 0:
            end_ret = end_ret.shift(p.skip_recent)
        return end_ret

    if p.strategy == "vol_adj_mom":
        mom = adj.pct_change(p.lookback - p.skip_recent)
        if p.skip_recent > 0:
            mom = mom.shift(p.skip_recent)
        vol = adj.pct_change().rolling(p.vol_window).std() * np.sqrt(252) + 1e-9
        return mom / vol

    if p.strategy == "ma_cross":
        # lookback = fast MA window; slow = 4x fast (sweep explores many ratios)
        fast_w = max(p.lookback, 5)
        slow_w = max(fast_w * 4, fast_w + 20)
        fast = adj.rolling(fast_w).mean()
        slow = adj.rolling(slow_w).mean()
        return (fast / slow - 1)  # positive → bullish

    if p.strategy == "rsi_reversal":
        # lookback drives RSI window
        return adj.apply(lambda col: _compute_rsi(col, p.lookback) * -1)

    if p.strategy == "bb_reversion":
        return adj.apply(lambda col: _compute_bb_position(col, p.bb_window, p.bb_std) * -1)

    raise ValueError(f"Unknown strategy: {p.strategy}")


def _build_weights(signal: pd.Series, p: BacktestParams) -> pd.Series:
    """Convert a cross-sectional signal slice into portfolio weights."""
    valid = signal.dropna()
    if valid.empty:
        return pd.Series(0.0, index=signal.index)

    n = max(1, int(len(valid) * p.top_pct))

    if p.direction in ("long_only", "long_short"):
        long_tickers = valid.nlargest(n).index
    else:
        long_tickers = pd.Index([])

    if p.direction in ("short_only", "long_short"):
        short_tickers = valid.nsmallest(n).index
    else:
        short_tickers = pd.Index([])

    w = pd.Series(0.0, index=signal.index)
    if len(long_tickers):
        w[long_tickers] = 1.0 / len(long_tickers)
    if len(short_tickers):
        w[short_tickers] -= 1.0 / len(short_tickers)
    return w


def run_backtest(prices_wide: pd.DataFrame, p: BacktestParams) -> BacktestResult:
    """
    prices_wide: DataFrame with dates as index, tickers as columns (close prices).
    Returns BacktestResult with performance metrics and equity curve.
    """
    daily_ret = prices_wide.pct_change()
    signal_matrix = _signal_matrix(prices_wide, p)

    cost = p.cost_bps / 10_000

    # Rebalance every p.holding days
    rebal_dates = prices_wide.index[:: p.holding]

    portfolio_returns: list[float] = []
    dates: list = []
    prev_weights = pd.Series(0.0, index=prices_wide.columns)

    for i, rebal_date in enumerate(rebal_dates[:-1]):
        next_rebal = rebal_dates[i + 1]
        signal = signal_matrix.loc[rebal_date]
        weights = _build_weights(signal, p)

        # Transaction cost on turnover
        turnover = (weights - prev_weights).abs().sum() / 2
        tc = turnover * cost

        # Returns during holding period
        hold_slice = daily_ret.loc[
            (daily_ret.index > rebal_date) & (daily_ret.index <= next_rebal)
        ]
        for date, row in hold_slice.iterrows():
            gross = (weights * row).sum()
            portfolio_returns.append(gross - tc if date == hold_slice.index[0] else gross)
            dates.append(date)
            tc = 0.0  # cost only applied on first day of holding period

        prev_weights = weights

    if not portfolio_returns:
        return BacktestResult(0, 0, 0, 0, 0, pd.Series(), p)

    ret_series = pd.Series(portfolio_returns, index=dates)
    equity = (1 + ret_series).cumprod()

    n_years = len(ret_series) / 252
    annual_return = equity.iloc[-1] ** (1 / max(n_years, 0.1)) - 1
    vol = ret_series.std() * np.sqrt(252)
    sharpe = annual_return / vol if vol > 0 else 0.0

    roll_max = equity.cummax()
    drawdown = (equity / roll_max - 1)
    max_dd = drawdown.min()

    win_rate = float((ret_series > 0).mean())

    return BacktestResult(
        sharpe=round(sharpe, 4),
        annual_return=round(annual_return, 4),
        max_drawdown=round(max_dd, 4),
        win_rate=round(win_rate, 4),
        n_trades=len(rebal_dates),
        equity_curve=equity,
        params=p,
    )


def load_prices_wide(db_path: str, start: str = "2010-01-01", end: str | None = None) -> pd.DataFrame:
    """Load close prices from DuckDB into a wide DataFrame (dates × tickers)."""
    import duckdb
    con = duckdb.connect(db_path, read_only=True)
    where = f"date >= '{start}'"
    if end:
        where += f" AND date <= '{end}'"
    df = con.execute(f"""
        SELECT date, ticker, close FROM prices
        WHERE {where}
        ORDER BY date, ticker
    """).df()
    con.close()
    wide = df.pivot(index="date", columns="ticker", values="close")
    wide.index = pd.to_datetime(wide.index)
    return wide
