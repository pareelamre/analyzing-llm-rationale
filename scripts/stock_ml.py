#!/usr/bin/env python3
"""ML-based signal discovery with walk-forward OOS validation.

Trains a gradient-boosted classifier to predict next-period stock direction
using technical features. Walk-forward: train on past N years, test on next year,
roll forward annually.

Usage:
    python scripts/stock_ml.py
    python scripts/stock_ml.py --holding 5 --train-years 5 --out results/stock_ml/
    python scripts/stock_ml.py --feature-importance   # print SHAP values

Output:
    results/stock_ml/wf_equity.csv    -- walk-forward equity curve
    results/stock_ml/wf_summary.json  -- annualized Sharpe, return, DD per fold
    results/stock_ml/feature_imp.csv  -- mean absolute feature importance
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DB_PATH       = str(ROOT / "data" / "stock_prices.duckdb")
FUND_DB_PATH  = str(ROOT / "data" / "stock_fundamentals.duckdb")
EDGAR_DB_PATH = str(ROOT / "data" / "stock_edgar.duckdb")


def load_fundamental_features(fund_db: str, prices_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Build a (ticker, date) → fundamental feature matrix using point-in-time data.

    earnings_surprise features are forward-filled from each earnings date, so on
    any given date a stock carries its most recent reported surprise — no look-ahead.
    Snapshot ratios (P/B, ROE, etc.) are as-of-today, used as a static cross-sectional
    signal (valid for recent data, stale for historical folds — flag in results).
    """
    import duckdb, os
    if not os.path.exists(fund_db):
        return pd.DataFrame()

    con = duckdb.connect(fund_db, read_only=True)
    es  = con.execute("SELECT ticker, date, surprise_pct FROM earnings_surprises").df()
    qf  = con.execute("""
        SELECT ticker, date, gross_margin, net_margin,
               revenue, LAG(revenue, 4) OVER (PARTITION BY ticker ORDER BY date) AS rev_4q_ago
        FROM quarterly_financials
    """).df()
    snap = con.execute("""
        SELECT ticker, price_to_book, roe, debt_to_equity, revenue_growth, forward_pe
        FROM snapshot_ratios
    """).df()
    con.close()

    all_dates = pd.to_datetime(prices_wide.index)
    rows = []

    for ticker in prices_wide.columns:
        # --- earnings surprise: forward-fill from report date ---
        tes = es[es["ticker"] == ticker].copy()
        if not tes.empty:
            tes["date"] = pd.to_datetime(tes["date"])
            tes = tes.sort_values("date").set_index("date")
            # reindex to all trading dates, forward-fill
            surp_ts = tes["surprise_pct"].reindex(all_dates, method="ffill")
            # 4-quarter EPS momentum: rolling mean of surprise over last 4 reports
            surp_roll = tes["surprise_pct"].rolling(4).mean().reindex(all_dates, method="ffill")
        else:
            surp_ts   = pd.Series(np.nan, index=all_dates)
            surp_roll = pd.Series(np.nan, index=all_dates)

        # --- quarterly financials: forward-fill ---
        tqf = qf[qf["ticker"] == ticker].copy()
        if not tqf.empty:
            tqf["date"] = pd.to_datetime(tqf["date"])
            tqf = tqf.sort_values("date").set_index("date")
            gm_ts  = tqf["gross_margin"].reindex(all_dates, method="ffill")
            nm_ts  = tqf["net_margin"].reindex(all_dates, method="ffill")
            rev_growth_ts = (
                (tqf["revenue"] / (tqf["rev_4q_ago"] + 1e-9) - 1)
                .reindex(all_dates, method="ffill")
            )
        else:
            gm_ts = nm_ts = rev_growth_ts = pd.Series(np.nan, index=all_dates)

        # --- snapshot ratios: static cross-section ---
        srow = snap[snap["ticker"] == ticker]
        ptb = float(srow["price_to_book"].iloc[0])  if not srow.empty else np.nan
        roe = float(srow["roe"].iloc[0])             if not srow.empty else np.nan
        dte = float(srow["debt_to_equity"].iloc[0])  if not srow.empty else np.nan
        fpe = float(srow["forward_pe"].iloc[0])      if not srow.empty else np.nan

        df = pd.DataFrame({
            "ticker":           ticker,
            "date":             all_dates,
            "eps_surprise":     surp_ts.values,
            "eps_surp_4q_avg":  surp_roll.values,
            "gross_margin":     gm_ts.values,
            "net_margin":       nm_ts.values,
            "revenue_growth_yoy": rev_growth_ts.values,
            "price_to_book":    ptb,
            "roe":              roe,
            "debt_to_equity":   dte,
            "forward_pe":       fpe,
        })
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_edgar_features(edgar_db: str, prices_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Build point-in-time fundamental features from SEC EDGAR filing data.
    Uses vectorized as-of joins — no per-date Python loops.

    Features (all strictly point-in-time via filed date):
      trailing_eps     — rolling 4-quarter sum of quarterly EPS
      pe_ratio         — price / trailing_eps
      gross_margin     — gross_profit / revenue
      net_margin       — net_income / revenue
      roe              — net_income / equity (trailing 4Q / spot equity)
      debt_to_equity   — long_term_debt / equity
      revenue_growth   — YoY: (current_rev - prior_year_rev) / |prior_year_rev|
      eps_surprise     — (current_Q_EPS - same_Q_prior_year_EPS) / |prior|
    """
    import duckdb as _duckdb
    con = _duckdb.connect(edgar_db, read_only=True)
    facts = con.execute("""
        SELECT ticker, metric, filed, period_end, fiscal_period, value
        FROM facts
        WHERE fiscal_period IN ('Q1','Q2','Q3','Q4','FY')
          AND form IN ('10-Q','10-K')
        ORDER BY ticker, metric, filed
    """).df()
    con.close()

    facts["filed"]      = pd.to_datetime(facts["filed"])
    facts["period_end"] = pd.to_datetime(facts["period_end"])
    all_dates           = pd.DatetimeIndex(pd.to_datetime(prices_wide.index))
    target_df           = pd.DataFrame({"date": all_dates})

    def ffill_to_dates(series_by_filed: pd.Series) -> pd.Series:
        """Forward-fill a filed-date indexed series to all trading dates."""
        s = series_by_filed.sort_index()
        s = s[~s.index.duplicated(keep="last")]
        return s.reindex(all_dates, method="ffill")

    def rolling4q(metric_df: pd.DataFrame) -> pd.Series:
        """
        Compute rolling 4-quarter sum, forward-filled to trading dates.
        Deduplicates on (filed, period_end) to avoid amendment double-counting,
        then keeps at most one entry per fiscal quarter (latest filing wins).
        """
        df = (metric_df
              .sort_values(["period_end", "filed"])
              .drop_duplicates("period_end", keep="last")   # latest amendment wins
              .sort_values("filed"))
        if len(df) < 4:
            return pd.Series(np.nan, index=all_dates)
        rolling = df.set_index("filed")["value"].rolling(4, min_periods=4).sum()
        return ffill_to_dates(rolling)

    def spot_metric(metric_df: pd.DataFrame) -> pd.Series:
        df = (metric_df
              .sort_values(["period_end", "filed"])
              .drop_duplicates("period_end", keep="last")
              .sort_values("filed"))
        if df.empty:
            return pd.Series(np.nan, index=all_dates)
        return ffill_to_dates(df.set_index("filed")["value"])

    rows = []
    for ticker, tdf in facts.groupby("ticker"):
        if ticker not in prices_wide.columns:
            continue
        prices = prices_wide[ticker].reindex(all_dates)

        eps_df  = tdf[tdf["metric"] == "eps_diluted"]
        rev_df  = tdf[tdf["metric"] == "revenue"]
        gp_df   = tdf[tdf["metric"] == "gross_profit"]
        ni_df   = tdf[tdf["metric"] == "net_income"]
        eq_df   = tdf[tdf["metric"] == "equity"]
        ltd_df  = tdf[tdf["metric"] == "long_term_debt"]

        trailing_eps = rolling4q(eps_df)
        pe_ratio     = (prices / trailing_eps.replace(0, np.nan)).clip(-100, 200)

        rev_spot = spot_metric(rev_df)
        gp_spot  = spot_metric(gp_df)
        ni_spot  = spot_metric(ni_df)
        eq_spot  = spot_metric(eq_df)
        ltd_spot = spot_metric(ltd_df)

        gross_margin   = (gp_spot  / rev_spot.replace(0, np.nan)).clip(-1, 1)
        net_margin     = (ni_spot  / rev_spot.replace(0, np.nan)).clip(-1, 1)
        ni_trail       = rolling4q(ni_df)
        roe            = (ni_trail / eq_spot.replace(0, np.nan)).clip(-2, 2)
        debt_to_equity = (ltd_spot / eq_spot.replace(0, np.nan)).clip(0, 20)

        # Revenue growth YoY: shift rev series by ~252 trading days
        rev_growth = pd.Series(np.nan, index=all_dates)
        if not rev_df.empty:
            rv = spot_metric(rev_df)
            rv_1y = rv.shift(252)
            rev_growth = ((rv - rv_1y) / rv_1y.abs().replace(0, np.nan)).clip(-1, 5)

        # EPS surprise: current Q EPS vs same-quarter prior-year EPS
        eps_surprise = pd.Series(np.nan, index=all_dates)
        eps_clean = (eps_df
                     .sort_values(["period_end", "filed"])
                     .drop_duplicates("period_end", keep="last")
                     .sort_values("period_end"))
        if len(eps_clean) >= 5:
            eps_clean["qtr"] = eps_clean["period_end"].dt.quarter
            # align current quarter with same quarter 4 rows back
            eps_clean["eps_yoy"] = (
                (eps_clean["value"] - eps_clean["value"].shift(4)) /
                eps_clean["value"].shift(4).abs().replace(0, np.nan)
            ).clip(-5, 5)
            surp_ts = ffill_to_dates(
                eps_clean.set_index("filed")["eps_yoy"]
                         .sort_index()
            )
            eps_surprise = surp_ts

        rows.append(pd.DataFrame({
            "ticker":         ticker,
            "date":           all_dates,
            "trailing_eps":   trailing_eps.values,
            "pe_ratio":       pe_ratio.values,
            "gross_margin":   gross_margin.values,
            "net_margin":     net_margin.values,
            "roe":            roe.values,
            "debt_to_equity": debt_to_equity.values,
            "revenue_growth": rev_growth.values,
            "eps_surprise":   eps_surprise.values,
        }))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compute_features(prices_wide: pd.DataFrame, fund_db: str | None = None,
                     edgar_db: str | None = None) -> pd.DataFrame:
    """Compute technical + fundamental features. Returns long-format DataFrame."""
    rows = []
    for ticker in prices_wide.columns:
        close = prices_wide[ticker].dropna()
        if len(close) < 300:
            continue

        df = pd.DataFrame({"close": close})
        # Momentum features
        for w in [5, 10, 21, 42, 63, 126, 252]:
            df[f"mom_{w}d"] = close.pct_change(w)
        # Volatility
        df["vol_21d"] = close.pct_change().rolling(21).std() * np.sqrt(252)
        df["vol_63d"] = close.pct_change().rolling(63).std() * np.sqrt(252)
        # Vol-adj momentum
        df["vol_adj_mom_63"] = df["mom_63d"] / (df["vol_21d"] + 1e-9)
        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        df["rsi_14"] = 100 - 100 / (1 + gain / (loss + 1e-9))
        # Bollinger position
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["bb_pos"] = (close - (ma20 - 2 * std20)) / (4 * std20 + 1e-9)
        # MA ratio
        df["ma_ratio_50_200"] = close.rolling(50).mean() / (close.rolling(200).mean() + 1e-9) - 1
        # Volume ratio (if available — proxy: NaN-safe)
        df["ticker"] = ticker
        df = df.reset_index().rename(columns={"index": "date"})
        rows.append(df)

    tech_df = pd.concat(rows, ignore_index=True)
    tech_df["date"] = pd.to_datetime(tech_df["date"])

    if fund_db and Path(fund_db).exists():
        print("  Merging yfinance fundamental features …")
        fund_df = load_fundamental_features(fund_db, prices_wide)
        if not fund_df.empty:
            fund_df["date"] = pd.to_datetime(fund_df["date"])
            tech_df = tech_df.merge(fund_df, on=["ticker", "date"], how="left")

    if edgar_db and Path(edgar_db).exists():
        print("  Merging EDGAR point-in-time features …")
        edgar_df = load_edgar_features(edgar_db, prices_wide)
        if not edgar_df.empty:
            edgar_df["date"] = pd.to_datetime(edgar_df["date"])
            tech_df = tech_df.merge(edgar_df, on=["ticker", "date"], how="left",
                                    suffixes=("", "_edgar"))

    return tech_df


def walk_forward(
    features_df: pd.DataFrame,
    prices_wide: pd.DataFrame,
    holding: int,
    train_years: int,
    top_pct: float,
    cost_bps: float,
) -> tuple[pd.Series, list[dict]]:
    """
    Annual walk-forward OOS evaluation.
    Returns (equity_curve, fold_summaries).
    """
    cost = cost_bps / 10_000
    feature_cols = [c for c in features_df.columns if c not in ("date", "ticker", "target")]
    all_dates = sorted(features_df["date"].unique())
    years = sorted(set(pd.Timestamp(d).year for d in all_dates))

    equity = pd.Series(dtype=float)
    folds: list[dict] = []

    for test_year in years[train_years:]:
        train_end = pd.Timestamp(f"{test_year - 1}-12-31")
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year}-12-31")

        dates_ts = pd.to_datetime(features_df["date"])
        train_mask = dates_ts <= train_end
        test_mask = (dates_ts >= test_start) & (dates_ts <= test_end)
        if train_mask.sum() < 500 or test_mask.sum() < 50:
            continue

        # Target: next holding-period return direction
        features_df = features_df.copy()
        features_df["target"] = np.nan
        for ticker in features_df["ticker"].unique():
            idx = features_df["ticker"] == ticker
            if ticker not in prices_wide.columns:
                continue
            fwd = prices_wide[ticker].pct_change(holding).shift(-holding)
            fwd_aligned = fwd.reindex(features_df.loc[idx, "date"].values)
            features_df.loc[idx, "target"] = (fwd_aligned.values > 0).astype(float)

        train_df = features_df[train_mask].dropna(subset=["target"]).copy()
        test_df  = features_df[test_mask].copy()

        # Drop columns that are >95% NaN in the training window (no signal)
        null_frac = train_df[feature_cols].isnull().mean()
        active_cols = [c for c in feature_cols if null_frac[c] < 0.95]

        # Impute remaining NaNs with training median; fall back to 0 if median is NaN
        col_medians = train_df[active_cols].median().fillna(0)
        train_df[active_cols] = train_df[active_cols].fillna(col_medians)
        test_df[active_cols]  = test_df[active_cols].fillna(col_medians)

        # Fill any leftover NaN in test columns not present in active_cols with 0
        for c in feature_cols:
            if c not in active_cols:
                train_df[c] = 0.0
                test_df[c]  = 0.0

        train_df = train_df.dropna(subset=active_cols + ["target"])
        test_df  = test_df.dropna(subset=active_cols)

        if len(train_df) < 100:
            continue

        X_train = train_df[active_cols].values
        y_train = train_df["target"].values
        X_test = test_df[active_cols].values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8,
            random_state=42
        )
        clf.fit(X_train, y_train)

        test_df = test_df.copy()
        test_df["score"] = clf.predict_proba(X_test)[:, 1]

        # Simulate: each rebal date, rank by score → long top_pct
        fold_rets = []
        fold_dates = []
        rebal_dates = pd.date_range(test_start, test_end, freq=f"{holding}D")
        prev_longs = set()

        for rebal_dt in rebal_dates:
            slice_df = test_df[test_df["date"] == rebal_dt.date()]
            if slice_df.empty:
                continue
            n = max(1, int(len(slice_df) * top_pct))
            longs = set(slice_df.nlargest(n, "score")["ticker"])

            # Transaction cost on turnover
            entered = longs - prev_longs
            exited = prev_longs - longs
            turnover = (len(entered) + len(exited)) / max(len(longs), 1)
            tc = turnover * cost

            # Holding period returns
            next_rebal = rebal_dt + pd.Timedelta(days=holding)
            for ticker in longs:
                if ticker not in prices_wide.columns:
                    continue
                p0 = prices_wide[ticker].asof(rebal_dt)
                p1 = prices_wide[ticker].asof(next_rebal)
                if pd.isna(p0) or pd.isna(p1) or p0 == 0:
                    continue
                ret = p1 / p0 - 1
                fold_rets.append(ret / max(len(longs), 1) - tc / max(len(longs), 1))
                fold_dates.append(rebal_dt)
            prev_longs = longs

        if not fold_rets:
            continue

        fold_series = pd.Series(fold_rets, index=fold_dates).groupby(level=0).sum()
        if equity.empty:
            equity = (1 + fold_series).cumprod()
        else:
            last_val = equity.iloc[-1]
            eq_chunk = last_val * (1 + fold_series).cumprod()
            equity = pd.concat([equity, eq_chunk])

        n_years_fold = (test_end - test_start).days / 365
        vol = fold_series.std() * np.sqrt(252 / holding)
        ann_ret = (1 + fold_series.mean()) ** (252 / holding) - 1
        sharpe = ann_ret / vol if vol > 0 else 0.0

        # Feature importance
        imp = dict(zip(active_cols, clf.feature_importances_))

        # AUC on test
        test_with_target = test_df.dropna(subset=["target"])
        auc = 0.5
        if len(test_with_target) > 10:
            try:
                auc = roc_auc_score(test_with_target["target"], test_with_target["score"])
            except Exception:
                pass

        folds.append({
            "year": test_year,
            "sharpe": round(sharpe, 3),
            "annual_return": round(ann_ret, 3),
            "auc": round(auc, 3),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "feature_importance": {k: round(v, 4) for k, v in sorted(imp.items(), key=lambda x: -x[1])[:10]},
        })
        print(f"  {test_year}: Sharpe={sharpe:.2f}  AnnRet={ann_ret:.1%}  AUC={auc:.3f}  n_test={len(test_df)}")

    return equity, folds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--holding", type=int, default=21, help="Holding period in days")
    ap.add_argument("--train-years", type=int, default=4, help="Training window in years")
    ap.add_argument("--top-pct", type=float, default=0.2, help="Top fraction to go long")
    ap.add_argument("--cost-bps", type=float, default=5.0, help="Transaction cost bps per side")
    ap.add_argument("--fund-db", default=FUND_DB_PATH, help="Fundamentals DB path")
    ap.add_argument("--edgar-db", default=EDGAR_DB_PATH, help="EDGAR fundamentals DB path")
    ap.add_argument("--no-fundamentals", action="store_true", help="Skip all fundamental features")
    ap.add_argument("--out", default=str(ROOT / "results" / "stock_ml"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from stock_backtest import load_prices_wide

    print("Loading prices …")
    prices = load_prices_wide(args.db)
    print(f"  {len(prices)} dates × {len(prices.columns)} tickers")

    fund_db  = None if args.no_fundamentals else args.fund_db
    edgar_db = None if args.no_fundamentals else args.edgar_db
    if edgar_db and Path(edgar_db).exists():
        print(f"EDGAR features:       {edgar_db}")
    else:
        print("EDGAR features: not found — technical only")
        edgar_db = None

    print("Computing features …")
    feat_df = compute_features(prices, fund_db=fund_db, edgar_db=edgar_db)
    feat_df["date"] = pd.to_datetime(feat_df["date"]).dt.date
    print(f"  Feature matrix: {len(feat_df):,} rows")

    print("Walk-forward OOS …")
    equity, folds = walk_forward(
        feat_df, prices,
        holding=args.holding,
        train_years=args.train_years,
        top_pct=args.top_pct,
        cost_bps=args.cost_bps,
    )

    if not equity.empty:
        equity.to_csv(out_dir / "wf_equity.csv")
        n_years = (equity.index[-1] - equity.index[0]).days / 365
        total_ret = equity.iloc[-1] - 1
        print(f"\nTotal return: {total_ret:.1%} over {n_years:.1f} years")

    (out_dir / "wf_summary.json").write_text(json.dumps(folds, indent=2))
    print(f"Results → {out_dir}/")


if __name__ == "__main__":
    main()
