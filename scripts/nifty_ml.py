#!/usr/bin/env python3
"""ML-based signal discovery on Nifty 100 with walk-forward OOS validation.

Trains a gradient-boosted classifier on technical (+ optional yfinance
fundamental) features to predict next-period direction. Walk-forward:
train 4 years, test 1 year, roll annually.

Usage:
    python scripts/nifty_ml.py
    python scripts/nifty_ml.py --holding 63 --cost-bps 10
    python scripts/nifty_ml.py --no-fundamentals

Output: results/nifty_ml/
  wf_equity.csv   -- walk-forward equity curve
  wf_summary.json -- Sharpe / AnnRet / AUC per fold
  feature_imp.csv -- mean feature importances
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent.parent
PRICES_DB = str(ROOT / "data" / "nifty_prices.duckdb")
FUND_DB   = str(ROOT / "data" / "nifty_fundamentals.duckdb")


def load_prices() -> pd.DataFrame:
    con = duckdb.connect(PRICES_DB, read_only=True)
    df = con.execute("SELECT ticker, date, close FROM prices ORDER BY ticker, date").df()
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="ticker", values="close").sort_index()


def load_fundamental_features(prices_wide: pd.DataFrame) -> pd.DataFrame:
    if not Path(FUND_DB).exists():
        return pd.DataFrame()
    con = duckdb.connect(FUND_DB, read_only=True)
    es   = con.execute("SELECT ticker, announce_date, surprise_pct FROM earnings_surprises").df()
    qf   = con.execute("SELECT ticker, period_end, gross_margin, net_margin FROM quarterly_financials").df()
    snap = con.execute("SELECT ticker, trailing_pe, price_to_book, roe, revenue_growth FROM snapshot_ratios").df()
    con.close()

    all_dates = prices_wide.index
    rows = []
    for ticker in prices_wide.columns:
        # EPS surprise: forward-fill from announce_date
        tes = es[es["ticker"] == ticker].copy()
        if not tes.empty:
            tes["announce_date"] = pd.to_datetime(tes["announce_date"])
            tes = tes.sort_values("announce_date").set_index("announce_date")
            surp_ts   = tes["surprise_pct"].reindex(all_dates, method="ffill")
            surp_roll = tes["surprise_pct"].rolling(4).mean().reindex(all_dates, method="ffill")
        else:
            surp_ts = surp_roll = pd.Series(np.nan, index=all_dates)

        # Quarterly financials: forward-fill from period_end
        tqf = qf[qf["ticker"] == ticker].copy()
        if not tqf.empty:
            tqf["period_end"] = pd.to_datetime(tqf["period_end"])
            tqf = tqf.sort_values("period_end").set_index("period_end")
            gm_ts = tqf["gross_margin"].reindex(all_dates, method="ffill")
            nm_ts = tqf["net_margin"].reindex(all_dates, method="ffill")
        else:
            gm_ts = nm_ts = pd.Series(np.nan, index=all_dates)

        # Snapshot ratios (static cross-section — valid only for recent folds)
        srow = snap[snap["ticker"] == ticker]
        ptb  = float(srow["price_to_book"].iloc[0]) if not srow.empty else np.nan
        roe  = float(srow["roe"].iloc[0])            if not srow.empty else np.nan
        rg   = float(srow["revenue_growth"].iloc[0]) if not srow.empty else np.nan

        rows.append(pd.DataFrame({
            "ticker":           ticker,
            "date":             all_dates,
            "eps_surprise":     surp_ts.values,
            "eps_surp_4q_avg":  surp_roll.values,
            "gross_margin":     gm_ts.values,
            "net_margin":       nm_ts.values,
            "price_to_book":    ptb,
            "roe":              roe,
            "revenue_growth_snap": rg,
        }))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compute_features(prices_wide: pd.DataFrame, use_fundamentals: bool = True) -> pd.DataFrame:
    rows = []
    for ticker in prices_wide.columns:
        close = prices_wide[ticker].dropna()
        if len(close) < 300:
            continue
        df = pd.DataFrame({"close": close})
        for w in [5, 10, 21, 42, 63, 126, 252]:
            df[f"mom_{w}d"] = close.pct_change(w)
        df["vol_21d"]       = close.pct_change().rolling(21).std() * np.sqrt(252)
        df["vol_63d"]       = close.pct_change().rolling(63).std() * np.sqrt(252)
        df["vol_adj_mom63"] = df["mom_63d"] / (df["vol_21d"] + 1e-9)
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        df["rsi_14"]         = 100 - 100 / (1 + gain / (loss + 1e-9))
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["bb_pos"]         = (close - (ma20 - 2 * std20)) / (4 * std20 + 1e-9)
        df["ma_ratio_50_200"] = close.rolling(50).mean() / (close.rolling(200).mean() + 1e-9) - 1
        df["ticker"] = ticker
        rows.append(df.reset_index().rename(columns={"index": "date"}))

    tech_df = pd.concat(rows, ignore_index=True)
    tech_df["date"] = pd.to_datetime(tech_df["date"])

    if use_fundamentals:
        print("  Merging yfinance fundamental features …")
        fund_df = load_fundamental_features(prices_wide)
        if not fund_df.empty:
            fund_df["date"] = pd.to_datetime(fund_df["date"])
            tech_df = tech_df.merge(fund_df, on=["ticker", "date"], how="left")

    return tech_df


def walk_forward(
    features_df: pd.DataFrame,
    prices_wide: pd.DataFrame,
    holding: int,
    train_years: int,
    top_pct: float,
    cost_bps: float,
) -> tuple[pd.Series, list[dict]]:
    cost = cost_bps / 10_000
    feature_cols = [c for c in features_df.columns if c not in ("date", "ticker", "close", "target")]
    all_dates = sorted(features_df["date"].unique())
    years = sorted(set(pd.Timestamp(d).year for d in all_dates))

    equity: pd.Series = pd.Series(dtype=float)
    folds: list[dict] = []
    all_importances: list[dict] = []

    for test_year in years[train_years:]:
        train_end   = pd.Timestamp(f"{test_year - 1}-12-31")
        test_start  = pd.Timestamp(f"{test_year}-01-01")
        test_end    = pd.Timestamp(f"{test_year}-12-31")

        dates_ts = pd.to_datetime(features_df["date"])
        train_mask = dates_ts <= train_end
        test_mask  = (dates_ts >= test_start) & (dates_ts <= test_end)
        if train_mask.sum() < 500 or test_mask.sum() < 50:
            continue

        # Compute forward return target
        fd = features_df.copy()
        fd["target"] = np.nan
        for ticker in fd["ticker"].unique():
            idx = fd["ticker"] == ticker
            if ticker not in prices_wide.columns:
                continue
            fwd = prices_wide[ticker].pct_change(holding).shift(-holding)
            fd.loc[idx, "target"] = (
                fwd.reindex(fd.loc[idx, "date"].values).values > 0
            ).astype(float)

        train_df = fd[train_mask].dropna(subset=["target"]).copy()
        test_df  = fd[test_mask].copy()

        null_frac   = train_df[feature_cols].isnull().mean()
        active_cols = [c for c in feature_cols if null_frac[c] < 0.95]
        col_medians = train_df[active_cols].median().fillna(0)
        train_df[active_cols] = train_df[active_cols].fillna(col_medians)
        test_df[active_cols]  = test_df[active_cols].fillna(col_medians)
        for c in feature_cols:
            if c not in active_cols:
                train_df[c] = 0.0
                test_df[c]  = 0.0

        train_df = train_df.dropna(subset=active_cols + ["target"])
        test_df  = test_df.dropna(subset=active_cols)
        if len(train_df) < 100:
            continue

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(train_df[active_cols].values)
        X_test  = scaler.transform(test_df[active_cols].values)
        y_train = train_df["target"].values

        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        clf.fit(X_train, y_train)
        test_df = test_df.copy()
        test_df["score"] = clf.predict_proba(X_test)[:, 1]

        fold_rets, fold_dates = [], []
        prev_longs: set[str] = set()
        for rebal_dt in pd.date_range(test_start, test_end, freq=f"{holding}D"):
            slice_df = test_df[test_df["date"] == rebal_dt.normalize()]
            if slice_df.empty:
                # try nearest trading date
                candidates = test_df[test_df["date"] <= rebal_dt]
                if candidates.empty:
                    continue
                nearest = candidates["date"].max()
                slice_df = test_df[test_df["date"] == nearest]
            if slice_df.empty:
                continue
            n = max(1, int(len(slice_df) * top_pct))
            longs = set(slice_df.nlargest(n, "score")["ticker"])
            turnover = (len(longs - prev_longs) + len(prev_longs - longs)) / max(len(longs), 1)
            tc = turnover * cost
            next_rebal = rebal_dt + pd.Timedelta(days=holding)
            for ticker in longs:
                if ticker not in prices_wide.columns:
                    continue
                p0 = prices_wide[ticker].asof(rebal_dt)
                p1 = prices_wide[ticker].asof(next_rebal)
                if pd.isna(p0) or pd.isna(p1) or p0 == 0:
                    continue
                fold_rets.append((p1 / p0 - 1) / max(len(longs), 1) - tc / max(len(longs), 1))
                fold_dates.append(rebal_dt)
            prev_longs = longs

        if not fold_rets:
            continue

        fold_series = pd.Series(fold_rets, index=fold_dates).groupby(level=0).sum()
        if equity.empty:
            equity = (1 + fold_series).cumprod()
        else:
            equity = pd.concat([equity, equity.iloc[-1] * (1 + fold_series).cumprod()])

        vol     = fold_series.std() * np.sqrt(252 / holding)
        ann_ret = (1 + fold_series.mean()) ** (252 / holding) - 1
        sharpe  = ann_ret / vol if vol > 0 else 0.0

        imp = dict(zip(active_cols, clf.feature_importances_))
        all_importances.append(imp)

        tgt = test_df.dropna(subset=["target"])
        auc = 0.5
        if len(tgt) > 10:
            try:
                auc = roc_auc_score(tgt["target"], tgt["score"])
            except Exception:
                pass

        folds.append({
            "year": int(test_year),
            "sharpe": round(sharpe, 3),
            "annual_return": round(ann_ret, 3),
            "auc": round(auc, 3),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "top_features": {k: round(v, 4) for k, v in sorted(imp.items(), key=lambda x: -x[1])[:8]},
        })
        print(f"  {test_year}: Sharpe={sharpe:.2f}  AnnRet={ann_ret:.1%}  AUC={auc:.3f}  n_test={len(test_df)}")

    return equity, folds, all_importances


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holding",       type=int,   default=63)
    ap.add_argument("--train-years",   type=int,   default=4)
    ap.add_argument("--top-pct",       type=float, default=0.20)
    ap.add_argument("--cost-bps",      type=float, default=10.0)
    ap.add_argument("--no-fundamentals", action="store_true")
    ap.add_argument("--out",           default=str(ROOT / "results" / "nifty_ml"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Nifty 100 prices …")
    prices_wide = load_prices()
    print(f"  {len(prices_wide)} dates × {len(prices_wide.columns)} tickers")

    print("Computing features …")
    feat_df = compute_features(prices_wide, use_fundamentals=not args.no_fundamentals)
    feat_df["date"] = pd.to_datetime(feat_df["date"]).dt.normalize()
    print(f"  Feature matrix: {len(feat_df):,} rows, {feat_df.shape[1]} columns")

    print(f"\nWalk-forward OOS  holding={args.holding}d  train={args.train_years}yr  "
          f"top={args.top_pct:.0%}  cost={args.cost_bps}bps")
    equity, folds, imps = walk_forward(
        feat_df, prices_wide,
        holding=args.holding, train_years=args.train_years,
        top_pct=args.top_pct, cost_bps=args.cost_bps,
    )

    if not folds:
        print("No folds generated.")
        return

    avg_sharpe = np.mean([f["sharpe"] for f in folds])
    avg_auc    = np.mean([f["auc"]    for f in folds])
    pos_folds  = sum(1 for f in folds if f["annual_return"] > 0)
    print(f"\nAvg Sharpe: {avg_sharpe:.2f}  Avg AUC: {avg_auc:.3f}  "
          f"Positive folds: {pos_folds}/{len(folds)}")

    if not equity.empty:
        n_years   = (equity.index[-1] - equity.index[0]).days / 365
        total_ret = equity.iloc[-1] - 1
        print(f"Total OOS return: {total_ret:.1%} over {n_years:.1f} years")
        equity.to_csv(out_dir / "wf_equity.csv")

    (out_dir / "wf_summary.json").write_text(json.dumps(folds, indent=2))

    # Aggregate feature importances
    if imps:
        all_keys = set(k for d in imps for k in d)
        imp_df = pd.DataFrame([{k: d.get(k, 0) for k in all_keys} for d in imps])
        imp_mean = imp_df.mean().sort_values(ascending=False)
        imp_mean.to_csv(out_dir / "feature_imp.csv", header=["importance"])
        print("\nTop features:")
        print(imp_mean.head(10).to_string())

    print(f"\nResults → {out_dir}/")


if __name__ == "__main__":
    main()
