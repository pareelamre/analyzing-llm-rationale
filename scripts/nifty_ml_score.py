#!/usr/bin/env python3
"""Live scoring for the Nifty 100 GBM momentum model.

No model artefact is saved by the training pipeline, so this script retrains
on all data up to the most recent date with a known 63-day forward return,
then scores the latest available price date.

Usage:
    python scripts/nifty_ml_score.py

Outputs:
    results/nifty_ml/live_scores.csv   — all tickers ranked by P(up in 63d)
    results/nifty_ml/live_top20.txt    — top-20 ticker symbols, one per line
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from nifty_ml import compute_features, load_prices  # noqa: E402

OUT_DIR = ROOT / "results" / "nifty_ml"
HOLDING = 63  # trading days — must match training

# Pipeline column → user-facing output column name
_RENAME = {
    "mom_252d": "mom_12m",
    "mom_126d": "mom_6m",
    "mom_63d":  "mom_3m",
    "vol_adj_mom63": "vol_adj_mom_3m",
}
# Columns to include in live_scores.csv (in order)
_OUT_COLS = ["ticker", "rank", "score",
             "ma_ratio_50_200", "mom_12m", "mom_6m", "mom_3m", "vol_63d"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Nifty 100 prices …")
    prices_wide = load_prices()
    print(f"  {len(prices_wide)} dates × {len(prices_wide.columns)} tickers")

    print("Computing features …")
    feat_df = compute_features(prices_wide, use_fundamentals=False)
    feat_df["date"] = pd.to_datetime(feat_df["date"]).dt.normalize()

    # All column names that are usable features
    feature_cols = [c for c in feat_df.columns
                    if c not in ("date", "ticker", "close", "target")]

    # Forward-return target: 1 if price is higher in HOLDING days, else 0
    feat_df["target"] = np.nan
    for ticker in feat_df["ticker"].unique():
        if ticker not in prices_wide.columns:
            continue
        mask = feat_df["ticker"] == ticker
        fwd = prices_wide[ticker].pct_change(HOLDING).shift(-HOLDING)
        feat_df.loc[mask, "target"] = (
            fwd.reindex(feat_df.loc[mask, "date"].values).values > 0
        ).astype(float)

    score_date = feat_df["date"].max()
    print(f"  Score date: {score_date.date()}  "
          f"Training cutoff: all dates with known {HOLDING}-day target")

    train_df = feat_df[feat_df["target"].notna()].copy()
    score_df = feat_df[feat_df["date"] == score_date].copy()

    if train_df.empty:
        sys.exit("No training data available.")
    if score_df.empty:
        sys.exit("No score-date rows — prices may not be up to date.")

    # Drop near-constant / mostly-null columns
    null_frac = train_df[feature_cols].isnull().mean()
    active_cols = [c for c in feature_cols if null_frac[c] < 0.95]
    col_medians = train_df[active_cols].median().fillna(0)

    train_df[active_cols] = train_df[active_cols].fillna(col_medians)
    score_df[active_cols] = score_df[active_cols].fillna(col_medians)
    for c in set(feature_cols) - set(active_cols):
        train_df[c] = 0.0
        score_df[c] = 0.0

    train_df = train_df.dropna(subset=active_cols + ["target"])
    score_df = score_df.dropna(subset=active_cols)

    print(f"  Training rows: {len(train_df):,}  "
          f"Tickers to score: {len(score_df)}")

    # Train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[active_cols].values)
    y_train = train_df["target"].values

    print("Training GBM …")
    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    clf.fit(X_train, y_train)

    # Score
    X_score = scaler.transform(score_df[active_cols].values)
    score_df = score_df.copy()
    score_df["score"] = clf.predict_proba(X_score)[:, 1]
    score_df = (score_df
                .sort_values("score", ascending=False)
                .reset_index(drop=True))
    score_df["rank"] = score_df.index + 1

    # Rename to user-facing names
    score_df = score_df.rename(columns=_RENAME)

    # live_scores.csv
    out_cols = [c for c in _OUT_COLS if c in score_df.columns]
    csv_path = OUT_DIR / "live_scores.csv"
    score_df[out_cols].to_csv(csv_path, index=False, float_format="%.4f")

    # live_top25.txt
    top20 = score_df["ticker"].head(25).tolist()
    txt_path = OUT_DIR / "live_top25.txt"
    txt_path.write_text("\n".join(top20) + "\n")

    # Summary
    print(f"\nDate scored : {score_date.date()}")
    print(f"Stocks ranked: {len(score_df)}")
    print(f"Top 5        : {', '.join(top20[:5])}")
    print(f"\nOutputs:")
    print(f"  {csv_path}")
    print(f"  {txt_path}")


if __name__ == "__main__":
    main()
