#!/usr/bin/env python3
"""
ML V2 Training Pipeline - PRADO9_EVO

Trains 24 models with enhanced features:
- 4 horizon models (1d, 3d, 5d, 10d) with 24 features each
- 20 regime models (5 regimes × 4 horizons) with 24 features each

Key Improvements:
1. 24 ML features (vs 9 original)
2. Horizon-specific labels (truly divergent)
3. Regime-conditioned labels
4. Saves to ~/.prado/models/<symbol>/ml_v2/

Usage:
    python train_ml_v2.py QQQ
    python train_ml_v2.py SPY --start 2015-01-01 --end 2024-12-31
"""

import os
import argparse
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
import joblib

from src.afml_system.ml.feature_builder_v2 import FeatureBuilderV2
from src.afml_system.ml.target_builder_v2 import TargetBuilderV2


def load_data(symbol: str, start_date: str = '2010-01-01', end_date: str = None) -> pd.DataFrame:
    """Load OHLCV data from yfinance"""
    print(f"\n{'='*80}")
    print(f"Loading data for {symbol}")
    print(f"{'='*80}")

    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    data = yf.download(symbol, start=start_date, end=end_date, progress=False)

    # Normalize column names
    data.columns = [str(col[0]).lower() if isinstance(col, tuple) else str(col).lower() for col in data.columns]

    print(f"Loaded {len(data)} bars from {data.index[0]} to {data.index[-1]}")
    return data


def train_horizon_model(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: str,
    symbol: str,
    save_dir: Path
) -> dict:
    """
    Train a single horizon model.

    Args:
        X: Feature DataFrame
        y: Label Series
        horizon: '1d', '3d', '5d', or '10d'
        symbol: Stock symbol
        save_dir: Directory to save model

    Returns:
        Training statistics dict
    """
    print(f"\n--- Training Horizon Model: {horizon} ---")

    # Check label balance
    label_counts = y.value_counts()
    print(f"  Label Distribution: {dict(label_counts)}")

    if len(label_counts) < 2:
        print(f"  ⚠️ Skipping {horizon} - insufficient label diversity")
        return {'horizon': horizon, 'status': 'skipped', 'reason': 'insufficient_labels'}

    # Train XGBoost classifier
    model = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X, y)

    # Save model
    save_path = save_dir / f"ml_horizon_{horizon}_v2.pkl"
    joblib.dump(model, save_path)
    print(f"  ✅ Saved: {save_path}")

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"  Top 5 Features: {list(importances.head(5).index)}")

    return {
        'horizon': horizon,
        'status': 'success',
        'samples': len(X),
        'label_dist': dict(label_counts),
        'top_features': list(importances.head(5).index)
    }


def train_regime_model(
    X: pd.DataFrame,
    y: pd.Series,
    regime_labels: pd.Series,
    regime: str,
    horizon: str,
    symbol: str,
    save_dir: Path
) -> dict:
    """
    Train a regime-specific model.

    Args:
        X: Feature DataFrame
        y: Label Series
        regime_labels: Series with regime assignments
        regime: 'trend_up', 'trend_down', 'choppy', 'high_vol', 'low_vol'
        horizon: '1d', '3d', '5d', or '10d'
        symbol: Stock symbol
        save_dir: Directory to save model

    Returns:
        Training statistics dict
    """
    print(f"\n--- Training Regime Model: {regime} × {horizon} ---")

    # Filter to regime-specific data
    regime_mask = regime_labels == regime
    X_regime = X[regime_mask]
    y_regime = y[regime_mask]

    print(f"  Regime samples: {len(X_regime)}/{len(X)} ({len(X_regime)/len(X)*100:.1f}%)")

    if len(X_regime) < 100:
        print(f"  ⚠️ Skipping {regime}×{horizon} - insufficient samples (< 100)")
        return {'regime': regime, 'horizon': horizon, 'status': 'skipped', 'reason': 'insufficient_samples'}

    # Check label balance
    label_counts = y_regime.value_counts()
    print(f"  Label Distribution: {dict(label_counts)}")

    if len(label_counts) < 2:
        print(f"  ⚠️ Skipping {regime}×{horizon} - insufficient label diversity")
        return {'regime': regime, 'horizon': horizon, 'status': 'skipped', 'reason': 'insufficient_labels'}

    # Train XGBoost classifier
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_regime, y_regime)

    # Save model
    save_path = save_dir / f"ml_regime_{regime}_{horizon}_v2.pkl"
    joblib.dump(model, save_path)
    print(f"  ✅ Saved: {save_path}")

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=X_regime.columns).sort_values(ascending=False)
    print(f"  Top 5 Features: {list(importances.head(5).index)}")

    return {
        'regime': regime,
        'horizon': horizon,
        'status': 'success',
        'samples': len(X_regime),
        'label_dist': dict(label_counts),
        'top_features': list(importances.head(5).index)
    }


def main():
    parser = argparse.ArgumentParser(description='Train ML V2 models for PRADO9_EVO')
    parser.add_argument('symbol', type=str, help='Stock symbol (e.g., QQQ, SPY)')
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD, default: today)')

    args = parser.parse_args()

    # Load data
    data = load_data(args.symbol, args.start, args.end)

    # Build features (V2)
    print(f"\n{'='*80}")
    print("Building V2 Features (24 features)")
    print(f"{'='*80}")
    X = FeatureBuilderV2.build_features_v2(data)
    print(f"Features built: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Feature names: {list(X.columns)}")

    # Build labels
    print(f"\n{'='*80}")
    print("Building Labels")
    print(f"{'='*80}")

    horizon_labels = TargetBuilderV2.build_horizon_labels(data)
    regime_labels = TargetBuilderV2.build_regime_labels(data)

    print(f"Horizon labels: {horizon_labels.shape}")
    print(f"Regime labels: {regime_labels.shape}")

    # Align features and labels
    common_idx = X.index.intersection(horizon_labels.index).intersection(regime_labels.index)
    X = X.loc[common_idx]
    horizon_labels = horizon_labels.loc[common_idx]
    regime_labels = regime_labels.loc[common_idx]

    print(f"\nAligned dataset: {len(X)} samples")

    # Create save directory
    save_dir = Path.home() / '.prado' / 'models' / args.symbol / 'ml_v2'
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nModel save directory: {save_dir}")

    # =========================================================================
    # TRAIN HORIZON MODELS (4 models)
    # =========================================================================
    print(f"\n{'='*80}")
    print("TRAINING HORIZON MODELS (4 models)")
    print(f"{'='*80}")

    horizon_stats = []
    for horizon in ['1d', '3d', '5d', '10d']:
        y = TargetBuilderV2.get_label_for_horizon(horizon_labels, horizon)
        stats = train_horizon_model(X, y, horizon, args.symbol, save_dir)
        horizon_stats.append(stats)

    # =========================================================================
    # TRAIN REGIME MODELS (20 models: 5 regimes × 4 horizons)
    # =========================================================================
    print(f"\n{'='*80}")
    print("TRAINING REGIME MODELS (20 models)")
    print(f"{'='*80}")

    regimes = ['trend_up', 'trend_down', 'choppy', 'high_vol', 'low_vol']
    horizons = ['1d', '3d', '5d', '10d']

    regime_stats = []
    for regime in regimes:
        for horizon in horizons:
            # Use horizon-specific label
            y = TargetBuilderV2.get_label_for_horizon(horizon_labels, horizon)
            regime_column = regime_labels['regime']
            stats = train_regime_model(X, y, regime_column, regime, horizon, args.symbol, save_dir)
            regime_stats.append(stats)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")

    horizon_success = sum(1 for s in horizon_stats if s['status'] == 'success')
    regime_success = sum(1 for s in regime_stats if s['status'] == 'success')

    print(f"\nHorizon Models: {horizon_success}/4 trained successfully")
    print(f"Regime Models:  {regime_success}/20 trained successfully")
    print(f"Total Models:   {horizon_success + regime_success}/24")

    print(f"\nModels saved to: {save_dir}")
    print(f"\n✅ ML V2 Training Complete!")

    # Save training metadata
    metadata = {
        'symbol': args.symbol,
        'start_date': args.start,
        'end_date': args.end or pd.Timestamp.today().strftime('%Y-%m-%d'),
        'samples': len(X),
        'features': list(X.columns),
        'num_features': X.shape[1],
        'horizon_models': horizon_stats,
        'regime_models': regime_stats,
        'horizon_success': horizon_success,
        'regime_success': regime_success,
        'total_success': horizon_success + regime_success
    }

    metadata_path = save_dir / 'training_metadata_v2.pkl'
    joblib.dump(metadata, metadata_path)
    print(f"\nMetadata saved: {metadata_path}")


if __name__ == '__main__':
    main()
