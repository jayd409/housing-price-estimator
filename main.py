#!/usr/bin/env python3
import sys, os
sys.path.insert(0, 'src')

from data_loader import load_data
from model import train_price_models
from visualizer import build_dashboard

def main():
    print("Housing Price Estimator")
    print("=" * 50)

    print("\nLoading housing data...")
    df = load_data()

    print("Training price estimation models...")
    metrics, df_pred, feature_importance, test_data = train_price_models(df)

    # Print model performance
    print(f"\n  Linear Regression :")
    print(f"    R² Score        : {metrics['ols_r2']:.4f}")
    print(f"    RMSE            : ${metrics['ols_rmse']:,.2f}")
    print(f"    CV Mean R²      : {metrics['cv_ols_mean']:.4f}")

    print(f"\n  Ridge Regression  :")
    print(f"    R² Score        : {metrics['ridge_r2']:.4f}")
    print(f"    RMSE            : ${metrics['ridge_rmse']:,.2f}")
    print(f"    CV Mean R²      : {metrics['cv_ridge_mean']:.4f}")

    print(f"\n  Dataset Summary   :")
    print(f"    Total Properties: {len(df):,}")
    print(f"    Price Range     : ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
    print(f"    Median Price    : ${df['price'].median():,.0f}")

    print("\nBuilding dashboard...")
    build_dashboard(df_pred, metrics, feature_importance, test_data)

    # Save predictions
    df_pred.to_csv('outputs/price_predictions.csv', index=False)
    print(f"  Predictions saved → outputs/price_predictions.csv")


    print("\nDone. Open outputs/dashboard.html to view.")

if __name__ == "__main__":
    main()
