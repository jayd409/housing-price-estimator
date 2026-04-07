import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import save_dashboard

def build_dashboard(df_pred, metrics, feature_importance, test_data):
    """Build 6-chart housing price estimation dashboard."""

    sns.set_style("whitegrid")

    # Chart 1: Price distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df_pred['price'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=df_pred['price'].median(), color='red', linestyle='--', linewidth=2, label=f'Median: ${df_pred["price"].median():,.0f}')
    ax.set_xlabel('Price ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Housing Price Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    charts = {'Price Distribution': fig}

    # Chart 2: Price vs Median Income (scatter, colored by region)
    fig, ax = plt.subplots(figsize=(10, 5))
    regions = df_pred['region'].unique()
    colors_map = {'Bay Area': '#1f77b4', 'Central CA': '#ff7f0e', 'SoCal': '#2ca02c'}
    for region in sorted(regions):
        mask = df_pred['region'] == region
        ax.scatter(df_pred[mask]['median_income'], df_pred[mask]['price'],
                   alpha=0.5, s=30, label=region, color=colors_map.get(region, 'gray'))
    ax.set_xlabel('Median Income')
    ax.set_ylabel('Price ($)')
    ax.set_title('Price vs Income (by Region)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    charts['Price vs Income'] = fig

    # Chart 3: Median price by region (bar)
    fig, ax = plt.subplots(figsize=(10, 5))
    region_prices = df_pred.groupby('region')['price'].mean().sort_values(ascending=False)
    ax.bar(range(len(region_prices)), region_prices.values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_xticks(range(len(region_prices)))
    ax.set_xticklabels(region_prices.index)
    ax.set_ylabel('Median Price ($)')
    ax.set_title('Median Home Price by Region', fontsize=12, fontweight='bold')
    for i, v in enumerate(region_prices.values):
        ax.text(i, v + 20000, f'${v:,.0f}', ha='center', fontweight='bold')
    charts['Price by Region'] = fig

    # Chart 4: Feature importance (top 8)
    fig, ax = plt.subplots(figsize=(10, 5))
    top_feat = feature_importance.head(8)
    ax.barh(range(len(top_feat)), top_feat['importance'].values, color='teal')
    ax.set_yticks(range(len(top_feat)))
    ax.set_yticklabels(top_feat['feature'].values)
    ax.set_xlabel('Coefficient Importance')
    ax.set_title('Feature Importance in Price Estimation', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    charts['Feature Importance'] = fig

    # Chart 5: Predicted vs Actual scatter
    fig, ax = plt.subplots(figsize=(10, 5))
    X_test, y_test, y_pred = test_data
    ax.scatter(y_test, y_pred, alpha=0.5, s=30, color='coral')
    # Perfect prediction line
    min_price = min(y_test.min(), y_pred.min())
    max_price = max(y_test.max(), y_pred.max())
    ax.plot([min_price, max_price], [min_price, max_price], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Price ($)')
    ax.set_ylabel('Predicted Price ($)')
    ax.set_title(f'Predicted vs Actual (R² = {metrics["ols_r2"]:.3f})', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    charts['Predictions'] = fig

    # Chart 6: Residual plot
    fig, ax = plt.subplots(figsize=(10, 5))
    residuals = y_test - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5, s=30, color='purple')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Price ($)')
    ax.set_ylabel('Residuals ($)')
    ax.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    charts['Residuals'] = fig

    # KPIs
    kpis = {
        'Linear R²': f"{metrics['ols_r2']:.3f}",
        'Ridge R²': f"{metrics['ridge_r2']:.3f}",
        'RMSE': f"${metrics['ols_rmse']:,.0f}",
        'CV Mean R²': f"{metrics['cv_ols_mean']:.3f}",
    }

    os.makedirs('outputs', exist_ok=True)
    save_dashboard(charts, 'Housing Price Estimator', 'outputs/dashboard.html', kpis=kpis)
