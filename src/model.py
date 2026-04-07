import numpy as np
import pandas as pd
from ml_utils import normalize, linear_regression, lr_predict, r_squared, rmse

def train_price_models(df):
    """
    Train both OLS linear regression and Ridge regression models.
    Perform cross-validation and compute feature importance.
    """
    # Select numeric features for regression
    feature_cols = [c for c in df.columns if c not in ['price', 'region', 'neighborhood']]
    X = df[feature_cols].values
    y = df['price'].values
    feature_names = np.array(feature_cols)

    # Normalize
    X_norm, mean, std = normalize(X)

    # Train/test split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))

    X_train, X_test = X_norm[idx[:split]], X_norm[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    # OLS regression
    theta_ols = linear_regression(X_train, y_train)
    y_pred_ols = lr_predict(X_test, theta_ols)
    r2_ols = r_squared(y_test, y_pred_ols)
    rmse_ols = rmse(y_test, y_pred_ols)

    # Ridge regression (manual: theta = inv(X'X + lambda*I) @ X'y)
    lambda_ridge = 1.0
    Xb = np.column_stack([np.ones(len(X_train)), X_train])
    ridge_matrix = Xb.T @ Xb + lambda_ridge * np.eye(Xb.shape[1])
    theta_ridge = np.linalg.solve(ridge_matrix, Xb.T @ y_train)
    y_pred_ridge = lr_predict(X_test, theta_ridge)
    r2_ridge = r_squared(y_test, y_pred_ridge)
    rmse_ridge = rmse(y_test, y_pred_ridge)

    # Cross-validation (5-fold manual)
    fold_size = len(X_train) // 5
    cv_scores_ols = []
    cv_scores_ridge = []

    for i in range(5):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < 4 else len(X_train)

        X_val = X_train[val_start:val_end]
        y_val = y_train[val_start:val_end]
        X_train_fold = np.vstack([X_train[:val_start], X_train[val_end:]])
        y_train_fold = np.hstack([y_train[:val_start], y_train[val_end:]])

        # OLS fold
        theta_fold = linear_regression(X_train_fold, y_train_fold)
        y_pred_fold = lr_predict(X_val, theta_fold)
        cv_scores_ols.append(r_squared(y_val, y_pred_fold))

        # Ridge fold
        Xb_fold = np.column_stack([np.ones(len(X_train_fold)), X_train_fold])
        ridge_mat_fold = Xb_fold.T @ Xb_fold + lambda_ridge * np.eye(Xb_fold.shape[1])
        theta_ridge_fold = np.linalg.solve(ridge_mat_fold, Xb_fold.T @ y_train_fold)
        y_pred_ridge_fold = lr_predict(X_val, theta_ridge_fold)
        cv_scores_ridge.append(r_squared(y_val, y_pred_ridge_fold))

    # Feature importance from OLS coefficients
    coef_importance = np.abs(theta_ols[1:])  # Skip bias
    coef_importance = coef_importance / coef_importance.sum()

    metrics = {
        'ols_r2': r2_ols,
        'ols_rmse': rmse_ols,
        'ridge_r2': r2_ridge,
        'ridge_rmse': rmse_ridge,
        'cv_ols_mean': round(np.mean(cv_scores_ols), 4),
        'cv_ridge_mean': round(np.mean(cv_scores_ridge), 4),
    }

    # Add predictions to original dataframe
    df_with_pred = df.copy()
    X_all_norm = (X - mean) / std
    y_pred_all = lr_predict(X_all_norm, theta_ols)
    df_with_pred['predicted_price'] = y_pred_all

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': coef_importance
    }).sort_values('importance', ascending=False)

    return metrics, df_with_pred, feature_importance, (X_test, y_test, y_pred_ols)
