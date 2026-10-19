"""Original model settings, with scaling inside each eligible training split."""
import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from task3_train import check_training_window, fit_model

SEED = 42
MLP_MAX_EPOCHS, MLP_HOLDOUT, MLP_BATCH = 30, 0.2, 1024


def new_mlp(spec):
    return MLPRegressor(hidden_layer_sizes=(spec["hidden"],), alpha=spec["alpha"],
                        batch_size=MLP_BATCH, random_state=SEED)


def mlp_validation_split(dates, target_dates):
    """Last 20% of feature days validate; older outcomes must precede that cut."""
    days = dates.unique().sort_values()
    if len(days) < 3:
        raise ValueError("MLP needs at least three eligible training days")
    cut = days[int(len(days) * (1 - MLP_HOLDOUT))]
    older = np.asarray((dates < cut) & (target_dates < cut))  # purge outcomes reaching validation
    newer = np.asarray(dates >= cut)
    if not older.any() or not newer.any():
        raise ValueError("empty MLP training or validation split after target-date purge")
    return older, newer, cut


def fit_mlp(spec, X, y, dates, predict_date, *, target_dates):
    """Choose epochs with a purged chronological holdout, then refit eligible rows."""
    check_training_window(dates, predict_date, target_dates=target_dates)
    older, newer, cut = mlp_validation_split(dates, target_dates)
    probe_scaler = StandardScaler().fit(X[older])  # no holdout features in this scaler
    X_old, X_new = probe_scaler.transform(X[older]), probe_scaler.transform(X[newer])
    probe, errors = new_mlp(spec), []
    for _ in range(spec.get("max_epochs", MLP_MAX_EPOCHS)):
        probe.partial_fit(X_old, y[older])
        errors.append(float(np.mean((probe.predict(X_new) - y[newer]) ** 2)))
    epochs = int(np.argmin(errors)) + 1
    scaler = StandardScaler().fit(X)  # final fit uses all eligible pre-prediction rows
    scaled, model = scaler.transform(X), new_mlp(spec)
    for _ in range(epochs):
        model.partial_fit(scaled, y)
    return make_pipeline(scaler, model), {
        "epochs": epochs, "holdout_mse": errors, "validation_first": str(cut.date()),
        "validation_train_target_last": str(target_dates[older].max().date()),
        "validation_scaler_mean": probe_scaler.mean_.tolist(),
    }


def fit_config(spec, X, y, dates, predict_date, *, target_dates):
    """Accept unscaled features; return a fitted scaling/model pipeline and audit info."""
    check_training_window(dates, predict_date, target_dates=target_dates)
    kind = spec["model"]
    if kind == "mlp":
        return fit_mlp(spec, X, y, dates, predict_date, target_dates=target_dates)
    if kind == "lasso":
        estimator = Lasso(alpha=spec["alpha"], precompute=True, max_iter=5000)
    elif kind == "ols":
        estimator = LinearRegression()
    elif kind == "rf":
        estimator = RandomForestRegressor(
            n_estimators=100, max_depth=spec["max_depth"],
            min_samples_leaf=spec["min_samples_leaf"], max_features=1 / 3,
            n_jobs=-1, random_state=SEED)
    else:
        raise ValueError(f"unknown model: {kind}")
    model = make_pipeline(StandardScaler(), estimator)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        fit_model(model, X, y, dates, predict_date, target_dates=target_dates)
    info = ({"importances": estimator.feature_importances_.tolist()} if kind == "rf"
            else {"coef": estimator.coef_.tolist()})
    return model, info


def lasso_alpha_grid(X, y, n=6):
    """Historical helper; corrected replay uses the exact original saved penalties."""
    X = StandardScaler().fit_transform(X)
    alpha_max = float(np.abs(X.T @ (y - y.mean())).max() / len(y))
    return alpha_max, [alpha_max * 0.5 * 0.4 ** k for k in range(n)]
