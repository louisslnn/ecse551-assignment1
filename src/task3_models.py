"""Task 3 models: Lasso, random forest and a one-hidden-layer MLP (scikit-learn).

A model is described by a plain dict (a "spec"), for example
    {"model": "lasso", "alpha": 0.001}
    {"model": "ols"}                       (Lasso with alpha = 0, i.e. least squares)
    {"model": "rf", "max_depth": 6, "min_samples_leaf": 200}
    {"model": "mlp", "hidden": 16, "alpha": 0.0001}
fit_config() turns a spec plus training data into a fitted model. Every fit
first checks that no training row is dated on or after the prediction date.
"""
import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.neural_network import MLPRegressor

from task3_train import check_training_window, fit_model

SEED = 42
MLP_MAX_EPOCHS = 30        # the number of epochs is chosen between 1 and this
MLP_HOLDOUT = 0.2          # most recent share of the training days used to choose it
MLP_BATCH = 1024


def new_mlp(spec):
    """A fresh, untrained MLP with one hidden layer (early_stopping stays off: it splits randomly)."""
    return MLPRegressor(hidden_layer_sizes=(spec["hidden"],), alpha=spec["alpha"],
                        batch_size=MLP_BATCH, random_state=SEED)


def fit_mlp(spec, X, y, dates, predict_date):
    """Fit an MLP, choosing the number of epochs on a time-ordered holdout.

    Step 1: train on the oldest 80% of the training days, one epoch at a
    time, and measure the squared error on the most recent 20% of the days.
    Step 2: train a fresh network on ALL training days for the number of
    epochs that gave the smallest holdout error.
    Why: a random split would mix future and past days inside the window.
    """
    check_training_window(dates, predict_date)
    train_days = dates.unique().sort_values()
    cut = train_days[int(len(train_days) * (1 - MLP_HOLDOUT))]    # first day of the most recent 20%
    older = np.asarray(dates < cut)                               # days before the cut: fit; from the cut on: holdout
    X_old, y_old, X_new, y_new = X[older], y[older], X[~older], y[~older]

    max_epochs = spec.get("max_epochs", MLP_MAX_EPOCHS)
    probe, errors = new_mlp(spec), []
    for epoch in range(max_epochs):
        probe.partial_fit(X_old, y_old)                           # one more pass over the older days
        errors.append(float(np.mean((probe.predict(X_new) - y_new) ** 2)))
    best_epochs = int(np.argmin(errors)) + 1

    model = new_mlp(spec)
    for epoch in range(best_epochs):
        model.partial_fit(X, y)                                   # all training days, all dated before predict_date
    return model, {"epochs": best_epochs, "holdout_mse": errors}


def fit_config(spec, X, y, dates, predict_date):
    """Fit the model described by spec. Returns (fitted model, info to record).

    X is already standardized with training-window statistics. dates holds
    the day of every training row and must all be before predict_date.
    """
    kind = spec["model"]
    if kind == "lasso":
        model = Lasso(alpha=spec["alpha"], precompute=True, max_iter=5000)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)    # a fit that did not converge must not pass silently
            fit_model(model, X, y, dates, predict_date)
        return model, {"coef": model.coef_.copy()}
    if kind == "ols":
        model = LinearRegression()                                # the Lasso limit alpha = 0
        fit_model(model, X, y, dates, predict_date)
        return model, {"coef": model.coef_.copy()}
    if kind == "rf":
        model = RandomForestRegressor(n_estimators=100, max_depth=spec["max_depth"],
                                      min_samples_leaf=spec["min_samples_leaf"],
                                      max_features=1 / 3, n_jobs=-1, random_state=SEED)
        fit_model(model, X, y, dates, predict_date)
        return model, {"importances": model.feature_importances_.copy()}
    if kind == "mlp":
        return fit_mlp(spec, X, y, dates, predict_date)
    raise ValueError(f"unknown model: {kind}")


def lasso_alpha_grid(X, y, n=6):
    """Six log-spaced Lasso penalties derived from one training window.

    alpha_max is the smallest penalty at which every coefficient is zero:
    max_j |mean(x_j * y)| for standardized features and a centred target.
    The grid goes down from alpha_max / 2 by a factor of 2.5 per step.
    """
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    alpha_max = float(np.abs(X.T @ (y - y.mean())).max() / len(y))
    return alpha_max, [alpha_max * 0.5 * 0.4 ** k for k in range(n)]
