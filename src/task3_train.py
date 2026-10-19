"""Training guards for v2: target outcomes precede the prediction date."""
import pandas as pd


def check_training_window(train_dates, predict_date, *, target_dates):
    """Check feature AND outcome dates against the actual trading calendar."""
    features, outcomes = pd.DatetimeIndex(train_dates), pd.DatetimeIndex(target_dates)
    if len(features) == 0 or len(features) != len(outcomes):
        raise ValueError("training needs nonempty, equally sized feature and outcome dates")
    assert not features.hasnans and not outcomes.hasnans, "lookahead: missing training dates"
    assert (features < outcomes).all(), "lookahead: targets must follow their feature dates"
    assert (outcomes < pd.Timestamp(predict_date)).all(), (
        "lookahead: every training target must be realized strictly before prediction")


def fit_model(model, X_train, y_train, train_dates, predict_date, *, target_dates):
    """Fit only after validating label availability, not just feature timestamps."""
    check_training_window(train_dates, predict_date, target_dates=target_dates)
    model.fit(X_train, y_train)
    return model
