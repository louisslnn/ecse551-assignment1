"""Task 3 training guard: no model may be trained on rows from the future.

The training target of row s is the return of day s+1, which is only known
at the close of day s+1. A model that predicts at the close of day t may
therefore only be trained on rows s <= t-1.
"""
import pandas as pd


def check_training_window(train_dates, predict_date):
    """Raise AssertionError unless every training row is strictly before predict_date.

    train_dates: the date of every training row. predict_date: the first day
    the fitted model will predict on (the refit date).
    """
    last_train = pd.DatetimeIndex(train_dates).max()
    predict_date = pd.Timestamp(predict_date)
    assert last_train < predict_date, (       # rows s must satisfy s <= t-1
        f"lookahead: training row dated {last_train.date()} is not before "
        f"the prediction date {predict_date.date()}")


def fit_model(model, X_train, y_train, train_dates, predict_date):
    """Fit a scikit-learn model after checking the training window. Returns the fitted model.

    Why: every fit in the walk-forward goes through this one function, so a
    lookahead mistake stops the run instead of producing a good-looking result.
    """
    check_training_window(train_dates, predict_date)
    model.fit(X_train, y_train)
    return model
