import time
from collections.abc import Iterable
from itertools import product
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from MLP import MLP

class Grid:
    def __init__(self, param_grid):
        self.keys = sorted(param_grid.keys())
        self.values = []
        for k in self.keys:
            if isinstance(param_grid[k], Iterable):
                self.values.append(param_grid[k])
            else:
                self.values.append([param_grid[k]])
        self.grid = np.zeros([len(l) for l in self.values]).tolist()

    def __getitem__(self, indices):
        g = self.grid
        for i in indices:
            g = g[i]
        return g

    def __setitem__(self, indices, value):
        g = self.grid
        for i in indices[:-1]:
            g = g[i]
        g[indices[-1]] = value

    def __iter__(self):
        return product(*(range(len(l)) for l in self.values))

    def index_to_params(self, idx):
        param_list = [self.values[i][j] for i, j in enumerate(idx)]
        return dict(zip(self.keys, param_list))

class MLP2(MLP):
    def __init__(self, input_size, hidden_size, output_size, learning_rate, epochs):
        self.epochs = epochs
        super().__init__(input_size, hidden_size, output_size, learning_rate)

    def fit(self, X, y):
        super().fit(X, y, self.epochs)
        return self

def accuracy(y_pred, y_true):
    return np.count_nonzero(y_pred == y_true) / y_pred.size

def cross_validate(build_model, params, X, y):
    scores = []
    fit_times = []
    # Without stratification validation scores are substantially lower than test scores
    for train_idx, val_idx in StratifiedKFold(5).split(X, y):
        model = build_model(params)
        # fit on training set
        start = time.perf_counter()
        model.fit(X[train_idx], y[train_idx])
        fit_times.append(time.perf_counter() - start)
        # score on validation set
        y_pred = model.predict(X[val_idx])
        scores.append(accuracy(y_pred, y[val_idx]))
    return {
        "test_score": np.mean(scores),
        "fit_time": np.mean(fit_times)
    }

def grid_search(build_model, param_grid, X, y):
    grid = Grid(param_grid)
    best_idx = None
    best_score = -np.inf
    count = 0
    total = np.prod([len(l) for l in grid.values])
    for idx in grid:
        progress = f"grid search progress: {count}/{total}"
        print(progress, end="\r", flush=True)
        params = grid.index_to_params(idx)
        model = build_model(params)
        grid[idx] = cross_validate(build_model, params, X, y)
        score = grid[idx]["test_score"]
        if score > best_score:
            best_idx = idx
            best_score = score
        count += 1
    print(" " * len(progress), end="\r", flush=True)
    return grid, best_idx

def plot_scores(xs, scores, best_x, xscale, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axvline(best_x, color='k', alpha=0.3)
    ax.plot(xs, scores, 'o-')
    ax.set_xscale(xscale)
    ax.set_ylim(ymin=min(*scores, 1-2*(1-np.mean(scores))), ymax=1)
    ax.set_xticks(xs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_fit_times(xs, fit_times, best_x, xscale, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axvline(best_x, color='k', alpha=0.3)
    ax.plot(xs, fit_times, 'o-')
    ax.set_xscale(xscale)
    ax.set_ylim(ymin=0, ymax=max(*fit_times, 2*np.mean(fit_times)))
    ax.set_xticks(xs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_slices(grid, best_idx, config):
    for i, j in enumerate(best_idx):
        param = grid.keys[i]
        if param not in config:
            continue
        # find scores and fit times along the i-th axis
        scores = []
        fit_times = []
        idx = list(best_idx)
        for k in range(len(grid.values[i])):
            idx[i] = k
            results = grid[idx]
            scores.append(results["test_score"])
            fit_times.append(results["fit_time"])
        xs = grid.values[i]
        if config[param][1] in ["accuracy", "both"]:
            plot_scores(xs, scores, xs[j], config[param][0], param, "CV accuracy")
        if config[param][1] in ["fit time", "both"]:
            plot_fit_times(xs, fit_times, xs[j], config[param][0], param, "mean CV fit time")
    plt.show()
    plt.close()
