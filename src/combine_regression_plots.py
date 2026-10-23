import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

class ElasticNetRegression():

    def __init__(self, learning_rate, iterations, l1_penalty, l2_penalty):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.W = None
        self.b = None

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0.0
        self.X = X
        self.Y = Y
        self.loss_history = []
        for _ in range(self.iterations):
            self.update_weights()
        return self

    def update_weights(self):
        Y_prediction = self.predict(self.X)

        #gradients
        grad_W = (self.X.T @ (Y_prediction - self.Y) / self.m + self.l1_penalty * np.sign(self.W) + self.l2_penalty * self.W)
        grad_b = np.mean(Y_prediction - self.Y)

        #gradient descent
        self.W -= self.learning_rate * grad_W
        self.b -= self.learning_rate * grad_b

        # calculate objective function
        residuals = self.Y - self.predict(self.X)
        mse_loss = np.sum(residuals ** 2) / (2 * self.m)
        l1_loss = self.l1_penalty * np.sum(np.abs(self.W))
        l2_loss = 0.5 * self.l2_penalty * np.sum(self.W ** 2)
        self.loss_history.append(mse_loss + l1_loss + l2_loss)

    def predict(self, X):
        return X @ self.W + self.b

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_sc = StandardScaler()
X_train_sc = X_sc.fit_transform(X_train)
X_test_sc = X_sc.transform(X_test)
y_sc = StandardScaler()
y_train_sc = y_sc.fit_transform(y_train.reshape(-1, 1)).ravel()

alpha = 0.05
l1_ratio = 0.5

l1_penalty = alpha*l1_ratio
l2_penalty = alpha*(1-l1_ratio)

mymodel = ElasticNetRegression(
    learning_rate=0.05,
    iterations=10000,
    l1_penalty=l1_penalty,
    l2_penalty=l2_penalty
)

mymodel.fit(X_train_sc, y_train_sc)
mymodel_n_iter = len(mymodel.loss_history)

sklearn_model = LinearRegression()

sklearn_model.fit(X_train_sc, y_train_sc)

scratch_scaled_preds = mymodel.predict(X_test_sc)
sklearn_scaled_preds = sklearn_model.predict(X_test_sc)

#unstandardize
scratch_preds = y_sc.inverse_transform(scratch_scaled_preds.reshape(-1, 1)).ravel()
sklearn_preds = y_sc.inverse_transform(sklearn_scaled_preds.reshape(-1, 1)).ravel()

def evaluate(name, y_true, predictions):
    print(f"\n {name}")
    print(f"MSE: {mean_squared_error(y_true, predictions):.2f}")
    print(f"R2:  {r2_score(y_true, predictions):.4f}")


evaluate("Elastic Net", y_test, scratch_preds)
evaluate("Scikit Learn Elastic Net", y_test, sklearn_preds)

print("\nscratch coefficients:", mymodel.W)
print("Learn coefficients:", sklearn_model.coef_)
print("Custom iterations:", mymodel_n_iter)

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

class LassoRegression():

    def __init__(self, learning_rate, iterations, l1_penalty, l2_penalty):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.W = None
        self.b = None

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0.0
        self.X = X
        self.Y = Y
        self.loss_history = []
        for _ in range(self.iterations):
            self.update_weights()
        return self
    @staticmethod
    def soft_thresholding(x, threshold):
        return np.sign(x)*np.maximum(np.abs(x)-threshold, 0)

    def update_weights(self):
        Y_prediction = self.predict(self.X)
        #gradients
        grad_W = (self.X.T @ (Y_prediction - self.Y) / self.m)
        grad_b = np.mean(Y_prediction - self.Y)

        #gradient descent
        W_temp = self.W - self.learning_rate * grad_W
        self.W = self.soft_thresholding(
        W_temp, self.learning_rate * self.l1_penalty)
        self.b -= self.learning_rate * grad_b
        # calculate objective function
        residuals = self.Y - self.predict(self.X)
        mse_loss = np.sum(residuals ** 2) / (2 * self.m)
        l1_loss = self.l1_penalty * np.sum(np.abs(self.W))
        self.loss_history.append(mse_loss + l1_loss)

    def predict(self, X):
        return X @ self.W + self.b

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_sc = StandardScaler()
X_train_sc = X_sc.fit_transform(X_train)
X_test_sc = X_sc.transform(X_test)
y_sc = StandardScaler()
y_train_sc = y_sc.fit_transform(y_train.reshape(-1, 1)).ravel()

alpha = 0.05
l1_ratio = 0.5

l1_penalty = alpha*l1_ratio
l2_penalty = alpha*(1-l1_ratio)

mymodel = LassoRegression(
    learning_rate=0.05,
    iterations=10000,
    l1_penalty=alpha,
    l2_penalty=l2_penalty
)

mymodel.fit(X_train_sc, y_train_sc)
mymodel_n_iter = len(mymodel.loss_history)

sklearn_model = LinearRegression()

sklearn_model.fit(X_train_sc, y_train_sc)

scratch_scaled_preds = mymodel.predict(X_test_sc)
sklearn_scaled_preds = sklearn_model.predict(X_test_sc)

#unstandardize
scratch_preds = y_sc.inverse_transform(scratch_scaled_preds.reshape(-1, 1)).ravel()
sklearn_preds = y_sc.inverse_transform(sklearn_scaled_preds.reshape(-1, 1)).ravel()

def evaluate(name, y_true, predictions):
    print(f"\n{name}")
    print(f"MSE: {mean_squared_error(y_true, predictions):.2f}")
    print(f"R2:  {r2_score(y_true, predictions):.4f}")


evaluate("Custom lasso Net", y_test, scratch_preds)
evaluate("Scikit Learn lasso", y_test, sklearn_preds)

print("\nscratch coefficients:", mymodel.W)
print("Scikit Learn coefficients:", sklearn_model.coef_)
print("Custom iterations:", mymodel_n_iter)

#Analyse how the choice of parameters affects the model’s performance.

def iterate_the_params(param_name, param_values):
    results = []
    # original parameters
    base = {
        "learning_rate": 0.05,
        "iterations": 10000,
        "alpha": 0.05,
        "l1_ratio": 0.5
    }
    for value in param_values:
        params = base.copy()
        params[param_name] = value
        # convert into L1 and L2 penalties
        l1 = params["alpha"]*params["l1_ratio"]
        l2 = params["alpha"]*(1-params["l1_ratio"])
        model = LassoRegression(
            learning_rate=params["learning_rate"],
            iterations=params["iterations"],
            l1_penalty=l1,
            l2_penalty=l2
        )
        model.fit(X_train_sc, y_train_sc)

        predictions_scaled = model.predict(X_test_sc)
        predictions = y_sc.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        results.append({
            "value": value,
            "mse": mse,
            "r2": r2,
        })
    return results


# Baseline hyperparameters
base = {
    "learning_rate": 0.05,
    "iterations": 10000,
    "alpha": 0.05,
    "l1_ratio": 0.5
}

experiments = {
    "learning_rate": [0.001, 0.01, 0.025, 0.05, 0.1, 0.2],
    "iterations": [100, 500, 1000, 2500, 5000, 10000],
    "alpha": [0.001, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0],
    "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0]
}

def iterate_the_params(param_name, param_values, model_class):
    results = []

    for value in param_values:
        params = base.copy()
        params[param_name] = value
        if model_class == ElasticNetRegression:
            l1 = params["alpha"] * params["l1_ratio"]
            l2 = params["alpha"] * (1 - params["l1_ratio"])
        else:
            l1 = params["alpha"]
            l2 = 0.0
        model = model_class(
            learning_rate=params["learning_rate"],
            iterations=params["iterations"],
            l1_penalty=l1,
            l2_penalty=l2
        )
        model.fit(X_train_sc, y_train_sc)
        predictions_scaled = model.predict(X_test_sc)
        predictions = y_sc.inverse_transform(predictions_scaled.reshape(-1, 1) ).ravel()
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        results.append({
            "value": value,
            "mse": mse,
            "r2": r2
        })
    return results

# Baseline hyperparameters
base = {
    "learning_rate": 0.05,
    "iterations": 10000,
    "l1_penalty": 0.025,
    "l2_penalty": 0.025
}

experiments = {
    "learning_rate": [0.001, 0.01, 0.025, 0.05, 0.1, 0.2],
    "iterations": [100, 500, 1000, 2500, 5000, 10000],
    "l1_penalty": [0.0, 0.001, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0],
    "l2_penalty": [0.0, 0.001, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0]
}

def iterate_the_params(param_name, param_values, model_class):
    results = []
    for value in param_values:
        params = base.copy()
        params[param_name] = value
        if model_class == LassoRegression:
            params["l2_penalty"] = 0.0
        model = model_class(
            learning_rate=params["learning_rate"],
            iterations=params["iterations"],
            l1_penalty=params["l1_penalty"],
            l2_penalty=params["l2_penalty"]
        )
        model.fit(X_train_sc, y_train_sc)
        predictions_scaled = model.predict(X_test_sc)
        predictions = y_sc.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        results.append({
            "value": value,
            "mse": mse,
            "r2": r2
        })
    return results

sns.set_theme(style="whitegrid")
for param_name, values in experiments.items():
    elastic_results = iterate_the_params(param_name, values, ElasticNetRegression)
    if param_name != "l2_penalty":
        lasso_results = iterate_the_params(
            param_name, values, LassoRegression
        )
    x = [r["value"] for r in elastic_results]
    elastic_mse = [r["mse"] for r in elastic_results]
    plt.figure(figsize=(7, 4))
    plt.plot(x, elastic_mse, marker="o", label="Elastic Net")
    if param_name != "l2_penalty":
        lasso_mse = [r["mse"] for r in lasso_results]
        plt.plot(x, lasso_mse,marker="s", label="Lasso")
    plt.xlabel(param_name)
    plt.ylabel("Test MSE")
    plt.title(f"Effect of {param_name} on Model Performance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    elastic_best = min(elastic_results, key=lambda r: r["mse"])
    print(f"\n{param_name}")
    print(f"Tested range: {min(values)} to {max(values)}")
    print(f"Elastic Net best: {elastic_best['value']}, "f"MSE = {elastic_best['mse']:.2f}")
    if param_name != "l2_penalty":
        lasso_best = min(lasso_results, key=lambda r: r["mse"])
        print(f"Lasso best: {lasso_best['value']}, "
              f"MSE = {lasso_best['mse']:.2f}")

