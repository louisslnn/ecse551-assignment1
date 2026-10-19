import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
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
        grad_W = (self.X.T @ (Y_prediction - self.Y) / self.m + self.l1_penalty * np.sign(self.W) + 2* self.l2_penalty * self.W)
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

sklearn_model = ElasticNet(
    alpha=alpha,
    l1_ratio=l1_ratio,
    max_iter=10000,
    tol=1e-8
)

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
print("Learn iterations:", sklearn_model.n_iter_)


#Plotting
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, scratch_preds, alpha=0.5, label="Custom")
plt.scatter( y_test, sklearn_preds, alpha=0.7, marker="x", label="Scikit Learn")
limits = [min(y_test.min(), scratch_preds.min(), sklearn_preds.min()), max(y_test.max(), scratch_preds.max(), sklearn_preds.max())]
plt.plot(limits, limits)
plt.title("Predicted vs. Actual")
plt.xlabel("Actual progression")
plt.ylabel("Predicted progression")
plt.legend()
plt.tight_layout()
plt.show()

scratch_residuals = y_test - scratch_preds
sklearn_residuals = y_test - sklearn_preds
plt.figure(figsize=(8, 6))
plt.hist(scratch_residuals, bins=15, alpha=0.5, density=True, label="Custom")
plt.hist(sklearn_residuals, bins=15, alpha=0.5, density=True, label="Scikit Learn")
plt.axvline(0, color="black", linestyle="--")
plt.title("Residual Distribution")
plt.xlabel("Actual - predicted")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

indices = np.arange(X.shape[1])
width = 0.5
plt.figure(figsize=(10, 6))
plt.bar(indices-width/2, mymodel.W, width,label="Custom")
plt.bar(indices+width/2, sklearn_model.coef_, width,label="Scikit Learn")
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Coefficient Comparison")
plt.xlabel("Feature index")
plt.ylabel("Coefficient (standardized)")
plt.xticks(indices)
plt.legend()
plt.tight_layout()
plt.show()

# objective vs iteration
plt.figure(figsize=(8, 6))
plt.semilogx(mymodel.loss_history)
plt.xlabel("Iteration")
plt.ylabel("Elastic Net objective")
plt.title("Custom Model Convergence")
plt.tight_layout()
plt.show()

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
        model = ElasticNetRegression(
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


# visualize the models performance
experiments = {
    "learning_rate": [0.001, 0.01, 0.025, 0.05, 0.1, 0.2],
    "iterations": [100, 500, 1000, 2500, 5000, 10000],
    "alpha": [0.001, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0],
    "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0]
}

# plot
for param_name, values in experiments.items():
    results = iterate_the_params(param_name, values)
    x = [r["value"] for r in results]
    mse = [r["mse"] for r in results]
    r2 = [r["r2"] for r in results]
    fig, ax1 = plt.subplots(figsize=(8, 6))
    line1 = ax1.plot(x, mse, marker="o", label="Test MSE")
    ax1.set_xlabel(param_name)
    ax1.set_ylabel("Test MSE")
    ax1.set_title(f"Effect of {param_name} on model performance")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()