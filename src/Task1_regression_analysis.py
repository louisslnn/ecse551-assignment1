from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
df = pd.DataFrame(X, columns=diabetes.feature_names)
df['target'] = y
print(f"Features: {list(diabetes.feature_names)}")

#statistical analysis
print("\nshape:", df.shape)
print("\nColumns:", df.columns.tolist())
expected_columns = list(diabetes.feature_names)
missing_columns = set(expected_columns)-set(df.columns)
print("\nmissing columns:", missing_columns)
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nMin and max:")
print(df.agg(["min", "max"]).T)
df=df.replace([np.inf, -np.inf], np.nan)
df=df.dropna().drop_duplicates()
print("\nShape w/o duplicates:", df.shape)
print(df.describe().loc[['min', 'max']])
print(df.describe().loc['mean'])

#cor matrix
correlation = df.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(correlation,annot=True,fmt=".2f",cmap="vlag",vmin=-1,vmax=1)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

featurenames = list(diabetes.feature_names)
print(df[featurenames].describe())
meanminmax = df[featurenames].agg(["mean", "min", "max"])
print(meanminmax.round(3))

df[featurenames].hist(bins=20,figsize=(12, 10),edgecolor="black")
plt.suptitle("Distributions")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 5, figsize=(15, 8))
for ax, feature in zip(axes.flat, featurenames):
    ax.boxplot(df[feature].dropna())
    ax.set_title(feature)
    ax.set_ylabel("Value")
plt.suptitle("Box Plots")
plt.tight_layout()
plt.show()
fig, axes = plt.subplots(1, 4, figsize=(15, 4))

axes[0].scatter(df["bmi"], df["bp"], alpha=0.5)
axes[0].set_xlabel("BMI")
axes[0].set_ylabel("Blood pressure")
axes[0].set_title("BMI vs Blood Pressure")
axes[1].scatter(df["bmi"], df["target"], alpha=0.5)
axes[1].set_xlabel("BMI")
axes[1].set_ylabel("Disease progression")
axes[1].set_title("BMI vs Target")
axes[2].scatter(df["s5"], df["target"], alpha=0.5)
axes[2].set_xlabel("s5")
axes[2].set_ylabel("Disease progression")
axes[2].set_title("s5 vs Target")
axes[3].scatter(df["s2"], df["s1"], alpha=0.5)
axes[3].set_xlabel("s2")
axes[3].set_ylabel("s1")
axes[3].set_title("s2 vs s1")

plt.tight_layout()
plt.show()

#not scaled data
diabetes = load_diabetes(scaled=False, as_frame=True)
print("\nMinimum and maximum:")
print(df.agg(["min", "max"]))
print(df.describe().loc[['min', 'max']])
print(df.describe().loc['mean'])

featurenames = list(diabetes.feature_names)
df = diabetes.frame.copy()

print(df[featurenames].describe())
meanminmax = df[featurenames].agg(["mean", "min", "max"])
print(meanminmax.round(3))

fig, axes = plt.subplots(2, 5, figsize=(15, 8))
for ax, feature in zip(axes.flat, featurenames):
    ax.boxplot(df[feature].dropna())
    ax.set_title(feature)
    ax.set_ylabel("Value")
plt.suptitle("Box Plots")
plt.tight_layout()
plt.show()

df[featurenames].hist(bins=20,figsize=(12, 10),edgecolor="black")
plt.suptitle("Distributions")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 4, figsize=(15, 4))

axes[0].scatter(df["bmi"], df["bp"], alpha=0.5)
axes[0].set_xlabel("BMI")
axes[0].set_ylabel("Blood pressure")
axes[0].set_title("BMI vs Blood Pressure")
axes[1].scatter(df["bmi"], df["target"], alpha=0.5)
axes[1].set_xlabel("BMI")
axes[1].set_ylabel("Disease progression")
axes[1].set_title("BMI vs Target")
axes[2].scatter(df["s5"], df["target"], alpha=0.5)
axes[2].set_xlabel("s5")
axes[2].set_ylabel("Disease progression")
axes[2].set_title("s5 vs Target")
axes[3].scatter(df["s2"], df["s1"], alpha=0.5)
axes[3].set_xlabel("s2")
axes[3].set_ylabel("s1")
axes[3].set_title("s2 vs s1")

plt.tight_layout()
plt.show()