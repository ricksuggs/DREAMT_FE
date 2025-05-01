import matplotlib.pyplot as plt
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Simulate toy dataset
np.random.seed(42)
n_samples = 500
data = pd.DataFrame({
    "HRV": np.random.uniform(0.05, 0.3, size=n_samples),
    "ACC_Index": np.random.uniform(0, 1, size=n_samples),
    "TEMP": np.random.uniform(30, 36, size=n_samples),
})

# Simulate binary labels: wake = 1, sleep = 0
data["label"] = ((data["HRV"] < 0.12) & (data["ACC_Index"] > 0.7) & (data["TEMP"] < 32.5)).astype(int)

# Prepare features and labels
X = data[["HRV", "ACC_Index", "TEMP"]]
y = data["label"]
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train a GBDT model
lgb_train = lgb.Dataset(X_train_scaled, label=y_train)
params = {"objective": "binary", "metric": "binary_logloss", "num_leaves": 5, "max_depth": 3}
model = lgb.train(params, lgb_train, num_boost_round=5)

# Plot the first tree
fig, ax = plt.subplots(figsize=(12, 6))
lgb.plot_tree(model, tree_index=0, ax=ax, show_info=['split_gain', 'internal_value', 'leaf_count'])
plt.title("GBDT Tree for Sleep/Wake Classification")
plt.tight_layout()
plt.savefig("sleep_tree.png", dpi=300)