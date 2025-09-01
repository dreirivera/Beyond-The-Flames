import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib 
import os
import csv
import matplotlib.pyplot as plt

# For Load Datasets
train_df = pd.read_csv("datasets/train.csv")
val_df = pd.read_csv("datasets/validation.csv")

feature_columns = ["Oxygen Concentration", "Humidity", "Temperature", "Wind Speed"]
target_columns = ["Fire Intensity", "Heat Release Rate"]

X_train = train_df[feature_columns]
y_train = train_df[target_columns]
X_val = val_df[feature_columns]
y_val = val_df[target_columns]

# For Defining Model and Hyperparameter Grid
base_model = RandomForestRegressor(random_state=42)
model = MultiOutputRegressor(base_model)

param_grid = {
    "estimator__n_estimators": [100],
    "estimator__max_depth": [None, 7],
    "estimator__min_samples_split": [2],
}

# For Performing Grid Search
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Predicting on Validation Set
y_pred_val = best_model.predict(X_val)
y_pred_train = best_model.predict(X_train)

print("\nHyperparameter tuning complete.")
print("Best parameters:", grid_search.best_params_)
print()

# For Calculating Metrics
mse_train = [
    mean_squared_error(y_train["Fire Intensity"], y_pred_train[:, 0]),
    mean_squared_error(y_train["Heat Release Rate"], y_pred_train[:, 1])
]
mse_val = [
    mean_squared_error(y_val["Fire Intensity"], y_pred_val[:, 0]),
    mean_squared_error(y_val["Heat Release Rate"], y_pred_val[:, 1])
]

r2_train = [
    r2_score(y_train["Fire Intensity"], y_pred_train[:, 0]),
    r2_score(y_train["Heat Release Rate"], y_pred_train[:, 1])
]
r2_val = [
    r2_score(y_val["Fire Intensity"], y_pred_val[:, 0]),
    r2_score(y_val["Heat Release Rate"], y_pred_val[:, 1])
]

# For Displaying Metrics
for i, target in enumerate(target_columns):
    print(f"* Train MSE for {target}: {mse_train[i]:.4f}")
    print(f"* Validation MSE for {target}: {mse_val[i]:.4f}")
    print(f"* Train R² for {target}: {r2_train[i]:.4f}")
    print(f"* Validation R² for {target}: {r2_val[i]:.4f}")
    print()

# For Saving Accuracy to History File
def save_accuracy_to_history():
    os.makedirs("datasets", exist_ok=True)
    history_file = "accuracy_history.csv"
    header = [
        "run_id",
        "r2_intensity_train", "r2_intensity_validation",
        "r2_hrr_train", "r2_hrr_validation",
        "mse_intensity_train", "mse_intensity_validation",
        "mse_hrr_train", "mse_hrr_validation"
    ]

    if os.path.exists(history_file):
        with open(history_file, "r", newline='') as f:
            reader = csv.DictReader(f)
            history = list(reader)
    else:
        history = []

    run_id = len(history) + 1
    new_row = {
        "run_id": run_id,
        "r2_intensity_train": f"{r2_train[0]:.4f}",
        "r2_intensity_validation": f"{r2_val[0]:.4f}",
        "r2_hrr_train": f"{r2_train[1]:.4f}",
        "r2_hrr_validation": f"{r2_val[1]:.4f}",
        "mse_intensity_train": f"{mse_train[0]:.4f}",
        "mse_intensity_validation": f"{mse_val[0]:.4f}",
        "mse_hrr_train": f"{mse_train[1]:.4f}",
        "mse_hrr_validation": f"{mse_val[1]:.4f}",
    }

    history.append(new_row)

    with open(history_file, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(history)




# For Saving Metrics and Charts
save_accuracy_to_history()


# For Saving Final Model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/regression_model.pkl")
print("Model saved as models/regression_model.pkl")
