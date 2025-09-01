import os
import csv
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score

# For defining constants used across the module
MIN_SIMULATIONS_FOR_FEATURE = 5
FEATURE_COLUMNS = ["Oxygen Concentration", "Humidity", "Temperature", "Wind Speed"]
TARGET_COLUMNS = ["Fire Intensity", "Heat Release Rate"]

_regression_model = None # For storing the loaded or initialized regression model instance

# For loading the regression model from disk or initializing a new one if absent
def load_model():
    global _regression_model
    if _regression_model is None:
        model_path = os.path.join("models", "regression_model.pkl")
        if os.path.exists(model_path):
            # For loading an existing trained model from file
            _regression_model = joblib.load(model_path)
        else:
            # For initializing a default RandomForest-based multioutput regressor if model file missing
            _regression_model = MultiOutputRegressor(RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42
            ))


# For predicting Fire Intensity and Heat Release Rate given input variables
def predict_fire_behavior(inputs):
    load_model()

    # For creating a DataFrame from user input dictionary matching feature columns
    input_df = pd.DataFrame([[inputs["oxygen"], inputs["humidity"], inputs["temperature"], inputs["wind_speed"]]],
                            columns=FEATURE_COLUMNS)
    # For making prediction using the loaded regression model
    prediction = _regression_model.predict(input_df)[0]
    predicted_intensity, predicted_hrr = prediction

    # For loading user simulation data to decide if feature importance can be computed
    user_sim_path = "user_simulations.csv"
    if os.path.exists(user_sim_path):
        user_data = pd.read_csv(user_sim_path)
    else:
        # For initializing empty DataFrame if no user simulation data found
        user_data = pd.DataFrame(columns=FEATURE_COLUMNS + TARGET_COLUMNS)

    # For computing average feature importances if enough user data exists
    if len(user_data) >= MIN_SIMULATIONS_FOR_FEATURE:
        try:
            feature_importances = [est.feature_importances_ for est in _regression_model.estimators_]
            avg_importances = np.mean(feature_importances, axis=0)
            influential_variable = FEATURE_COLUMNS[np.argmax(avg_importances)]
        except AttributeError:
            # For handling cases where feature importances are unavailable
            influential_variable = "Unavailable (no feature importances)"
    else:
        # For indicating insufficient data to compute feature importance
        influential_variable = "Not enough simulations"

    # For returning the predicted fire intensity, HRR, and the most influential variable
    return predicted_intensity, predicted_hrr, influential_variable
    # For returns (predicted_intensity, predicted_hrr)
    return (some_float, some_other_float)


# For appending new model accuracy metrics to history CSV for tracking performance over runs
def save_accuracy_to_history(mse_train, mse_validation, r2_train, r2_validation):
    history_file = "accuracy_history.csv"
    header = [
        "run_id",
        "r2_intensity_train", "r2_intensity_validation",
        "r2_hrr_train", "r2_hrr_validation",
        "mse_intensity_train", "mse_intensity_validation",
        "mse_hrr_train", "mse_hrr_validation"
    ]

    # For reading existing accuracy history if present
    if os.path.exists(history_file):
        with open(history_file, "r", newline='') as f:
            reader = csv.DictReader(f)
            history = list(reader)
    else:
        # For initializing an empty history if no file exists
        history = []

    # For generating the next run ID as length of history + 1
    run_id = len(history) + 1
    new_row = {
        "run_id": run_id,
        "r2_intensity_train": f"{r2_train[0]:.4f}",
        "r2_intensity_validation": f"{r2_validation[0]:.4f}",
        "r2_hrr_train": f"{r2_train[1]:.4f}",
        "r2_hrr_validation": f"{r2_validation[1]:.4f}",
        "mse_intensity_train": f"{mse_train[0]:.4f}",
        "mse_intensity_validation": f"{mse_validation[0]:.4f}",
        "mse_hrr_train": f"{mse_train[1]:.4f}",
        "mse_hrr_validation": f"{mse_validation[1]:.4f}",
    }

    # For appending the new metrics row to the history list
    history.append(new_row)

    # For writing the updated history back to the CSV file
    with open(history_file, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(history)


# For generating and saving line charts visualizing accuracy metrics for training and validation datasets
def generate_accuracy_charts(mse_train, mse_validation, r2_train, r2_validation):
    # For plotting the R² score line chart
    plt.figure(figsize=(8, 5))
    plt.plot(["Train", "Validation"], [r2_train[0], r2_validation[0]], marker='o', label="Fire Intensity")
    plt.plot(["Train", "Validation"], [r2_train[1], r2_validation[1]], marker='o', label="Heat Release Rate")
    plt.title("R² Score for Fire Intensity and HRR")
    plt.ylabel("R² Score")
    plt.xlabel("Dataset")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("r2_line_chart.png")
    plt.close()

    # For plotting the Mean Squared Error line chart
    plt.figure(figsize=(8, 5))
    plt.plot(["Train", "Validation"], [mse_train[0], mse_validation[0]], marker='o', label="Fire Intensity")
    plt.plot(["Train", "Validation"], [mse_train[1], mse_validation[1]], marker='o', label="Heat Release Rate")
    plt.title("Mean Squared Error for Fire Intensity and HRR")
    plt.ylabel("MSE")
    plt.xlabel("Dataset")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mse_line_chart.png")
    plt.close()


# For adding a new user simulation entry to CSV, avoiding duplicates based on input features
def update_user_simulations(inputs, predicted_intensity, predicted_hrr, influential_variable):
    user_row = {
        "Oxygen Concentration": inputs["oxygen"],
        "Humidity": inputs["humidity"],
        "Temperature": inputs["temperature"],
        "Wind Speed": inputs["wind_speed"],
        "Fire Intensity": predicted_intensity,
        "Heat Release Rate": predicted_hrr,
        "Most Influential Variable": influential_variable
    }

    user_df = pd.DataFrame([user_row])

    # For checking and appending to existing CSV while preventing duplicate feature rows
    if os.path.exists("user_simulations.csv"):
        df_user = pd.read_csv("user_simulations.csv")
        is_duplicate = (
            (df_user["Oxygen Concentration"] == user_row["Oxygen Concentration"]) &
            (df_user["Humidity"] == user_row["Humidity"]) &
            (df_user["Temperature"] == user_row["Temperature"]) &
            (df_user["Wind Speed"] == user_row["Wind Speed"])
        ).any()
        if not is_duplicate:
            df_user = pd.concat([df_user, user_df], ignore_index=True)
            df_user.to_csv("user_simulations.csv", index=False)
            # For returning updated number of simulations after successful append
            return len(df_user)
        else:
            # For indicating duplicate input found; no append done
            return -1
    else:
        # For creating new CSV if not exists
        user_df.to_csv("user_simulations.csv", index=False)
        return 1


# For retrieving model performance metrics (MSE and R²) for train and validation datasets
def get_model_metrics():
    model_path = os.path.join("models", "regression_model.pkl")
    train_path = os.path.join("datasets", "train.csv")
    validation_path = os.path.join("datasets", "validation.csv")

    # For handling missing model or train dataset gracefully with zeroed metrics
    if not os.path.exists(model_path) or not os.path.exists(train_path):
        return {
            "mse_train": [0, 0],
            "mse_validation": [0, 0],
            "r2_train": [0, 0],
            "r2_validation": [0, 0]
        }

    # For loading the trained model and training data
    model = joblib.load(model_path)
    train_data = pd.read_csv(train_path)
    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMNS]
    y_train_pred = model.predict(X_train)

    # For calculating training MSE and R² scores for both targets
    mse_train = [
        mean_squared_error(y_train["Fire Intensity"], y_train_pred[:, 0]),
        mean_squared_error(y_train["Heat Release Rate"], y_train_pred[:, 1])
    ]
    r2_train = [
        r2_score(y_train["Fire Intensity"], y_train_pred[:, 0]),
        r2_score(y_train["Heat Release Rate"], y_train_pred[:, 1])
    ]

    # For loading validation data and calculating validation metrics if available
    if os.path.exists(validation_path):
        val_data = pd.read_csv(validation_path)
        X_val = val_data[FEATURE_COLUMNS]
        y_val = val_data[TARGET_COLUMNS]
        y_val_pred = model.predict(X_val)
        mse_validation = [
            mean_squared_error(y_val["Fire Intensity"], y_val_pred[:, 0]),
            mean_squared_error(y_val["Heat Release Rate"], y_val_pred[:, 1])
        ]
        r2_validation = [
            r2_score(y_val["Fire Intensity"], y_val_pred[:, 0]),
            r2_score(y_val["Heat Release Rate"], y_val_pred[:, 1])
        ]
    else:
        mse_validation = [0, 0]
        r2_validation = [0, 0]

    return {
        "mse_train": mse_train,
        "mse_validation": mse_validation,
        "r2_train": r2_train,
        "r2_validation": r2_validation
    }
