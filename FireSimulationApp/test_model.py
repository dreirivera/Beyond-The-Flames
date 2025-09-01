import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# For loading test dataset
test_df = pd.read_csv("datasets/test.csv")

feature_columns = ["Oxygen Concentration", "Humidity", "Temperature", "Wind Speed"]
target_columns = ["Fire Intensity", "Heat Release Rate"]

X_test = test_df[feature_columns]
y_test = test_df[target_columns]

# For loading trained model
model = joblib.load("models/regression_model.pkl")

# For predicting on test set
y_pred = model.predict(X_test)

# For calculating metrics
mse_test = [
    mean_squared_error(y_test["Fire Intensity"], y_pred[:, 0]),
    mean_squared_error(y_test["Heat Release Rate"], y_pred[:, 1])
]

r2_test = [
    r2_score(y_test["Fire Intensity"], y_pred[:, 0]),
    r2_score(y_test["Heat Release Rate"], y_pred[:, 1])
]

# For printing test scores
print(f"Test MSE Fire Intensity: {mse_test[0]:.4f}")
print(f"Test R² Fire Intensity: {r2_test[0]:.4f}")
print(f"Test MSE Heat Release Rate: {mse_test[1]:.4f}")
print(f"Test R² Heat Release Rate: {r2_test[1]:.4f}")
