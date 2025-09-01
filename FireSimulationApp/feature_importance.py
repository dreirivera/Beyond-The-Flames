import joblib
import numpy as np

feature_columns = ["Oxygen Concentration", "Humidity", "Temperature", "Wind Speed"]

# For loading the multi-output model
best_model = joblib.load("models/regression_model.pkl")

# best_model is a MultiOutputRegressor, so we access individual estimators
intensity_model = best_model.estimators_[0]
hrr_model = best_model.estimators_[1]

# For getting feature importances for each output
intensity_importances = intensity_model.feature_importances_
hrr_importances = hrr_model.feature_importances_

# For identifying most influential features
intensity_max_idx = np.argmax(intensity_importances)
hrr_max_idx = np.argmax(hrr_importances)

print()
print("=" * 50)
print("Feature Importance for Fire Intensity:")
for name, importance in zip(feature_columns, intensity_importances):
    print(f"  {name:<22}: {importance:.4f}")
print(f"Most influential: {feature_columns[intensity_max_idx]}")
print("-" * 50)
print("Feature Importance for Heat Release Rate:")
for name, importance in zip(feature_columns, hrr_importances):
    print(f"  {name:<22}: {importance:.4f}")
print(f"Most influential: {feature_columns[hrr_max_idx]}")
print("=" * 50)
print()
