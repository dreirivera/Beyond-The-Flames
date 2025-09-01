import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os

# For loading the raw dataset
df = pd.read_csv("dataset.csv")

# For defining input features and targets
feature_columns = [
    "Oxygen Concentration",
    "Humidity",
    "Temperature",
    "Wind Speed"
]
label_columns = [
    "Fire Intensity",
    "Heat Release Rate"
]

X = df[feature_columns]
y = df[label_columns]

# Number of bins for stratification
num_bins = 2

# For creating bins for each target separately
fire_intensity_bins = pd.qcut(df["Fire Intensity"], q=num_bins, labels=False, duplicates='drop')
heat_release_bins = pd.qcut(df["Heat Release Rate"], q=num_bins, labels=False, duplicates='drop')

# For combining the bins into a single stratification label by pairing them
combined_bins = fire_intensity_bins.astype(str) + "_" + heat_release_bins.astype(str)

# First split: 70% train, 30% temp (val+test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=combined_bins
)

# For creating combined bins for temp data for stratification in the second split
temp_fire_intensity_bins = pd.qcut(y_temp["Fire Intensity"], q=num_bins, labels=False, duplicates='drop')
temp_heat_release_bins = pd.qcut(y_temp["Heat Release Rate"], q=num_bins, labels=False, duplicates='drop')
temp_combined_bins = temp_fire_intensity_bins.astype(str) + "_" + temp_heat_release_bins.astype(str)

# Second split: 15% val, 15% test (half of temp each)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=temp_combined_bins
)

# For saving datasets
os.makedirs("datasets", exist_ok=True)
X_train.join(y_train).to_csv("datasets/train.csv", index=False)
X_val.join(y_val).to_csv("datasets/validation.csv", index=False)
X_test.join(y_test).to_csv("datasets/test.csv", index=False)

print("✅ Stratified dataset split completed for multi-output model.")
print("🔹 Training set: 'datasets/train.csv' (70%)")
print("🔹 Validation set: 'datasets/validation.csv' (15%)")
print("🔹 Test set: 'datasets/test.csv' (15%)")
