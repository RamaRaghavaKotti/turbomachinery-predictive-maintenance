import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)

# ---------------------------------------------------------
# 1. Define Column Headers from NASA C-MAPSS Paper
# ---------------------------------------------------------
index_cols = ['unit_number', 'time_cycles']
setting_cols = ['setting_1', 'setting_2', 'setting_3']
sensor_cols = [
    'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 
    'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc', 
    'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32'
]
col_names = index_cols + setting_cols + sensor_cols

# ---------------------------------------------------------
# 2. Load the Dataset
# ---------------------------------------------------------
file_path = 'train_FD001.txt' 
df = pd.read_csv(file_path, sep=r'\s+', header=None, names=col_names)
print(f"Dataset Loaded Successfully: {df.shape[0]} rows, {df.shape[1]} columns")

# ---------------------------------------------------------
# 3. Calculate RUL and Longevity Target (30-Cycle Horizon)
# ---------------------------------------------------------
max_cycles = df.groupby('unit_number')['time_cycles'].transform('max')
df['RUL'] = max_cycles - df['time_cycles']

HORIZON = 30
df['target_longevity'] = (df['RUL'] > HORIZON).astype(int)

# ---------------------------------------------------------
# 4. Isolate Active Thermodynamic Sensors (Features X) and Target (y)
# ---------------------------------------------------------
active_sensors = [
    'T24', 'T30', 'T50', 'P15', 'P30', 'Nf', 'Nc', 
    'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'htBleed', 'W31', 'W32'
]

X = df[active_sensors].copy()
y = df['target_longevity']

# ---------------------------------------------------------
# 5. Stratified Train-Test Split (75% Train, 25% Test)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 6. Fit Gaussian Naive Bayes Model
# ---------------------------------------------------------
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

# ---------------------------------------------------------
# 7. Model Performance Metrics & Classification Report
# ---------------------------------------------------------
print("\n" + "="*50)
print("       OVERALL MODEL PERFORMANCE SUMMARY")
print("="*50)
print(f"Overall Accuracy:    {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision (Class 1): {precision_score(y_test, y_pred):.3f}")
print(f"Recall (Class 1):    {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score (Class 1):  {f1_score(y_test, y_pred):.3f}")

print("\n" + "="*50)
print("     DETAILED CLASSIFICATION REPORT BY STATE")
print("="*50)
target_names = ['Overhaul Warning (0)', 'Healthy (1)']
print(classification_report(y_test, y_pred, target_names=target_names, digits=3))

# ---------------------------------------------------------
# 8. Physical Grounding: Sensor Drift Analysis
# ---------------------------------------------------------
print("="*50)
print("  THERMODYNAMIC SENSOR DRIFT (DEGRADATION SIGNATURE)")
print("="*50)

mean_healthy = X[y == 1].mean()
mean_overhaul = X[y == 0].mean()

drift_df = pd.DataFrame({
    'Healthy Mean': mean_healthy,
    'Overhaul Mean': mean_overhaul,
    '% Change': ((mean_overhaul - mean_healthy) / mean_healthy) * 100
}).round(2)

drift_df['abs_drift'] = drift_df['% Change'].abs()
drift_df = drift_df.sort_values(by='abs_drift', ascending=False).drop(columns=['abs_drift'])
print(drift_df)

# ---------------------------------------------------------
# 9. Confusion Matrix Visualization
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred, labels=nb.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Overhaul (0)', 'Healthy (1)'])

disp.plot(cmap='Blues', values_format='d')
plt.title("Gaussian Naive Bayes: Machine Longevity Classification")
plt.tight_layout()
plt.show()
