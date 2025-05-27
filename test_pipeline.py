import pandas as pd
import numpy as np
import joblib
from utils import (
    handle_categorical_missing_values,
    handle_numerical_missing_values,
    encode_features,
    scale_features,
    handle_outliers_iqr,
    treat_skewness,
)
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION ---
TEST_CSV = "data/test data.csv"
MODEL_PATH = "saved_models/mlp_model.pkl"
LABEL_ENCODER_PATH = "stubs/label_encoder.pkl"
OUTPUT_CSV = "test_predictions.csv"

# --- LOAD DATA ---
df = pd.read_csv(TEST_CSV)

# --- PREPROCESSING ---

# 1. Handle missing values
df = handle_categorical_missing_values(df)
df = handle_numerical_missing_values(df)

# 2. Standardize text fields
df['Name'] = df['Name'].str.title()
df['Gender'] = df['Gender'].str.capitalize()
df['Medical Condition'] = df['Medical Condition'].str.capitalize()
df['Doctor'] = df['Doctor'].str.title()
df['Hospital'] = df['Hospital'].str.title()
df['Insurance Provider'] = df['Insurance Provider'].str.title()
df['Medication'] = df['Medication'].str.capitalize()
df['Admission Type'] = df['Admission Type'].str.capitalize()

# 3. Convert date columns
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce", dayfirst=True)
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"], errors="coerce", dayfirst=True)
df['Length of Stay in Days'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

# 4. Reorder columns (move Length of Stay before 'Test Results' if present)
if 'Test Results' in df.columns:
    target_index = df.columns.get_loc('Test Results')
    cols = list(df.columns)
    cols.insert(target_index, cols.pop(cols.index('Length of Stay in Days')))
    df = df[cols]

# 5. Drop unused columns
df.drop(columns=['Discharge Date', 'Date of Admission'], inplace=True)
df.drop(columns=['ID', 'Name', 'Room Number'], inplace=True)

# 6. Outlier handling
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features = [col for col in numerical_features if col not in ['Age']]
df = handle_outliers_iqr(df, numerical_features)

# 7. Skewness treatment
df, skewed_features, transformation_details = treat_skewness(df, numerical_features)

# 8. Scaling
df, scaled_cols, _ = scale_features(df, target_col=None, scaler_type='standard')

# 9. Encoding categorical features (no target)
X_encoded = encode_features(df, target_col=None)
X_final = X_encoded

# --- LOAD MODEL AND ENCODER ---
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# --- PREDICTION ---
y_pred = model.predict(X_final)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# --- OUTPUT ---
# Reload IDs from original file
test_ids = pd.read_csv(TEST_CSV)['ID']
output_df = pd.DataFrame({'ID': test_ids, 'Predicted Test Results': y_pred_labels})
output_df.to_csv(OUTPUT_CSV, index=False)

print(f"Predictions saved to {OUTPUT_CSV}")