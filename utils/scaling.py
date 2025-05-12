from sklearn.preprocessing import StandardScaler, MinMaxScaler


def scale_features(df, target_col=None, scaler_type="standard"):

    df_scaled = df.copy()

    numerical_cols = df_scaled.select_dtypes(include=['int64', 'float64']).columns.tolist()
    drop_cols = ['ID']
    if target_col:
        drop_cols.append(target_col)
    numerical_cols = [col for col in numerical_cols if col not in drop_cols]

    # Define known categorical columns that may have been encoded
    encoded_categoricals = ['Doctor', 'Hospital', 'Admission Type', 'Gender', 'Insurance Provider']

    # Filter for true continuous columns
    continuous_cols = [col for col in numerical_cols if col not in encoded_categoricals]

    print(f"[INFO] Numerical columns: {numerical_cols}")
    print(f"[INFO] Continuous columns selected for scaling: {continuous_cols}")

    if not continuous_cols:
        print("[WARN] No continuous columns selected for scaling.")
        return df_scaled, []

    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    print(f"[INFO] Using {'StandardScaler' if scaler_type == 'standard' else 'MinMaxScaler'}")

    df_scaled[continuous_cols] = scaler.fit_transform(df_scaled[continuous_cols])
    print(f"[SUCCESS] Scaled features: {continuous_cols}")

    return df_scaled, continuous_cols