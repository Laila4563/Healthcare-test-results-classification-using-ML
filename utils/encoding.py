import pandas as pd
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder


def encode_features(df, target_col=None, cardinality_threshold=10):
    df_encoded = df.copy()

    ordinal_map = {"Admission Type": {"Elective": 0, "Urgent": 1, "Emergency": 2}}

    for col, mapping in ordinal_map.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping)
            print(f"Ordinal encoding applied to '{col}' with mapping {mapping}")

    binary_map = {"Male": 0, "Female": 1, "Yes": 1, "No": 0, "True": 1, "False": 0}
    df_encoded = df_encoded.replace(binary_map)
    print("Binary replacement applied where possible using map:", binary_map)

    # Separate target
    y = None
    if target_col:
        if df_encoded[target_col].dtype == object:
            le = LabelEncoder()
            df_encoded[target_col] = le.fit_transform(df_encoded[target_col])
            print(f"Target column '{target_col}' label encoded.")
        y = df_encoded[target_col]
        df_encoded = df_encoded.drop(columns=[target_col])
        print(f"Target column '{target_col}' separated from features.")

    # Encode categorical features
    for col in df_encoded.select_dtypes(include="object").columns:
        unique_vals = df_encoded[col].nunique()
        print(f"'{col}' has {unique_vals} unique values")

        if unique_vals <= cardinality_threshold:
            df_encoded = pd.get_dummies(df_encoded, columns=[col], prefix=col)
            print(
                f"One-hot encoding applied to '{col}' (<= {cardinality_threshold} unique values)"
            )
        else:
            if y is not None:
                te = TargetEncoder()
                df_encoded[col] = te.fit_transform(df_encoded[col], y)
                print(
                    f"Target encoding applied to '{col}' (> {cardinality_threshold} unique values)"
                )
            else:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                print(
                    f"Label encoding applied to '{col}' (> {cardinality_threshold} unique values, no target provided)"
                )

    # Ensure all boolean columns are 0/1
    for col in df_encoded.columns:
        if df_encoded[col].dtype == bool:
            df_encoded[col] = df_encoded[col].astype(int)
            print(f"Boolean column '{col}' converted to integers (0/1)")

    if target_col:
        return df_encoded, y
    else:
        return df_encoded
