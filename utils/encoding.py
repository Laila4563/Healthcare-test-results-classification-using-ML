import pandas as pd
from sklearn.preprocessing import LabelEncoder
# pip install category_encoders
from category_encoders import TargetEncoder


def encode_features(df, target_col=None, cardinality_threshold=10):
    df_encoded = df.copy()
    
    ordinal_map = {
    'Admission Type': {
        'Elective': 0,
        'Urgent': 1,
        'Emergency': 2
        }
    }

    for col, mapping in ordinal_map.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping)

    binary_map = {"Male": 0, "Female": 1, "Yes": 1, "No": 0, "True": 1, "False": 0}
    df_encoded = df_encoded.replace(binary_map)

    # Separate target
    y = None
    if target_col:
        if df_encoded[target_col].dtype == object:
            le = LabelEncoder()
            df_encoded[target_col] = le.fit_transform(df_encoded[target_col])
        y = df_encoded[target_col]
        df_encoded = df_encoded.drop(columns=[target_col])

    # Encode features
    for col in df_encoded.select_dtypes(include="object").columns:
        unique_vals = df_encoded[col].nunique()
        print(f"{col} has {unique_vals} unique features")

        if unique_vals <= cardinality_threshold:
            df_encoded = pd.get_dummies(df_encoded, columns=[col], prefix=col)
        else:
            if y is not None:
                te = TargetEncoder()
                df_encoded[col] = te.fit_transform(df_encoded[col], y)
            else:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])

    # Ensure all boolean columns are 0/1
    for col in df_encoded.columns:
        if df_encoded[col].dtype == bool:
            df_encoded[col] = df_encoded[col].astype(int)

    # Final return
    if target_col:
        return df_encoded, y
    else:
        return df_encoded
