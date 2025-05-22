import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



def handle_categorical_missing_values(df):
    # Display missing data info before handling
    print("Missing values before handling categorical columns:")
    display_missing_info(df)
    
    # Categorical columns: use mode or "Unknown"
    categorical_cols = ['Blood Type', 'Doctor', 'Hospital', 'Insurance Provider', 'Admission Type']
    for col in categorical_cols:
        if col in df.columns:  # Check if column exists in dataframe
            if df[col].isnull().mean() > 0.05:
                df[col].fillna("Unknown", inplace=True)  # For higher missing %
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Display missing data info after handling
    print("\nMissing values after handling categorical columns:")
    display_missing_info(df)
    
    return df

def handle_numerical_missing_values(df):
    # Display missing data info before handling
    print("Missing values before handling numerical columns:")
    display_missing_info(df)
    
    # Numerical columns: use median
    if 'Billing Amount' in df.columns:  # Check if column exists in dataframe
        df['Billing Amount'].fillna(df['Billing Amount'].median(), inplace=True)
    
    # Display missing data info after handling
    print("\nMissing values after handling numerical columns:")
    display_missing_info(df)
    
    return df

def display_missing_info(df):
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    column_types = df.dtypes
    
    missing_data = pd.DataFrame({
        'Column Type': column_types,
        'Missing Values': missing_values,
        'Percentage (%)': missing_percentage
    })
    
    # Filter only columns with missing values
    missing_data = missing_data[missing_data['Missing Values'] > 0]
    
    if missing_data.empty:
        print("No missing values found!")
    else:
        print(missing_data)


def look_for_outliers(df):
    # Look for outliers:
    for col in df.select_dtypes(include="number").columns:
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.show()


# Function to handle outliers
def handle_outliers_iqr(df, columns):
    """
    Handle outliers in a DataFrame column using the IQR method.
    - Caps outliers at the lower and upper bounds.
    """

    df = df.copy()  # Avoid modifying the original data

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count outliers before capping
        num_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

        # Print information
        print(f"Feature: {col}, Outliers Detected: {num_outliers}")
        print(f" - Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}\n")

        # Cap outliers at the bounds
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    return df
