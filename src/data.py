import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath="data/raw/application_train.csv"):
    """
    Loads raw data, cleans anomalies, and returns X and y.
    """
    df = pd.read_csv(filepath)
    
    # Drop ID column as it contains no predictive information
    if "SK_ID_CURR" in df.columns:
        df = df.drop(columns=["SK_ID_CURR"])
        
    # Fix the 1000-year employment anomaly
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
        
    # Separate features and target
    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]
    
    return X, y

def get_data_splits(X, y):
    """
    Splits the data into Train (70%), Validation (15%), and Test (15%),
    stratified by the target variable.
    """
    # First split: 70% Train, 30% Temp (Validation + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    
    # Second split: Split the 30% Temp into two 15% halves
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    print("Loading and cleaning data...")
    X, y = load_and_clean_data()
    
    print("Splitting data (70% Train / 15% Validation / 15% Test)...")
    X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(X, y)
    
    print("\n--- Split Verification ---")
    
    def print_split_stats(name, X_split, y_split):
        default_rate = y_split.mean() * 100
        print(f"{name}:")
        print(f"  Shape: X={X_split.shape}, y={y_split.shape}")
        print(f"  Default Rate: {default_rate:.3f}%")
        
    print_split_stats("Train", X_train, y_train)
    print_split_stats("Validation", X_val, y_val)
    print_split_stats("Test", X_test, y_test)
    
    # Save the splits to data/processed for faster loading in later stages
    print("\nSaving splits to data/processed/...")
    X_train.to_pickle("data/processed/X_train.pkl")
    X_val.to_pickle("data/processed/X_val.pkl")
    X_test.to_pickle("data/processed/X_test.pkl")
    y_train.to_pickle("data/processed/y_train.pkl")
    y_val.to_pickle("data/processed/y_val.pkl")
    y_test.to_pickle("data/processed/y_test.pkl")
    print("Done!")
