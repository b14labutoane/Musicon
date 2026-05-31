import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

from config import DATASET_PATH, DATA_PROCESSED, FEATURE_COLUMNS, MOOD_LABELS, RANDOM_SEED, TEST_SIZE

def load_raw_data():
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def normalize_features(df):
    scaler = MinMaxScaler()
    df[FEATURE_COLUMNS] = scaler.fit_transform(df[FEATURE_COLUMNS])

    print("Feature ranges after normalization:")
    for col in FEATURE_COLUMNS:
        print(f" {col}: [{df[col].min():.3f}, {df[col].max():.3f}]")

    return df, scaler

def split_and_save(df, scaler):
    os.makedirs(DATA_PROCESSED, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["mood_code"],
    )

    train_df.to_csv(os.path.join(DATA_PROCESSED, "train.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_PROCESSED, "test.csv"), index=False)

    joblib.dump(scaler, os.path.join(DATA_PROCESSED, "scaler.pkl"))

    print(f"\nTrain: {len(train_df)} rows")
    print(f"Test: {len(test_df)} rows")
    print(f"Saved to {DATA_PROCESSED}/")

def main():
    df = load_raw_data()
    df, scaler = normalize_features(df)
    split_and_save(df, scaler)
    print("\nDone.")

if __name__=="__main__":
    main()
