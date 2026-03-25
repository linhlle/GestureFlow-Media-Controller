import pandas as pd

df = pd.read_csv('../data/gesture_data.csv')
print("--- DATASET SUMMARY ---")
print(f"Total frame captured: {len(df)}")
print("\nSamples per Gesture: ")
print(df['label'].value_counts().sort_index())
print("\nMissing Values Check:")
print(df.isnull().sum().sum()) 

