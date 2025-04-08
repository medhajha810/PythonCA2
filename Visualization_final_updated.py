import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the dataset with encoding fallback
encodings = ['utf-8', 'latin1', 'cp1252']
df = None

for encoding in encodings:
    try:
        df = pd.read_csv('Covid Beds.csv', encoding=encoding)
        print(f"Successfully loaded with {encoding} encoding")
        break
    except UnicodeDecodeError:
        continue

if df is None:
    raise ValueError("Failed to load CSV - tried encodings: utf-8, latin1, cp1252")

# --------------------------
# 🧹 Data Preprocessing
# --------------------------

# Initial data inspection
print("\nInitial Data Inspection:")
print("First 5 rows:")
print(df.head())
print("\nData shape:", df.shape)

# 1. Drop irrelevant columns (if any)
cols_to_drop = [col for col in df.columns if 'Unnamed' in col]
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"\nDropped columns: {cols_to_drop}")

# 2. Handle missing values
print("\nMissing values before treatment:")
print(df.isnull().sum())

# Forward fill for time-based data
time_sensitive_cols = ['Number of COVID beds', 'Number of ICU beds', 'Number of ventilators or ABD']
df[time_sensitive_cols] = df[time_sensitive_cols].ffill()

# Backward fill for remaining missing values
df = df.bfill()

print("\nMissing values after treatment:")
print(df.isnull().sum())

# 3. Data cleaning
print("\nData Cleaning:")
# Strip whitespace from all string columns
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Convert empty strings to NaN
df.replace('', np.nan, inplace=True)

# Handle numeric columns
numeric_cols = ['Total no. of beds', 'Number of COVID beds', 'Number of ICU beds', 'Number of ventilators or ABD']
for col in numeric_cols:
    # Convert to numeric, coerce errors to NaN
    df[col] = pd.to_numeric(df[col], errors='coerce')
    # Fill remaining NaN with 0 for these specific metrics
    df[col].fillna(0, inplace=True)
    # Convert to integers
    df[col] = df[col].astype(int)

print("\nData types after conversion:")
print(df.dtypes)

