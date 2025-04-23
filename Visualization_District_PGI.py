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
        df = pd.read_csv('District_PGI_Table_1.csv', encoding=encoding)
        print(f"Successfully loaded with {encoding} encoding")
        break
    except UnicodeDecodeError:
        continue

if df is None:
    raise ValueError("Failed to load CSV - tried encodings: utf-8, latin1, cp1252")

# Data Preprocessing

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
time_sensitive_cols = ['District score 2021-22 - Overall']  # Adjust as needed
df[time_sensitive_cols] = df[time_sensitive_cols].ffill()

# Backward fill for remaining missing values
df = df.bfill()

print("\nMissing values after treatment:")
print(df.isnull().sum())

# 3. Data cleaning
print("\nData Cleaning:")
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df.replace('', np.nan, inplace=True)

# Handle numeric columns
numeric_cols = ['District score 2021-22 - Overall']  # Focus on overall score

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(0)
    df[col] = df[col].astype(int)

print("\nData types after conversion:")
print(df.dtypes)

# 4. Remove duplicates
initial_count = len(df)
df.drop_duplicates(inplace=True)
removed_count = initial_count - len(df)
print(f"\nRemoved {removed_count} duplicate rows")

# Final data inspection
print("\nFinal Data Shape:", df.shape)
print("\nFirst 5 rows after preprocessing:")
print(df.head())

# Univariate Analysis

# Define metrics for visualizations
metrics = ['District score 2021-22 - Overall']  # Focus on overall score
titles = ['Overall District Score']  # Titles for plots

# Set default font and style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('seaborn-v0_8-colorblind')

# Initialize figure counter
fig_num = 1

# 1. Hist plot
plt.figure(figsize=(10,6))
sns.histplot(df['District score 2021-22 - Overall'], bins=30, kde=True, color='skyblue', edgecolor='black')
plt.title('Distribution of Overall District Scores', fontsize=16)
plt.xlabel('Overall Score', fontsize=14)
plt.ylabel('Number of Districts', fontsize=14)
plt.show()
fig_num += 1

# 2. Box Chart
category_cols = [
    'District score 2021-22 - Category - 1.Outcome (290)',
    'District score 2021-22 - Category - 2. ECT (90)',
    'District score 2021-22 - Category - 3. IF&SE (51)',
    'District score 2021-22 - Category - 4.SS&CP (35)',
    'District score 2021-22 - Category - 5. DL (50)',
    'District score 2021-22 - Category - 6. GP (84)'
]

df_melted = df.melt(id_vars=['State/UT'], value_vars=category_cols, var_name='Category', value_name='Score')
plt.figure(figsize=(12,6))
sns.boxplot(x='Category', y='Score', hue='Category', data=df_melted, palette='Set2', legend=False)
plt.xticks(rotation=45)
plt.title('Distribution of Scores by Category', fontsize=16)
plt.tight_layout()
plt.show()
fig_num += 1

top10 = df.sort_values(by='District score 2021-22 - Overall', ascending=False).head(10)
plt.figure(figsize=(12,6))
sns.barplot(x='District', y='District score 2021-22 - Overall', hue='District', data=top10, palette='Set1', legend=False)
plt.xticks(rotation=45)
plt.title('Top 10 Districts by Overall Score', fontsize=16)
plt.ylabel('Overall Score', fontsize=14)
plt.show()
fig_num += 1

# 3. Count plot
plt.figure(figsize=(8,6))
sns.countplot(x='Grade', hue='Grade', data=df, order=df['Grade'].value_counts().index, palette='pastel', legend=False)
plt.title('Distribution of Grades Across Districts', fontsize=16)
plt.xlabel('Grade', fontsize=14)
plt.ylabel('Number of Districts', fontsize=14)
plt.show()
fig_num += 1

# 4. Correlation Heatmap
plt.figure(figsize=(10,8))
corr = df[category_cols].astype(float).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Category Scores', fontsize=16)
plt.show()
fig_num += 1

# 5. Pair plot
df_pair = df[[
    'District score 2021-22 - Category - 1.Outcome (290)',
    'District score 2021-22 - Category - 2. ECT (90)',
    'District score 2021-22 - Category - 3. IF&SE (51)',
    'District score 2021-22 - Category - 4.SS&CP (35)',
    'District score 2021-22 - Category - 5. DL (50)',
    'District score 2021-22 - Category - 6. GP (84)',
    'District score 2021-22 - Overall'
]].copy()

# Rename columns for simplicity
df_pair.columns = ['Outcome', 'ECT', 'IF_SE', 'SS_CP', 'DL', 'GP', 'Overall']

# Convert to numeric and drop rows with missing values
df_pair = df_pair.apply(pd.to_numeric, errors='coerce').dropna()

# Generate pair plot with enhanced aesthetics
sns.pairplot(df_pair, corner=True, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30})
plt.suptitle('Pair Plot of PGI Scores', y=1.02, fontsize=16)
plt.show()

# Show all plots
plt.show()
print("\nAll visualizations displayed successfully! Close each plot window to continue.")
