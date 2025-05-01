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
print("\nData Overview after loading:")
print("First 5 rows:")
print(df.head())
print("\nData shape:", df.shape)
print("Last 5 rows")
print(df.tail())
print("\nData info:")
print(df.info())
print("\nData types:")
print(df.dtypes)
print("Statistics Summary")
print(df.describe(()))


# Drop irrelevant columns
cols_to_drop = [col for col in df.columns if 'Unnamed' in col]
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"\nDropped columns: {cols_to_drop}")

print("\nMissing values before treatment:")
print(df.isnull().sum())




# Fill with empty string
obj_cols = df.select_dtypes(include=['object']).columns
df[obj_cols] = df[obj_cols].fillna('')

# Fill numeric columns with 0
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(0)

print("\nMissing values after treatment:")
print(df.isnull().sum())

# Data cleaning
print("\nData Cleaning:")
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Handle numeric columns
numeric_cols = ['District score 2021-22 - Overall']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(0)
    df[col] = df[col].astype(int)

print("\nData types after conversion:")
print(df.dtypes)

# Remove duplicates
initial_count = len(df)
df.drop_duplicates(inplace=True)
removed_count = initial_count - len(df)
print(f"\nRemoved {removed_count} duplicate rows")

#quartiles and IQR
q1 = df['District score 2021-22 - Overall'].quantile(0.25)
q3 = df['District score 2021-22 - Overall'].quantile(0.75)
iqr = q3 - q1
print(f"\nQuartile 1 (Q1): {q1}")
print(f"Quartile 3 (Q3): {q3}")
print(f"Interquartile Range (IQR): {iqr}")

#outliers based on IQR
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
outliers_iqr = df[(df['District score 2021-22 - Overall'] < lower_bound) | (df['District score 2021-22 - Overall'] > upper_bound)]
print(f"\nNumber of outliers based on IQR: {len(outliers_iqr)}")
if not outliers_iqr.empty:
    print("Outliers based on IQR:")
    print(outliers_iqr[['District', 'District score 2021-22 - Overall']])

#  z-scores
mean_score = df['District score 2021-22 - Overall'].mean()
std_score = df['District score 2021-22 - Overall'].std()
df['z_score'] = (df['District score 2021-22 - Overall'] - mean_score) / std_score
print(f"\nMean of scores: {mean_score}")
print(f"Standard deviation of scores: {std_score}")

#outliers based on z-score
outliers_z = df[(df['z_score'] > 3) | (df['z_score'] < -3)]
print(f"\nNumber of outliers based on z-score: {len(outliers_z)}")
if not outliers_z.empty:
    print("Outliers based on z-score:")
    print(outliers_z[['District', 'District score 2021-22 - Overall', 'z_score']])

print("\nFinal Data Shape:", df.shape)
print("\nFirst 5 rows after preprocessing:")
print(df.head())

# average district scores by state
state_avg_scores = df.groupby('State/UT')['District score 2021-22 - Overall'].mean().sort_values(ascending=False)
print("\nAverage District Scores by State/UT:")
print(state_avg_scores)

#average district scores by state
plt.figure(figsize=(14,8))
state_avg_df = state_avg_scores.reset_index()
state_avg_df.columns = ['State/UT', 'Average Score']
sns.barplot(x='State/UT', y='Average Score', hue='State/UT', data=state_avg_df, palette='viridis', legend=False)
plt.xticks(rotation=90)
plt.title('Average District Score by State/UT', fontsize=16)
plt.xlabel('State/UT', fontsize=14)
plt.ylabel('Average District Score', fontsize=14)
plt.tight_layout()
plt.show()

# Visualizations

metrics = ['District score 2021-22 - Overall']
titles = ['Overall District Score']

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('seaborn-v0_8-colorblind')

fig_num = 1

plt.figure(figsize=(10,6))
sns.histplot(df['District score 2021-22 - Overall'], bins=30, kde=True, color='skyblue', edgecolor='black')
plt.title('Distribution of Overall District Scores', fontsize=16)
plt.xlabel('Overall Score', fontsize=14)
plt.ylabel('Number of Districts', fontsize=14)
plt.show()
fig_num += 1

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

plt.figure(figsize=(8,6))
sns.countplot(x='Grade', hue='Grade', data=df, order=df['Grade'].value_counts().index, palette='pastel', legend=False)
plt.title('Distribution of Grades Across Districts', fontsize=16)
plt.xlabel('Grade', fontsize=14)
plt.ylabel('Number of Districts', fontsize=14)
plt.show()
fig_num += 1

plt.figure(figsize=(10,8))
corr = df[category_cols].astype(float).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Category Scores', fontsize=16)
plt.show()
fig_num += 1

df_pair = df[category_cols + ['District score 2021-22 - Overall']].copy()
df_pair.columns = ['Outcome', 'ECT', 'IF_SE', 'SS_CP', 'DL', 'GP', 'Overall']
df_pair = df_pair.apply(pd.to_numeric, errors='coerce').dropna()

sns.pairplot(df_pair, corner=True, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30})
plt.suptitle('Pair Plot of PGI Scores', y=1.02, fontsize=16)
plt.show()

plt.show()
print("\nAll visualizations displayed successfully! Close each plot window to continue.")
