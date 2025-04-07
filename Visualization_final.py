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

# 4. Remove duplicates
initial_count = len(df)
df.drop_duplicates(inplace=True)
removed_count = initial_count - len(df)
print(f"\nRemoved {removed_count} duplicate rows")

# Final data inspection
print("\nFinal Data Shape:", df.shape)
print("\nFirst 5 rows after preprocessing:")
print(df.head())

# Show the last few rows
print("Last 5 rows of the dataset:")
print(df.tail())

# Basic info
print("\nDataset Info:")
print(df.info())

# Dataset shape
print(f"\nDataset has {df.shape[0]} rows and {df.shape[1]} columns")

# Check for missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Check for duplicates
print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())

# Summary statistics
print("\n Summary Statistics (Numerical Features):")
print(df.describe())

# Categorical data summary
print("\nSummary of Categorical Features:")
print(df.describe(include=['object']))

# Value counts of categorical columns (if any)
print("\nValue Counts for Categorical Columns:")
for col in df.select_dtypes(include='object').columns:
    print(f"\nColumn: {col}")
    print(df[col].value_counts())

# --------------------------
# 📈 Univariate Analysis
# --------------------------

# Define metrics for visualizations
metrics = ['Total no. of beds', 'Number of COVID beds', 'Number of ICU beds', 'Number of ventilators or ABD']
titles = ['Total Beds', 'COVID Beds', 'ICU Beds', 'Ventilators']

# Set default font and style
plt.rcParams['font.family'] = 'DejaVu Sans'  # More widely available font
plt.style.use('seaborn-v0_8-colorblind')

# Initialize figure counter
fig_num = 1

# Histograms for numeric columns
plt.figure(fig_num, figsize=(12, 8))
df.select_dtypes(include=np.number).hist(bins=20, edgecolor='black')
plt.suptitle('Histogram of Numerical Features')
plt.tight_layout()
fig_num += 2

# 1. Box Plot with Visible X-axis
plt.figure(fig_num, figsize=(14, 8))
sns.boxplot(x='Zone Name', y='Total no. of beds', data=df, 
           hue='Zone Name', palette='pastel', legend=False)
plt.title('Bed Distribution by Zone', fontsize=14, pad=15)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Zone Name', fontsize=10)
plt.ylabel('Number of Beds', fontsize=12)
plt.tight_layout()
plt.xticks(rotation=45)

# 2. Line Plot - COVID Beds Trend
plt.figure(fig_num)
fig_num += 1
zone_stats = df.groupby('Zone Name')['Number of COVID beds'].sum().sort_values(ascending=False)
zone_stats.plot(kind='line', marker='o')
plt.title('COVID Beds Trend by Zone')
plt.ylabel('Number of COVID Beds')
plt.grid(True)

# 3. Scatter Plot - ICU vs Ventilators
plt.figure(fig_num)
fig_num += 1
sns.scatterplot(x='Number of ICU beds', y='Number of ventilators or ABD', 
                hue='Zone Name', data=df, s=100)
plt.title('ICU Beds vs Ventilators')

# 4. Correlation Heatmap
plt.figure(fig_num, figsize=(10, 8))
fig_num += 1
corr_matrix = df[metrics].corr()
heatmap = sns.heatmap(corr_matrix, 
                     annot=True, 
                     fmt=".2f", 
                     cmap='coolwarm', 
                     center=0,
                     linewidths=0.5,
                     cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Between Hospital Resources', fontsize=14, pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# 5. Enhanced Count Plot - Hospitals by Zone
plt.figure(fig_num, figsize=(12, 8))
fig_num += 1
sns.countplot(y='Zone Name', data=df, order=df['Zone Name'].value_counts().index,
             hue='Zone Name', palette='viridis', legend=False)
plt.title('Hospitals Count by Zone', fontsize=14, pad=20)
plt.xlabel('Number of Hospitals', fontsize=12)
plt.ylabel('Zone', fontsize=12)



# 6. Enhanced Scatter Plot with Visible Legend
plt.figure(fig_num, figsize=(14, 8))
fig_num += 1
scatter = sns.scatterplot(data=df, x='Number of ICU beds', y='Number of ventilators or ABD',
                         hue='Zone Name', size='Total no. of beds',
                         sizes=(50, 500), alpha=0.8, palette='tab20')
plt.title('ICU Beds vs Ventilators by Zone', fontsize=16, pad=25)
plt.xlabel('Number of ICU Beds', fontsize=12)
plt.ylabel('Number of Ventilators', fontsize=12)
plt.legend(bbox_to_anchor=(1.25, 1), loc='upper right', title='Zone Name')
plt.tight_layout()

# 7. Enhanced Pie Charts with Better Label Placement
plt.figure(fig_num, figsize=(16, 12))
fig_num += 1
plt.subplots_adjust(top=0.9, hspace=0.6, wspace=0.6)


# Get unique zones and assign vibrant colors
zones = df['Zone Name'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(zones)))
zone_colors = dict(zip(zones, colors))

for i, (metric, title) in enumerate(zip(metrics, titles), 1):
    plt.subplot(2, 2, i)
    zone_stats = df.groupby('Zone Name')[metric].sum().sort_values(ascending=False)
    
    # Calculate percentages
    percentages = zone_stats/zone_stats.sum()*100
    
    # Plot pie with non-overlapping percentage labels
    wedges, texts, autotexts = plt.pie(zone_stats,
            labels=None,
            autopct=lambda p: f'{p:.1f}%' if p >= 5 else '',
            colors=[zone_colors[zone] for zone in zone_stats.index],
            startangle=90,
            wedgeprops={'edgecolor':'black', 'linewidth':1},
            textprops={'fontsize':9, 'color':'black', 'weight':'bold'},
            pctdistance=0.8,
            rotatelabels=True)
    
    # Adjust label positions to prevent overlap
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.title(f'{title} Distribution\nTotal: {int(zone_stats.sum())}', 
             fontsize=12, pad=15)
    plt.axis('equal')

# Add simple legend without percentages
handles = [plt.Rectangle((0,0),1,1, color=zone_colors[zone]) for zone in zones]
plt.figlegend(handles, zones,
             title='Zones',
             loc='center right',
             bbox_to_anchor=(1.0, 0.5),
             fontsize=9)
# 8. Total Beds by Zone Bar Chart
plt.figure(fig_num, figsize=(14, 8))
fig_num += 1
sns.barplot(data=df, x='Zone Name', y='Total no. of beds', hue='Zone Name',
           palette='viridis', estimator=sum, errorbar=None, legend=False)
plt.title('Total Beds by Zone', fontsize=16, pad=20)
plt.xlabel('Zone', fontsize=14)
plt.ylabel('Total Beds', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# 9. Enhanced Histogram (Fixed)
plt.close(11)
plt.figure(11, figsize=(14, 8))
sns.histplot(data=df, x='Total no. of beds', bins=15, 
            kde=True, color='skyblue',
            edgecolor='black', alpha=0.8)
plt.title('Hospital Bed Capacity Distribution', fontsize=16, pad=20)
plt.xlabel('Number of Beds', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# 10. Enhanced Pair Plot with Visible Titles
plt.figure(12, figsize=(12, 10))
pairplot = sns.pairplot(df[metrics], diag_kind='kde',
            plot_kws={'alpha':0.7, 's':60, 'color':'teal'},
            diag_kws={'color':'salmon', 'fill':True})
pairplot.fig.suptitle('Pairwise Relationship of Resources', y=1.02, fontsize=14, weight='bold')
pairplot.fig.subplots_adjust(top=0.88, bottom=0.12)
for ax in pairplot.axes.flatten():
    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel().replace('Number', 'No.'), fontsize=11, labelpad=12)
    if ax.get_ylabel():
        ylabel = ax.get_ylabel()
        ylabel = ylabel.replace('Number', 'No.')
        ylabel = ylabel.replace('ventilators or ABD', 'Ventilators')
        ax.set_ylabel(ylabel, fontsize=11, labelpad=12)
plt.tight_layout()


# Show all plots
plt.show()
print("\nAll 12 visualizations displayed successfully! Close each plot window to continue.")
