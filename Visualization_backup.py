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

# Data cleaning
# Remove extra spaces in column names and values
df.columns = df.columns.str.strip()
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Convert numerical columns to numeric
numeric_cols = ['Total no. of beds', 'Number of COVID beds', 'Number of ICU beds', 'Number of ventilators or ABD']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

# 1. Basic Statistics
print("\n=== Basic Statistics ===")
print(f"Total hospitals: {len(df)}")
print(f"Total beds: {df['Total no. of beds'].sum()}")
print(f"Total COVID beds: {df['Number of COVID beds'].sum()}")
print(f"Total ICU beds: {df['Number of ICU beds'].sum()}")
print(f"Total ventilators: {df['Number of ventilators or ABD'].sum()}")

# 2. Zone-wise Analysis
zone_stats = df.groupby('Zone Name').agg({
    'Total no. of beds': 'sum',
    'Number of COVID beds': 'sum',
    'Number of ICU beds': 'sum',
    'Number of ventilators or ABD': 'sum'
}).sort_values('Total no. of beds', ascending=False)

print("\n=== Zone-wise Statistics ===")
print(zone_stats)

# Visualization 1: Total Beds by Zone
plt.figure(figsize=(14, 8))
sns.barplot(x=zone_stats['Total no. of beds'], y=zone_stats.index, 
            hue=zone_stats.index, palette='viridis', legend=False)
plt.title('Total Hospital Beds by Zone', fontsize=16)
plt.xlabel('Total Beds', fontsize=14)
plt.ylabel('Zone', fontsize=14)
plt.tight_layout()
plt.savefig('total_beds_by_zone.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: total_beds_by_zone.png")

# Visualization 2: COVID Beds vs ICU Beds by Zone
plt.figure(figsize=(14, 8))
zone_stats[['Number of COVID beds', 'Number of ICU beds']].plot(kind='barh', stacked=True)
plt.title('COVID Beds vs ICU Beds by Zone', fontsize=16)
plt.xlabel('Number of Beds', fontsize=14)
plt.ylabel('Zone', fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('covid_icu_beds_by_zone.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: covid_icu_beds_by_zone.png")

# 3. Hospital Type Analysis
hospital_type = df['Type of hospital_Private or Public'].value_counts()

# Visualization 3: Hospital Type Distribution
plt.figure(figsize=(10, 6))
sns.countplot(y='Type of hospital_Private or Public', data=df, order=hospital_type.index)
plt.title('Distribution of Hospital Types', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Hospital Type', fontsize=14)
plt.tight_layout()
plt.savefig('hospital_type_distribution.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: hospital_type_distribution.png")

# 4. Top Hospitals Analysis
top_hospitals = df.sort_values('Total no. of beds', ascending=False).head(10)

# Visualization 4: Top Hospitals by Capacity
plt.figure(figsize=(14, 8))
sns.barplot(x='Total no. of beds', y='Type of hospital_Private or Public', data=top_hospitals, 
            hue='Zone Name', dodge=False, palette='muted', legend=True)
plt.title('Top 10 Hospitals by Total Bed Capacity', fontsize=16)
plt.xlabel('Total Beds', fontsize=14)
plt.ylabel('Hospital', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('top_hospitals_by_capacity.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: top_hospitals_by_capacity.png")

# 5. Ventilator Availability
ventilator_stats = df[df['Number of ventilators or ABD'] > 0].sort_values('Number of ventilators or ABD', ascending=False)

# Visualization 5: Hospitals with Ventilators
plt.figure(figsize=(14, 8))
sns.scatterplot(x='Number of ventilators or ABD', y='Zone Name', 
                size='Number of ICU beds', hue='Total no. of beds',
                data=ventilator_stats, sizes=(50, 500), palette='coolwarm')
plt.title('Ventilator Availability by Zone', fontsize=16)
plt.xlabel('Number of Ventilators', fontsize=14)
plt.ylabel('Zone', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('ventilator_availability.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: ventilator_availability.png")

# 6. Correlation Analysis
corr = df[numeric_cols].corr()

# Visualization 6: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Between Bed Types', fontsize=16)
plt.tight_layout()
plt.savefig('bed_type_correlation.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: bed_type_correlation.png")

# 7. COVID Bed Utilization
df['COVID_bed_utilization'] = df['Number of COVID beds'] / df['Total no. of beds']
df['COVID_bed_utilization'] = df['COVID_bed_utilization'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Visualization 7: COVID Bed Utilization by Zone
plt.figure(figsize=(14, 8))
sns.boxplot(x='COVID_bed_utilization', y='Zone Name', data=df)
plt.title('COVID Bed Utilization by Zone', fontsize=16)
plt.xlabel('COVID Bed Utilization Ratio', fontsize=14)
plt.ylabel('Zone', fontsize=14)
plt.tight_layout()
plt.savefig('covid_bed_utilization.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: covid_bed_utilization.png")

print("\nEDA completed successfully! Visualizations saved as PNG files.")
