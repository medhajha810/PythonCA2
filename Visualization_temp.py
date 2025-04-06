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
df.columns = df.columns.str.strip()
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
numeric_cols = ['Total no. of beds', 'Number of COVID beds', 'Number of ICU beds', 'Number of ventilators or ABD']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

# 1. Pie Chart - Hospital Type Distribution
plt.figure()
hospital_type = df['Type of hospital_Private or Public'].value_counts()
plt.pie(hospital_type, labels=hospital_type.index, autopct='%1.1f%%', startangle=90)
plt.title('Hospital Type Distribution')
plt.show()

# 2. Box Plot - Bed Distribution by Zone
plt.figure()
sns.boxplot(x='Zone Name', y='Total no. of beds', data=df)
plt.title('Bed Distribution by Zone')
plt.xticks(rotation=45)
plt.show()

print("\nVisualizations displayed successfully!")
