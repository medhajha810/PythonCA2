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

# Set default font and style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('seaborn-v0_8-colorblind')



# 2. Box Plot with Visible X-axis (Fixed)
plt.figure(2, figsize=(14, 8))
sns.boxplot(x='Zone Name', y='Total no. of beds', data=df, 
           hue='Zone Name', palette='pastel', legend=False)
plt.title('Bed Distribution by Zone', fontsize=14, pad=15)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Zone Name', fontsize=10)
plt.ylabel('Number of Beds', fontsize=12)
plt.tight_layout()
plt.xticks(rotation=45)

# 3. Line Plot - COVID Beds Trend
plt.figure(3)
zone_stats = df.groupby('Zone Name')['Number of COVID beds'].sum().sort_values(ascending=False)
zone_stats.plot(kind='line', marker='o')
plt.title('COVID Beds Trend by Zone')
plt.ylabel('Number of COVID Beds')
plt.grid(True)

# 4. Scatter Plot - ICU vs Ventilators
plt.figure(4)
sns.scatterplot(x='Number of ICU beds', y='Number of ventilators or ABD', 
                hue='Zone Name', data=df, s=100)
plt.title('ICU Beds vs Ventilators')


# 6. Heatmap with Light Pastel Colors
plt.figure(6, figsize=(16, 10))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='YlGnBu', center=0,
           annot_kws={'size': 10, 'weight': 'bold', 'color':'black'},
           linewidths=0.5, linecolor='white',
           cbar_kws={'shrink': 0.7, 'pad': 0.02},
           vmin=-1, vmax=1)
plt.title('Correlation Matrix of Hospital Resources', 
         fontsize=16, pad=20, weight='bold', y=1.02)
plt.xticks(fontsize=10, rotation=45, ha='right')
plt.yticks(fontsize=10, rotation=0, va='center')
plt.xlabel('Resource Types', fontsize=12, labelpad=10)
plt.ylabel('Resource Types', fontsize=12, labelpad=10)
plt.subplots_adjust(top=0.5, bottom=0.15, left=0.25, right=0.75)

# 7. Enhanced Count Plot - Hospitals by Zone
plt.figure(7, figsize=(12, 8))
sns.countplot(y='Zone Name', data=df, order=df['Zone Name'].value_counts().index,
             hue='Zone Name', palette='viridis', legend=False)
plt.title('Hospitals Count by Zone', fontsize=14, pad=20)
plt.xlabel('Number of Hospitals', fontsize=12)
plt.ylabel('Zone', fontsize=12)



# 10. Enhanced Scatter Plot with Visible Legend
plt.close(10)
plt.figure(10, figsize=(14, 8))
scatter = sns.scatterplot(data=df, x='Number of ICU beds', y='Number of ventilators or ABD',
                         hue='Zone Name', size='Total no. of beds',
                         sizes=(50, 500), alpha=0.8, palette='tab20')
plt.title('ICU Beds vs Ventilators by Zone', fontsize=16, pad=25)
plt.xlabel('Number of ICU Beds', fontsize=12)
plt.ylabel('Number of Ventilators', fontsize=12)
plt.legend(bbox_to_anchor=(1.25, 1), loc='upper right', title='Zone Name')
plt.tight_layout()

# 11. Enhanced Pie Charts with Percentage Labels
plt.figure(11, figsize=(18, 14))
plt.subplots_adjust(top=0.85, hspace=0.4, wspace=0.4)  # Added more spacing

metrics = ['Total no. of beds', 'Number of ICU beds', 
          'Number of COVID beds', 'Number of ventilators or ABD']
titles = ['Total Beds', 'ICU Beds', 'COVID Beds', 'Ventilators']

# Get unique zones and assign vibrant colors
zones = df['Zone Name'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(zones)))  # More visible colors
zone_colors = dict(zip(zones, colors))

for i, (metric, title) in enumerate(zip(metrics, titles), 1):
    plt.subplot(2, 2, i)
    zone_stats = df.groupby('Zone Name')[metric].sum().sort_values(ascending=False)
    
    # Calculate percentages
    percentages = zone_stats/zone_stats.sum()*100
    
    # Plot pie with percentage labels on slices
    wedges, texts, autotexts = plt.pie(zone_stats,
            labels=None,
            autopct='%1.1f%%',
            colors=[zone_colors[zone] for zone in zone_stats.index],
            startangle=90,
            wedgeprops={'edgecolor':'black', 'linewidth':1},
            textprops={'fontsize':10, 'color':'black', 'weight':'bold'})
    
    # Adjust label positions
    for autotext in autotexts:
        autotext.set_color('black')
    
    plt.title(f'{title} Distribution\nTotal: {int(zone_stats.sum())}', 
             fontsize=12, pad=15)  # Reduced font size and padding
    
    plt.axis('equal')

# Add simple legend without percentages
handles = [plt.Rectangle((0,0),1,1, color=zone_colors[zone]) for zone in zones]
plt.figlegend(handles, zones,
             title='Zones',
             loc='center right',
             bbox_to_anchor=(1.0, 0.5),
             fontsize=10)  # Smaller font size

plt.suptitle('Hospital Resources by Zone', fontsize=16, y=0.98, weight='bold')  # Adjusted position

# 12. Visible Bar Chart
plt.figure(12, figsize=(14, 8))
zone_totals = df.groupby('Zone Name')[metrics].sum()
zone_totals.plot(kind='bar', stacked=True, 
                color=plt.cm.Pastel1.colors,
                edgecolor='black')
plt.title('Resource Distribution by Zone', 
         fontsize=16, pad=15, weight='bold')
plt.xlabel('Zone', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Resource Type', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()  # Explicitly show figure 12

# 13. Enhanced Histogram (Fixed)
plt.figure(13, figsize=(14, 8))
sns.histplot(data=df, x='Total no. of beds', bins=15, 
            kde=True, color='skyblue',
            edgecolor='black', alpha=0.8)
plt.title('Hospital Bed Capacity Distribution', fontsize=16, pad=20)
plt.xlabel('Number of Beds', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# 14. Restored Pair Plot
plt.figure(14, figsize=(12, 10))
sns.pairplot(df[metrics], diag_kind='kde',
            plot_kws={'alpha':0.7, 's':60, 'color':'teal'},
            diag_kws={'color':'salmon', 'fill':True})
plt.suptitle('Pairwise Relationship of Resources', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()  # Explicitly show figure 14

plt.show()
print("\nAll 14 visualizations displayed successfully! Close each plot window to continue.")
