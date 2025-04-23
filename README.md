# Visualization_District_PGI

This Python script performs data preprocessing and visualization on the District PGI (Performance Grading Index) dataset.

## Overview

The script loads the dataset `District_PGI_Table_1.csv` with multiple encoding fallbacks to ensure compatibility. It then performs comprehensive data preprocessing including:

- Dropping irrelevant columns (e.g., unnamed columns)
- Handling missing values using forward fill (`ffill`), backward fill (`bfill`), and filling remaining missing values with appropriate defaults
- Cleaning string data by stripping whitespace
- Converting relevant columns to numeric types using `pd.to_numeric` with error coercion
- Removing duplicate rows using `drop_duplicates(inplace=True)`

After preprocessing, the script generates several visualizations to analyze the district scores:

- Histogram of overall district scores
- Box plots of scores by category
- Bar plot of top 10 districts by overall score
- Count plot of grades distribution
- Correlation heatmap of category scores
- Pair plot of PGI scores

## Usage

1. Ensure the dataset file `District_PGI_Table_1.csv` is in the same directory as the script.
2. Run the script using Python 3:

```bash
python Visualization_District_PGI.py
```

3. The script will display the visualizations sequentially. Close each plot window to proceed to the next.

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn

Install dependencies via pip if needed:

```bash
pip install pandas numpy matplotlib seaborn
```

## Notes

- The script uses robust encoding fallbacks to load the CSV file.
- Missing data is handled carefully to maintain data integrity.
- Visualizations use seaborn styles for better aesthetics.

