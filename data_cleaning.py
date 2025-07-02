import pandas as pd

# Load the CSV files
df_historical = pd.read_csv('historical_data.csv')
df_fear = pd.read_csv('fear_greed_index.csv')

# Quick look at the data
print("Historical Data sample:")
print(df_historical.head())
print("\nFear Greed Index sample:")
print(df_fear.head())

# Check info and data types
print("\nHistorical Data info:")
print(df_historical.info())
print("\nFear Greed Index info:")
print(df_fear.info())

# Convert Date and Time columns to datetime if they exist
if 'Date' in df_historical.columns:
    df_historical['Date'] = pd.to_datetime(df_historical['Date'], errors='coerce')

if 'Time' in df_historical.columns:
    df_historical['Time'] = pd.to_datetime(df_historical['Time'], format='%H:%M:%S', errors='coerce').dt.time

if 'Date' in df_fear.columns:
    df_fear['Date'] = pd.to_datetime(df_fear['Date'], errors='coerce')

# Check missing values
print("\nMissing values in Historical Data:")
print(df_historical.isnull().sum())

print("\nMissing values in Fear Greed Index:")
print(df_fear.isnull().sum())

# Check duplicates
print("\nDuplicates in Historical Data:", df_historical.duplicated().sum())
print("Duplicates in Fear Greed Index:", df_fear.duplicated().sum())

# Drop duplicates if any
df_historical.drop_duplicates(inplace=True)
df_fear.drop_duplicates(inplace=True)

# Fill missing data or drop rows
df_historical.ffill( inplace=True)
df_fear.ffill (inplace=True)

# Normalize column names
df_historical.columns = df_historical.columns.str.lower().str.replace(' ', '_')
df_fear.columns = df_fear.columns.str.lower().str.replace(' ', '_')

print("\nHistorical Data Columns: ", df_historical.columns.tolist())


# ✅ MERGE STEP STARTS HERE

# Extract just the date part for merging
if 'time' in df_historical.columns:
    df_historical['trade_date'] = pd.to_datetime(df_historical['timestamp'], unit='ms').dt.date
else:
    df_historical['trade_date'] = pd.to_datetime(df_historical['timestamp'], unit='ms').dt.date

df_fear['date'] = pd.to_datetime(df_fear['date']).dt.date

# Merge on date
merged_df = pd.merge(df_historical, df_fear, left_on='trade_date', right_on='date', how='left')

# Drop extra column if needed
merged_df.drop(columns=['date'], inplace=True)

# Final output
print("\n✅ Merged Data Sample:")
print(merged_df.head())
print("\n✅ Merged Data Info:")
print(merged_df.info())

# Optionally, save the merged data to a new file
merged_df.to_csv('merged_data.csv', index=False)

# Check how many sentiment classifications are missing
missing_class = merged_df['classification'].isnull().sum()
total_rows = len(merged_df)
print(f"\n❓ Missing classification: {missing_class} out of {total_rows} total rows")

# See sample trade dates
print("\n📅 Sample trade_date values from historical data:")
print(df_historical['trade_date'].dropna().unique()[:5])

print("\n📅 Sample date values from fear/greed data:")
print(df_fear['date'].dropna().unique()[:5])
