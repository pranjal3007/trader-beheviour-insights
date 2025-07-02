import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure output directory exists
os.makedirs('outputs/plots', exist_ok=True)
sns.set(style="whitegrid")

# Load the merged dataset
merged_df = pd.read_csv('merged/merged_data.csv')
merged_df = merged_df.dropna(subset=['classification'])

# Check column names
print("Columns in merged data:", merged_df.columns.tolist())

# -------------------------------
# 1. Average / Median Closed PnL
# -------------------------------
pnl_stats = merged_df.groupby('classification')['closed_pnl'].agg(['mean', 'median'])
print("\n📈 Average/Median Closed PnL by Sentiment:\n", pnl_stats)

# -------------------------------
# 2. Number of Trades per Sentiment
# -------------------------------
trade_counts = merged_df['classification'].value_counts()
print("\n📊 Number of Trades by Sentiment:\n", trade_counts)
print("\nMissing classification values:", merged_df['classification'].isnull().sum())
print("Total rows:", len(merged_df))
print("\nSample trade_date values:", merged_df['trade_date'].dropna().unique()[:5])

# Bar chart
plt.figure()
trade_counts.plot(kind='bar', title='Number of Trades per Sentiment', color='skyblue')
plt.xlabel("Sentiment")
plt.ylabel("Number of Trades")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/plots/trade_counts_by_sentiment.png')
plt.show()

# -------------------------------
# 3. Win Rate (% of profitable trades)
# -------------------------------
merged_df['is_win'] = merged_df['closed_pnl'] > 0
win_rate = merged_df.groupby('classification')['is_win'].mean() * 100
print("\n✅ Win Rate by Sentiment (%):\n", win_rate)

plt.figure()
win_rate.plot(kind='bar', title='Win Rate by Sentiment', color='lightgreen')
plt.ylabel("Win Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/plots/win_rate_by_sentiment.png')
plt.show()

# -------------------------------
# 4. Leverage Usage (Start Position)
# -------------------------------
if 'start_position' in merged_df.columns:
    plt.figure()
    merged_df.boxplot(column='start_position', by='classification')
    plt.title('Leverage (Start Position) by Sentiment')
    plt.suptitle('')
    plt.xlabel('Sentiment')
    plt.ylabel('Start Position')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/plots/leverage_by_sentiment.png')
    plt.show()

# -------------------------------
# 5. Position Size (USD)
# -------------------------------
if 'size_usd' in merged_df.columns:
    plt.figure()
    merged_df.boxplot(column='size_usd', by='classification')
    plt.title('Position Size (USD) by Sentiment')
    plt.suptitle('')
    plt.xlabel('Sentiment')
    plt.ylabel('Position Size ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/plots/position_size_by_sentiment.png')
    plt.show()

# -------------------------------
# 6. Overall Summary Statistics
# -------------------------------
print("\n🔍 Overall Summary:")
print(merged_df[['closed_pnl', 'size_usd', 'start_position']].describe())
