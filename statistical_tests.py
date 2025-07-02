import pandas as pd
from scipy.stats import ttest_ind, pearsonr

# Load merged dataset
df = pd.read_csv("merged/merged_data.csv")
df = df.dropna(subset=['value', 'closed_pnl'])

# Correlation between sentiment value and PnL
correlation, p_value = pearsonr(df['value'], df['closed_pnl'])
print(f"Correlation: {correlation:.2f}, P-value: {p_value:.4f}")

# Compare PnL in Fear vs Greed markets
fear_pnl = df[df['classification'].str.contains("Fear", na=False)]['closed_pnl']
greed_pnl = df[df['classification'].str.contains("Greed", na=False)]['closed_pnl']

t_stat, p_val = ttest_ind(fear_pnl, greed_pnl, equal_var=False)
print(f"T-test: t-stat={t_stat:.2f}, p-value={p_val:.4f}")
