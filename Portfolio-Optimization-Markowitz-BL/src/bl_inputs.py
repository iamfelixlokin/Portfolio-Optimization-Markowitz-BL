import pandas as pd

prices = pd.read_csv("../data/stocks_data.csv", index_col=0, parse_dates=True)

returns = prices.pct_change().dropna()

mu = returns.mean() * 252

cov = returns.cov() * 252

print("Expected returns:")
print(mu)

print("\nCovariance matrix:")
print(cov)
