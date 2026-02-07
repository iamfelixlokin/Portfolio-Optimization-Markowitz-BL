import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def download_data():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]

    prices = yf.download(
        tickers,
        start="2020-01-01",
        end="2025-12-31",
        auto_adjust=True
    )["Close"]

    returns = prices.pct_change().dropna()

    DATA_DIR.mkdir(exist_ok=True)
    prices.to_csv(DATA_DIR / "stocks_data.csv")
    returns.to_csv(DATA_DIR / "stocks_returns.csv")

    print("Data saved to /data")

if __name__ == "__main__":
    download_data()

