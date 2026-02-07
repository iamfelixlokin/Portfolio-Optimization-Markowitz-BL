from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths 
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR    = BASE_DIR / "data"
OUTPUT_DIR  = BASE_DIR / "outputs"
WEIGHTS_DIR = OUTPUT_DIR / "weights"
PERF_DIR    = OUTPUT_DIR / "performance"

CSV_PATH = DATA_DIR / "stocks_data.csv"

MARKOWITZ_W_PATH = WEIGHTS_DIR / "weights_markowitz.csv"
BL_W_PATH        = WEIGHTS_DIR / "weights_bl.csv"

OUT_EQUITY_CSV = PERF_DIR / "equity_curves.csv"
OUT_DD_CSV     = PERF_DIR / "drawdowns.csv"
OUT_EQUITY_PNG = BASE_DIR / "equity_curves.png"
OUT_DD_PNG     = BASE_DIR / "drawdowns.png"

TEST_START = "2025-01-01"
TEST_END   = "2025-12-31"

# Helpers
def load_prices(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices = prices.select_dtypes(include=[np.number]).dropna(how="any")
    return prices.sort_index()


def load_weights(path: Path, tickers: list[str]) -> np.ndarray:
    df = pd.read_csv(path)
    w_map = dict(zip(df["ticker"], df["weight"]))
    w = np.array([w_map.get(t, 0.0) for t in tickers], dtype=float)

    # normalize
    if abs(w.sum()) > 1e-12:
        w /= w.sum()
    return w


def equity_curve(daily_returns: pd.Series) -> pd.Series:
    return (1.0 + daily_returns).cumprod()


def drawdown_curve(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0

# Main
def main():
    # Load prices & returns
    prices = load_prices(CSV_PATH)
    tickers = list(prices.columns)

    returns = prices.pct_change().dropna()
    test_rets = returns.loc[TEST_START:TEST_END]

    if test_rets.empty:
        raise ValueError("Test period has no data. Check TEST_START / TEST_END.")

    # Equal Weight baseline
    n = len(tickers)
    w_eq = np.ones(n) / n
    port_eq = pd.Series(
        test_rets.values @ w_eq,
        index=test_rets.index,
        name="EqualWeight"
    )

    # Markowitz
    w_mk = load_weights(MARKOWITZ_W_PATH, tickers)
    port_mk = pd.Series(
        test_rets.values @ w_mk,
        index=test_rets.index,
        name="Markowitz"
    )

    # Black-Litterman
    curves = {
        "EqualWeight": port_eq,
        "Markowitz": port_mk
    }

    if BL_W_PATH.exists():
        w_bl = load_weights(BL_W_PATH, tickers)
        port_bl = pd.Series(
            test_rets.values @ w_bl,
            index=test_rets.index,
            name="BlackLitterman"
        )
        curves["BlackLitterman"] = port_bl
    else:
        print("[Info] weights_bl.csv not found → skip BL curve")

    # Equity & Drawdown
    equity_df = pd.DataFrame({k: equity_curve(v) for k, v in curves.items()})
    dd_df = pd.DataFrame({k: drawdown_curve(equity_df[k]) for k in equity_df.columns})

    PERF_DIR.mkdir(parents=True, exist_ok=True)

    equity_df.to_csv(OUT_EQUITY_CSV)
    dd_df.to_csv(OUT_DD_CSV)

    # Plot Equity
    plt.figure(figsize=(12, 6))
    for col in equity_df.columns:
        plt.plot(equity_df.index, equity_df[col], label=col)

    plt.title("Equity Curves (Out-of-sample)")
    plt.xlabel("Date")
    plt.ylabel("Equity (Start = 1.0)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_EQUITY_PNG, dpi=200)
    plt.show()

    # Plot Drawdown
    plt.figure(figsize=(12, 5))
    for col in dd_df.columns:
        plt.plot(dd_df.index, dd_df[col], label=col)

    plt.title("Drawdown Curves (Out-of-sample)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DD_PNG, dpi=200)
    plt.show()

    print("\nSaved:")
    print(f" - {OUT_EQUITY_CSV}")
    print(f" - {OUT_DD_CSV}")
    print(f" - {OUT_EQUITY_PNG}")
    print(f" - {OUT_DD_PNG}")


if __name__ == "__main__":
    main()
