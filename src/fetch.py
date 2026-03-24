import pandas as pd
import yfinance as yf
from datetime import datetime


def get_risk_free_rate() -> float:
    """Fetch the 13-week T-bill yield (^IRX) as a proxy for the risk-free rate."""
    irx = yf.Ticker("^IRX")
    rate = irx.history(period="1d")["Close"].iloc[-1] / 100
    return rate


def get_options_chain(ticker: str) -> tuple[float, pd.DataFrame]:
    """
    Fetch all available options expiries for a ticker and return a unified DataFrame.

    Returns:
        spot: current underlying price
        df: DataFrame with columns [expiry, tte, strike, flag, bid, ask, volume, openInterest]
    """
    tk = yf.Ticker(ticker)
    spot = tk.history(period="1d")["Close"].iloc[-1]
    today = datetime.today()

    records = []
    for expiry in tk.options:
        exp_date = datetime.strptime(expiry, "%Y-%m-%d")
        tte = (exp_date - today).days / 365.0

        if tte <= 0:
            continue

        chain = tk.option_chain(expiry)

        for flag, df in [("c", chain.calls), ("p", chain.puts)]:
            df = df.copy()
            df["flag"] = flag
            df["expiry"] = expiry
            df["tte"] = tte
            records.append(df)

    all_options = pd.concat(records, ignore_index=True)
    all_options = all_options[[
        "expiry", "tte", "strike", "flag",
        "bid", "ask", "volume", "openInterest",
    ]]

    return spot, all_options
