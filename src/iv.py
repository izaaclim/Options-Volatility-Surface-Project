import numpy as np
import pandas as pd
from py_vollib.black_scholes.implied_volatility import implied_volatility as bs_iv


def compute_mid(df: pd.DataFrame) -> pd.DataFrame:
    """Add mid-price column from bid and ask."""
    df = df.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2
    return df


def filter_options(
    df: pd.DataFrame,
    spot: float,
    min_volume: int = 1,
    max_spread_pct: float = 0.50,
    moneyness_range: tuple[float, float] = (0.70, 1.30),
) -> pd.DataFrame:
    """
    Remove illiquid and unreliable options before computing IV.

    Filters applied:
    - Zero bid (no market on the option)
    - Volume below threshold (stale prices)
    - Bid/ask spread too wide relative to mid (poor price quality)
    - Strike too far from spot (deep OTM IVs are noisy)
    """
    df = df.copy()
    df = df[df["bid"] > 0]
    df = df[df["volume"] >= min_volume]

    spread_pct = (df["ask"] - df["bid"]) / df["mid"]
    df = df[spread_pct <= max_spread_pct]

    lo, hi = moneyness_range
    df = df[(df["strike"] >= spot * lo) & (df["strike"] <= spot * hi)]

    return df.reset_index(drop=True)


def compute_iv(df: pd.DataFrame, spot: float, risk_free_rate: float) -> pd.DataFrame:
    """
    Back out implied volatility for each option via Black-Scholes inversion.

    Uses the mid-price as the target option price. Drops rows where
    the solver fails or returns an implausible IV.
    """
    df = df.copy()
    ivs = []

    for _, row in df.iterrows():
        try:
            iv = bs_iv(
                price=row["mid"],
                S=spot,
                K=row["strike"],
                t=row["tte"],
                r=risk_free_rate,
                flag=row["flag"],
            )
            ivs.append(iv if iv > 0 else np.nan)
        except Exception:
            ivs.append(np.nan)

    df["iv"] = ivs
    df = df.dropna(subset=["iv"])
    df = df[(df["iv"] > 0.01) & (df["iv"] < 5.0)]  # drop nonsensical IVs

    return df.reset_index(drop=True)

