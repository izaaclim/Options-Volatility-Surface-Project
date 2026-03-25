"""
Loader for OptionsDX-style end-of-day options CSV data.

Expected columns (with bracket notation):
    [QUOTE_DATE], [QUOTE_UNIXTIME], [UNDERLYING_LAST],
    [EXPIRE_DATE], [DTE],
    [C_BID], [C_ASK], [C_IV], [C_DELTA], [C_VOLUME],
    [P_BID], [P_ASK], [P_IV], [P_DELTA], [P_VOLUME],
    [STRIKE], [STRIKE_DISTANCE_PCT]
"""

import pandas as pd
from pathlib import Path


# Columns we actually use downstream; everything else is dropped.
_KEEP = [
    "quote_date",
    "underlying_last",
    "expire_date",
    "dte",
    "strike",
    "strike_distance_pct",
    "c_bid", "c_ask", "c_iv", "c_delta", "c_volume",
    "p_bid", "p_ask", "p_iv", "p_delta", "p_volume",
]


def _strip_brackets(name: str) -> str:
    """[QUOTE_DATE] -> quote_date"""
    return name.strip().strip("[]").lower()


def load_options_data(path: str | Path, drop_zero_dte: bool = True) -> pd.DataFrame:
    """
    Load and clean an OptionsDX-style options CSV.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    drop_zero_dte : bool
        Drop rows where DTE == 0 (same-day expiry). These typically
        have unreliable or missing IVs and are not useful for forward
        vol calculations. Default True.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with standardised lowercase column names.
        One row = one (quote_date, expire_date, strike) combination,
        with both call and put fields side-by-side.
    """
    df = pd.read_csv(
        path,
        # C_SIZE / P_SIZE are "N x M" strings — keep everything as object first
        dtype=str,
        on_bad_lines="warn",
    )

    # Normalise column names
    df.columns = [_strip_brackets(c) for c in df.columns]

    # Keep only columns we need (ignore missing ones gracefully)
    keep = [c for c in _KEEP if c in df.columns]
    df = df[keep].copy()

    # Parse dates
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["expire_date"] = pd.to_datetime(df["expire_date"])

    # Numeric coercion — blank strings become NaN
    numeric_cols = [
        "underlying_last", "dte", "strike", "strike_distance_pct",
        "c_bid", "c_ask", "c_iv", "c_delta", "c_volume",
        "p_bid", "p_ask", "p_iv", "p_delta", "p_volume",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if drop_zero_dte:
        df = df[df["dte"] > 0]

    df = df.reset_index(drop=True)
    return df


def summarise(df: pd.DataFrame) -> None:
    """Print a quick sanity-check summary of a loaded options DataFrame."""
    print(f"Rows          : {len(df):,}")
    print(f"Quote dates   : {df['quote_date'].nunique()} "
          f"({df['quote_date'].min().date()} → {df['quote_date'].max().date()})")
    print(f"Expiry dates  : {df['expire_date'].nunique()}")
    print(f"Strikes       : {df['strike'].nunique()} "
          f"({df['strike'].min()} – {df['strike'].max()})")
    print(f"Underlying Δ  : {df['underlying_last'].min():.2f} – {df['underlying_last'].max():.2f}")

    iv_ok_c = df["c_iv"].notna().sum()
    iv_ok_p = df["p_iv"].notna().sum()
    print(f"C_IV non-null : {iv_ok_c:,} ({100*iv_ok_c/len(df):.1f}%)")
    print(f"P_IV non-null : {iv_ok_p:,} ({100*iv_ok_p/len(df):.1f}%)")

    print(f"\nDTE range     : {int(df['dte'].min())} – {int(df['dte'].max())} days")
    dte_counts = df.groupby("quote_date")["expire_date"].nunique()
    print(f"Expiries/day  : min={dte_counts.min()}, "
          f"median={dte_counts.median():.0f}, max={dte_counts.max()}")
