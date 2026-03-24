from src.fetch import get_options_chain, get_risk_free_rate
from src.iv import compute_mid, filter_options, compute_iv
from src.surface import build_surface, build_surface_plotly

TICKER = "SPY"


def main():
    print(f"Fetching options chain for {TICKER}...")
    spot, options = get_options_chain(TICKER)
    rfr = get_risk_free_rate()
    print(f"Spot: ${spot:.2f} | Risk-free rate: {rfr:.2%}")
    print(f"Raw options: {len(options)}")

    options = compute_mid(options)
    options = filter_options(options, spot)
    print(f"After liquidity filter: {len(options)}")

    options = compute_iv(options, spot, rfr)
    print(f"After IV computation: {len(options)}")

    print("Building vol surface...")
    build_surface(options, spot)
    build_surface_plotly(options, spot)


if __name__ == "__main__":
    main() 






