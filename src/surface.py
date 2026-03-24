import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import plotly.graph_objects as go


def select_otm(df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """
    Keep only OTM options for each strike — the market-standard convention
    for constructing a vol surface, since OTM options are most liquid.

    - OTM calls: strike >= spot
    - OTM puts:  strike <  spot
    """
    df = df.copy()
    df["moneyness"] = df["strike"] / spot

    otm = df[
        ((df["flag"] == "c") & (df["moneyness"] >= 1.0)) |
        ((df["flag"] == "p") & (df["moneyness"] < 1.0))
    ]
    return otm.reset_index(drop=True)


def build_surface(df: pd.DataFrame, spot: float, output_path: str = "vol_surface.png") -> pd.DataFrame:
    """
    Interpolate IV onto a regular grid and plot the vol surface.

    Produces:
    - 3D surface plot (IV vs moneyness vs time to expiry)
    - 2D heatmap (same data, easier to read precise levels)
    """
    otm = select_otm(df, spot)

    x = otm["moneyness"].values
    y = otm["tte"].values
    z = otm["iv"].values

    # Interpolate scattered points onto a uniform grid
    xi = np.linspace(x.min(), x.max(), 60)
    yi = np.linspace(y.min(), y.max(), 60)
    Xi, Yi = np.meshgrid(xi, yi)
    method = "cubic" if len(x) >= 10 else "linear"
    Zi = griddata((x, y), z, (Xi, Yi), method=method)

    fig = plt.figure(figsize=(15, 6))

    # --- 3D surface ---
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(Xi, Yi, Zi, cmap="viridis", alpha=0.88, edgecolor="none")
    ax1.scatter(x, y, z, color="red", s=8, zorder=5, label="Market quotes")
    ax1.set_xlabel("Moneyness (K/S)")
    ax1.set_ylabel("Time to Expiry (yrs)")
    ax1.set_zlabel("Implied Volatility")
    ax1.set_title("Implied Volatility Surface")

    # --- 2D heatmap ---
    ax2 = fig.add_subplot(122)
    cf = ax2.contourf(Xi, Yi, Zi, levels=25, cmap="viridis")
    ax2.scatter(x, y, c="red", s=8, zorder=5, label="Market quotes")
    plt.colorbar(cf, ax=ax2, label="Implied Volatility")
    ax2.set_xlabel("Moneyness (K/S)")
    ax2.set_ylabel("Time to Expiry (yrs)")
    ax2.set_title("IV Heatmap")
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Surface saved to {output_path}")

    return otm


def build_surface_plotly(df: pd.DataFrame, spot: float, output_path: str = "vol_surface.html") -> None:
    """
    Interactive Plotly vol surface — rotate, zoom, and hover to inspect IV levels.

    Produces an HTML file you can open in any browser.
    """
    otm = select_otm(df, spot)

    x = otm["moneyness"].values
    y = otm["tte"].values
    z = otm["iv"].values

    xi = np.linspace(x.min(), x.max(), 60)
    yi = np.linspace(y.min(), y.max(), 60)
    Xi, Yi = np.meshgrid(xi, yi)
    method = "cubic" if len(x) >= 10 else "linear"
    Zi = griddata((x, y), z, (Xi, Yi), method=method)

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=Xi, y=Yi, z=Zi,
        colorscale="Viridis",
        opacity=0.88,
        colorbar=dict(title="Implied Vol"),
    ))

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=3, color="red"),
        name="Market quotes",
    ))

    fig.update_layout(
        title="Implied Volatility Surface",
        scene=dict(
            xaxis_title="Moneyness (K/S)",
            yaxis_title="Time to Expiry (yrs)",
            zaxis_title="Implied Volatility",
        ),
        width=1000,
        height=700,
    )

    fig.write_html(output_path)
    fig.show()
    print(f"Interactive surface saved to {output_path}")
