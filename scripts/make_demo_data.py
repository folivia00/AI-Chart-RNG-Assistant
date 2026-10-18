"""Generate the synthetic demo dataset at data/demo_timeseries.csv.

The data is fully synthetic: a linear trend, yearly seasonality and Gaussian noise
for three daily series. A fixed seed keeps the output reproducible.

Usage:
    python scripts/make_demo_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20240101
START = "2023-01-01"
DAYS = 3 * 365 + 1
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "demo_timeseries.csv"


def build_demo_frame(seed: int = SEED) -> pd.DataFrame:
    """Return a daily dataframe with `date`, `sales`, `visitors` and `avg_price` columns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, periods=DAYS, freq="D")
    t = np.arange(DAYS)
    yearly = np.sin(2 * np.pi * t / 365.25)
    weekly = np.sin(2 * np.pi * t / 7)

    visitors = 1200 + 0.6 * t + 250 * yearly + 120 * weekly + rng.normal(0, 60, DAYS)
    avg_price = 24.0 + 0.004 * t - 1.5 * yearly + rng.normal(0, 0.6, DAYS)
    conversion = 0.045 + 0.005 * yearly + rng.normal(0, 0.002, DAYS)
    sales = visitors * conversion * avg_price

    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "sales": sales.round(2),
            "visitors": visitors.round().astype(int),
            "avg_price": avg_price.round(2),
        }
    )


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame = build_demo_frame()
    frame.to_csv(OUTPUT, index=False, lineterminator="\n")
    print(f"Wrote {len(frame)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
