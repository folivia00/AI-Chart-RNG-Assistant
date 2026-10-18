"""Random number generation: pure functions with seeding and a cryptographically secure mode.

How values are produced:

* Integers are drawn uniformly from [low, high] (both inclusive).
* Floats with `decimals` digits are drawn uniformly from the grid of values with that
  precision inside [low, high]; for example 0, 0.001, ..., 1 for decimals=3.
  Unique floats therefore means "unique after rounding": `count` may not exceed the
  number of grid points.
* Unique values are sampled without materialising the whole range, so memory stays
  O(count) even for huge ranges.
* Secure mode uses `random.SystemRandom` (backed by the OS, like `secrets`) and ignores
  the seed, because an OS entropy source cannot be seeded.
"""

from __future__ import annotations

import math
import random
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Literal

Kind = Literal["int", "float"]

MAX_COUNT = 1_000_000
MAX_ABS_VALUE = 1e12
MAX_DECIMALS = 6


class RngError(ValueError):
    """Invalid generator parameters. `str(error)` is safe to show to the user."""


def _float_grid(low: float, high: float, decimals: int) -> tuple[int, int]:
    """Return the inclusive integer bounds of the grid `value * 10**decimals`."""
    scale = Decimal(10) ** decimals
    grid_low = int((Decimal(str(low)) * scale).to_integral_value(ROUND_CEILING))
    grid_high = int((Decimal(str(high)) * scale).to_integral_value(ROUND_FLOOR))
    return grid_low, grid_high


def validate_params(
    count: int,
    low: float,
    high: float,
    kind: Kind = "int",
    decimals: int = 2,
    unique: bool = False,
) -> None:
    """Raise `RngError` if the parameters cannot produce a valid result."""
    if kind not in ("int", "float"):
        raise RngError("Kind must be 'int' or 'float'.")
    if not 1 <= count <= MAX_COUNT:
        raise RngError(f"Count must be between 1 and {MAX_COUNT:,}.")
    if not (math.isfinite(low) and math.isfinite(high)):
        raise RngError("Min and max must be finite numbers.")
    if abs(low) > MAX_ABS_VALUE or abs(high) > MAX_ABS_VALUE:
        raise RngError(f"Min and max must be within ±{MAX_ABS_VALUE:,.0f}.")
    if low > high:
        raise RngError("Min must be less than or equal to max.")

    if kind == "int":
        if not (float(low).is_integer() and float(high).is_integer()):
            raise RngError("For integers, min and max must be whole numbers.")
        size = int(high) - int(low) + 1
        if unique and count > size:
            raise RngError(
                f"Cannot pick {count:,} unique integers from a range of only {size:,} values."
            )
        return

    if not 0 <= decimals <= MAX_DECIMALS:
        raise RngError(f"Decimals must be between 0 and {MAX_DECIMALS}.")
    grid_low, grid_high = _float_grid(low, high, decimals)
    size = grid_high - grid_low + 1
    if size < 1:
        raise RngError(f"There are no numbers with {decimals} decimals between {low} and {high}.")
    if unique and count > size:
        raise RngError(
            f"Cannot pick {count:,} unique values: only {size:,} distinct numbers with "
            f"{decimals} decimals exist between {low} and {high}."
        )


def generate_numbers(
    count: int,
    low: float,
    high: float,
    kind: Kind = "int",
    decimals: int = 2,
    unique: bool = False,
    seed: int | None = None,
    secure: bool = False,
) -> list[int] | list[float]:
    """Generate `count` random numbers in [low, high].

    `decimals` applies to floats only. `seed` makes results reproducible and is
    ignored when `secure` is True. Raises `RngError` on invalid parameters.
    """
    validate_params(count, low, high, kind, decimals, unique)
    rng: random.Random = random.SystemRandom() if secure else random.Random(seed)

    if kind == "int":
        return _draw_ints(rng, count, int(low), int(high), unique)

    grid_low, grid_high = _float_grid(low, high, decimals)
    scale = 10**decimals
    return [
        round(value / scale, decimals)
        for value in _draw_ints(rng, count, grid_low, grid_high, unique)
    ]


def _draw_ints(rng: random.Random, count: int, low: int, high: int, unique: bool) -> list[int]:
    if unique:
        # random.sample on a range object selects indices lazily: O(count) memory.
        return rng.sample(range(low, high + 1), count)
    return [rng.randint(low, high) for _ in range(count)]
