"""Random number generator."""

from __future__ import annotations

import pytest

from app.rng.generator import RngError, generate_numbers


def test_int_bounds() -> None:
    values = generate_numbers(1000, 10, 20, seed=1)
    assert len(values) == 1000
    assert all(isinstance(v, int) and 10 <= v <= 20 for v in values)
    assert set(values) == set(range(10, 21))


def test_float_bounds_and_decimals() -> None:
    values = generate_numbers(500, 0, 1, kind="float", decimals=3, seed=2)
    assert all(0 <= v <= 1 for v in values)
    assert all(round(v, 3) == v for v in values)


def test_unique_ints() -> None:
    values = generate_numbers(20, 1, 20, unique=True, seed=3)
    assert sorted(values) == list(range(1, 21))


def test_unique_on_huge_range_is_cheap() -> None:
    values = generate_numbers(1000, 0, 10**12, unique=True, seed=4)
    assert len(set(values)) == 1000


def test_unique_floats_respect_grid() -> None:
    values = generate_numbers(11, 0, 1, kind="float", decimals=1, unique=True, seed=5)
    assert sorted(values) == [i / 10 for i in range(11)]


def test_seed_is_reproducible() -> None:
    assert generate_numbers(10, 1, 100, seed=42) == generate_numbers(10, 1, 100, seed=42)
    assert generate_numbers(10, 1, 100, seed=42) != generate_numbers(10, 1, 100, seed=43)


def test_secure_mode_works() -> None:
    values = generate_numbers(50, 1, 6, seed=1, secure=True)
    assert all(1 <= v <= 6 for v in values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": 0, "low": 1, "high": 2},
        {"count": 1_000_001, "low": 1, "high": 2},
        {"count": 1, "low": 5, "high": 1},
        {"count": 11, "low": 1, "high": 10, "unique": True},
        {"count": 1, "low": 1.5, "high": 3},
        {"count": 12, "low": 0, "high": 1, "kind": "float", "decimals": 1, "unique": True},
        {"count": 1, "low": 0.11, "high": 0.19, "kind": "float", "decimals": 0},
        {"count": 1, "low": 0, "high": 1, "kind": "float", "decimals": 99},
    ],
)
def test_validation_errors(kwargs: dict) -> None:
    with pytest.raises(RngError):
        generate_numbers(**kwargs)
