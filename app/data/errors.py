"""User-facing exceptions raised while loading and preparing tabular data."""

from __future__ import annotations


class DataError(Exception):
    """Base class for data problems. `str(error)` is safe to show to the user."""


class EmptyFileError(DataError):
    """The file has no rows (or no content at all)."""

    def __init__(self, message: str = "The file is empty: it has no data rows.") -> None:
        super().__init__(message)


class UnreadableFileError(DataError):
    """The file could not be parsed as CSV or XLSX."""


class NoDateColumnError(DataError):
    """No column could be interpreted as dates."""

    def __init__(
        self,
        message: str = (
            "No date column found. Add a column with dates (for example 2024-01-31) "
            "or pick the date column manually."
        ),
    ) -> None:
        super().__init__(message)


class NoNumericColumnsError(DataError):
    """No column could be interpreted as numbers."""

    def __init__(
        self,
        message: str = "No numeric columns found. The file needs at least one column of numbers.",
    ) -> None:
        super().__init__(message)
