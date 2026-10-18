"""Read CSV and XLSX files into raw dataframes."""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path

import pandas as pd

from app.data.errors import EmptyFileError, UnreadableFileError

logger = logging.getLogger(__name__)

# utf-8-sig is tried first: it reads plain UTF-8 exactly like utf-8 and also strips a BOM.
# latin-1 accepts any byte sequence, so it is the last resort.
CSV_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "cp1252", "latin-1")
CSV_EXTENSIONS = {".csv", ".txt"}
EXCEL_EXTENSIONS = {".xlsx"}
DEMO_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "demo_timeseries.csv"


def load_table(content: bytes, filename: str) -> pd.DataFrame:
    """Parse file bytes into a dataframe based on the file extension.

    Fully empty rows and columns are dropped. Raises `EmptyFileError` or
    `UnreadableFileError` with a user-friendly message.
    """
    suffix = Path(filename).suffix.lower()
    if not content or not content.strip():
        raise EmptyFileError()

    if suffix in EXCEL_EXTENSIONS:
        frame = _read_excel(content)
    elif suffix in CSV_EXTENSIONS:
        frame = _read_csv(content)
    else:
        raise UnreadableFileError(
            f"Unsupported file type '{suffix or filename}'. Please upload a .csv or .xlsx file."
        )

    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if frame.empty:
        raise EmptyFileError()
    frame.columns = [str(column).strip() for column in frame.columns]
    logger.info("Loaded table with %d rows and %d columns", len(frame), len(frame.columns))
    return frame.reset_index(drop=True)


def load_path(path: str | Path) -> pd.DataFrame:
    """Read a CSV/XLSX file from disk."""
    path = Path(path)
    return load_table(path.read_bytes(), path.name)


def _read_csv(content: bytes) -> pd.DataFrame:
    for encoding in CSV_ENCODINGS:
        try:
            text = content.decode(encoding)
        except UnicodeDecodeError:
            continue
        try:
            return pd.read_csv(io.StringIO(text), sep=_sniff_separator(text))
        except pd.errors.EmptyDataError as exc:
            raise EmptyFileError() from exc
        except (pd.errors.ParserError, ValueError) as exc:
            raise UnreadableFileError(
                "The CSV file could not be parsed. Check that it is a valid comma- or "
                "semicolon-separated file with a header row."
            ) from exc
    raise UnreadableFileError("The file encoding is not supported. Please save it as UTF-8.")


def _sniff_separator(text: str) -> str:
    sample = "\n".join(text.splitlines()[:20])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _read_excel(content: bytes) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(content), engine="openpyxl")
    except Exception as exc:  # openpyxl raises many unrelated exception types
        raise UnreadableFileError(
            "The Excel file could not be read. Please upload a valid .xlsx file."
        ) from exc
