"""Mutable dataset schema shared across modules.

Set once at data-load time (by data.py for freMTPL2freq, or by
internal_data.py from a JSON config). Models read it at fit time, so the
same pipeline works for any dataset.
"""
CATEGORICAL: list[str] = []
NUMERIC: list[str] = []


def set_schema(categorical: list[str], numeric: list[str]) -> None:
    CATEGORICAL.clear(); CATEGORICAL.extend(categorical)
    NUMERIC.clear(); NUMERIC.extend(numeric)


def features() -> list[str]:
    return CATEGORICAL + NUMERIC
