"""Shared utilities for the reproducible analysis code package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def repo_root() -> Path:
    """Return repository root, assuming this file is stored in ./scripts."""
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path, section: str) -> dict[str, Any]:
    """Load a named section from a JSON configuration file."""
    path = Path(config_path)
    if not path.is_absolute():
        path = repo_root() / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if section not in config:
        available = ", ".join(config.keys())
        raise KeyError(f"Section '{section}' was not found in {path}. Available sections: {available}")
    return config[section]


def resolve_path(value: str | Path, base: str | Path | None = None) -> Path:
    """Resolve a path. Relative paths are interpreted from the repository root by default."""
    path = Path(value)
    if path.is_absolute():
        return path
    root = Path(base) if base is not None else repo_root()
    return (root / path).resolve()


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def as_list(value: Any, name: str = "value") -> list[str]:
    """Return a JSON value as a list of strings."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"'{name}' must be a list in the config file.")
    return [str(item) for item in value]


def safe_name(value: Any) -> str:
    """Create a safe filename fragment."""
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(value))


def find_station_col(columns: Iterable[Any]) -> Any | None:
    """Find a likely station/name column in table headers."""
    for col in columns:
        text = str(col).strip().lower().replace(" ", "_")
        if text in {"station", "station_name", "station_full", "name", "id", "id_point"}:
            return col
        if "station" in text:
            return col
    return None


def get_station_column(df: pd.DataFrame, configured: str | None = None) -> Any:
    """Return configured station column, or the first column when not configured."""
    if configured:
        if configured not in df.columns:
            raise KeyError(f"Configured station column '{configured}' not found. Available columns: {list(df.columns)}")
        return configured
    return df.columns[0]


def require_columns(df: pd.DataFrame, columns: list[str], file_label: str) -> None:
    """Raise a clear error if any expected column is missing."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {file_label}: {missing}")


def numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return selected columns as numeric values."""
    out = df[columns].copy()
    out = out.replace(r"^\s*$", np.nan, regex=True)
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def zscore(series: pd.Series) -> pd.Series:
    """Population-standard-deviation z-score, matching sklearn StandardScaler."""
    s = pd.to_numeric(series, errors="coerce")
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s - s.mean()
    return (s - s.mean()) / sd


def build_equation_label(
    left_label: str,
    params: pd.Series | dict[str, float],
    terms: list[str],
    sig: int = 3,
    terms_per_line: int = 2,
) -> str:
    """Build a compact multiline equation string for plots and Excel sheets."""
    p = pd.Series(params).reindex(["const"] + terms).astype(float).fillna(0.0)
    pieces = [f"{p['const']:.{sig}g}"] + [f"{p[t]:+.{sig}g}*{t}" for t in terms]
    chunks = [pieces[1:][i:i + terms_per_line] for i in range(0, len(pieces[1:]), terms_per_line)]
    if not chunks:
        return f"{left_label} = {pieces[0]}"
    lines = [f"{left_label} = {pieces[0]} " + " ".join(chunks[0])]
    indent = " " * (len(str(left_label)) + 3)
    for chunk in chunks[1:]:
        lines.append(indent + " ".join(chunk))
    return "\n".join(lines)


def pretty_input_term(variable: str, log_variables: list[str]) -> str:
    """Display transformed variables clearly in equations."""
    return f"ln({variable})" if variable in log_variables else variable


def build_input_equation_label(
    left_label: str,
    intercept: float,
    coefficients: pd.Series,
    log_variables: list[str],
    sig: int = 3,
    terms_per_line: int = 2,
) -> str:
    """Build equation using raw variables and ln(variable) labels where applicable."""
    terms = [pretty_input_term(var, log_variables) for var in coefficients.index.tolist()]
    coefs = coefficients.values.astype(float)
    pieces = [f"{float(intercept):.{sig}g}"] + [f"{coefs[i]:+.{sig}g}*{terms[i]}" for i in range(len(terms))]
    chunks = [pieces[1:][i:i + terms_per_line] for i in range(0, len(pieces[1:]), terms_per_line)]
    if not chunks:
        return f"{left_label} = {pieces[0]}"
    lines = [f"{left_label} = {pieces[0]} " + " ".join(chunks[0])]
    indent = " " * (len(str(left_label)) + 3)
    for chunk in chunks[1:]:
        lines.append(indent + " ".join(chunk))
    return "\n".join(lines)
