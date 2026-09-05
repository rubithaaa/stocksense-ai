"""
validators.py
--------------
Deterministic validation of the retail CSV before anything else in the
pipeline (analytics, Gemini reasoning) is allowed to touch it.

Design goals:
- Fail loudly and specifically on malformed data (no silent corruption).
- Distinguish hard errors (block processing) from soft warnings (proceed,
  but surface to the user / to Gemini as data-quality caveats).
- Never raise raw pandas/numpy exceptions up to the Flask layer.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd


# Columns that MUST be present (by canonical lowercase name) for the pipeline
# to run at all.
REQUIRED_COLUMNS = [
    "date",
    "store_id",
    "product_id",
    "product_name",
    "category",
    "units_sold",
    "unit_price",
    "current_stock",
]

# Optional columns we understand and will use if present.
OPTIONAL_COLUMNS = [
    "store_name",
    "revenue",
    "reorder_level",
]

NUMERIC_COLUMNS = ["units_sold", "unit_price", "current_stock", "revenue", "reorder_level"]

MAX_REASONABLE_UNIT_PRICE = 1_000_000
MAX_REASONABLE_UNITS_SOLD = 1_000_000
MAX_REASONABLE_STOCK = 10_000_000


class ValidationResult:
    def __init__(self):
        self.is_valid: bool = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.row_count: int = 0
        self.rows_dropped: int = 0

    def add_error(self, msg: str):
        self.is_valid = False
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "row_count": self.row_count,
            "rows_dropped": self.rows_dropped,
        }


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def validate_csv_bytes(raw_bytes: bytes) -> Tuple[pd.DataFrame, ValidationResult]:
    """
    Parse raw CSV bytes into a DataFrame and validate it.
    Returns (dataframe_or_empty, ValidationResult). If ValidationResult.is_valid
    is False, the dataframe should NOT be used by the rest of the pipeline.
    """
    result = ValidationResult()

    if raw_bytes is None or len(raw_bytes) == 0:
        result.add_error("Uploaded file is empty.")
        return pd.DataFrame(), result

    try:
        import io
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except pd.errors.EmptyDataError:
        result.add_error("CSV file has no columns / is empty.")
        return pd.DataFrame(), result
    except pd.errors.ParserError as e:
        result.add_error(f"CSV could not be parsed: {e}")
        return pd.DataFrame(), result
    except UnicodeDecodeError:
        result.add_error("CSV encoding is not readable (expected UTF-8).")
        return pd.DataFrame(), result
    except Exception as e:  # noqa: BLE001 - surface any unexpected parse failure safely
        result.add_error(f"Unexpected error reading CSV: {e}")
        return pd.DataFrame(), result

    return validate_dataframe(df, result)


def validate_dataframe(df: pd.DataFrame, result: ValidationResult = None) -> Tuple[pd.DataFrame, ValidationResult]:
    if result is None:
        result = ValidationResult()

    if df is None or df.empty:
        result.add_error("Dataset has no rows.")
        return pd.DataFrame(), result

    df = _normalize_columns(df)
    result.row_count = len(df)

    # --- Required columns -------------------------------------------------
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        result.add_error(f"Missing required column(s): {', '.join(missing)}")
        return pd.DataFrame(), result  # cannot safely proceed

    # --- Date parsing -------------------------------------------------------
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    n_bad_dates = parsed_dates.isna().sum()
    if n_bad_dates > 0:
        result.add_warning(f"{n_bad_dates} row(s) had unparseable dates and will be dropped.")
    df = df.assign(date=parsed_dates)

    # --- Numeric coercion -----------------------------------------------
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            coerced = pd.to_numeric(df[col], errors="coerce")
            n_bad = coerced.isna().sum() - df[col].isna().sum()
            if n_bad > 0:
                result.add_warning(f"{n_bad} row(s) had non-numeric values in '{col}' and will be treated as missing.")
            df[col] = coerced

    # --- Drop rows that are unusable after coercion ------------------------
    before = len(df)
    required_numeric = ["units_sold", "unit_price", "current_stock"]
    df = df.dropna(subset=["date", "product_id", "store_id"] + required_numeric)
    after = len(df)
    if before - after > 0:
        result.rows_dropped += (before - after)
        result.add_warning(f"Dropped {before - after} row(s) with missing critical fields "
                            f"(date/product_id/store_id/units_sold/unit_price/current_stock).")

    if df.empty:
        result.add_error("No valid rows remained after validation/cleaning.")
        return pd.DataFrame(), result

    # --- Sanity range checks (soft warnings, not blocking) ------------------
    neg_units = (df["units_sold"] < 0).sum()
    if neg_units > 0:
        result.add_warning(f"{neg_units} row(s) have negative units_sold (possible returns) - kept as-is.")

    neg_stock = (df["current_stock"] < 0).sum()
    if neg_stock > 0:
        result.add_warning(f"{neg_stock} row(s) have negative current_stock - kept as-is but flagged as anomalous.")

    extreme_price = (df["unit_price"] > MAX_REASONABLE_UNIT_PRICE).sum()
    if extreme_price > 0:
        result.add_warning(f"{extreme_price} row(s) have unit_price above a sane threshold "
                            f"({MAX_REASONABLE_UNIT_PRICE}); please double check source data.")

    dup_count = df.duplicated(subset=["date", "store_id", "product_id"]).sum()
    if dup_count > 0:
        result.add_warning(f"{dup_count} duplicate (date, store_id, product_id) row(s) found; "
                            f"all duplicates are kept and will be aggregated at analysis time.")

    # --- Fill optional columns with safe defaults ---------------------------
    if "store_name" not in df.columns:
        df["store_name"] = df["store_id"].astype(str)
    if "revenue" not in df.columns or df["revenue"].isna().all():
        df["revenue"] = df["units_sold"] * df["unit_price"]
    else:
        # Fill any missing individual revenue values from units*price
        computed = df["units_sold"] * df["unit_price"]
        df["revenue"] = df["revenue"].fillna(computed)
    if "reorder_level" not in df.columns:
        df["reorder_level"] = pd.NA

    # Normalize text identifiers
    for col in ["product_id", "product_name", "category", "store_id", "store_name"]:
        df[col] = df[col].astype(str).str.strip()

    df = df.sort_values("date").reset_index(drop=True)

    return df, result
