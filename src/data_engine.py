"""
data_engine.py
--------------
Owns the in-memory, validated retail dataset for the running process.

There is NO external database and NO vector store per the project
constraints - this is a deliberately simple, single-process, in-memory
holder of a pandas DataFrame. It is the single source of truth that
analytics.py reads from and that ai_engine.py's evidence is derived from.

Thread-safety note: Flask's dev server / a single gunicorn worker will call
into this sequentially for a hackathon demo. A lock is still included so
concurrent uploads/queries don't corrupt state under a threaded server.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import threading
import pandas as pd

from .validators import validate_csv_bytes, validate_dataframe, ValidationResult


class DataEngine:
    """Holds the currently loaded, validated retail dataset."""

    def __init__(self):
        self._lock = threading.Lock()
        self._df: Optional[pd.DataFrame] = None
        self._last_validation: Optional[ValidationResult] = None
        self._source_filename: Optional[str] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_from_bytes(self, raw_bytes: bytes, filename: str = "upload.csv") -> ValidationResult:
        df, result = validate_csv_bytes(raw_bytes)
        with self._lock:
            if result.is_valid:
                self._df = df
                self._source_filename = filename
            self._last_validation = result
        return result

    def load_from_path(self, path: str) -> ValidationResult:
        with open(path, "rb") as f:
            raw = f.read()
        return self.load_from_bytes(raw, filename=path)

    def load_from_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        clean_df, result = validate_dataframe(df)
        with self._lock:
            if result.is_valid:
                self._df = clean_df
                self._source_filename = "in-memory"
            self._last_validation = result
        return result

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._df is not None and not self._df.empty

    def get_dataframe(self) -> pd.DataFrame:
        """Returns a COPY of the current dataset. Raises if nothing is loaded."""
        with self._lock:
            if self._df is None:
                raise RuntimeError("No dataset has been loaded yet. Upload a CSV via /api/upload first.")
            return self._df.copy()

    def get_last_validation(self) -> Optional[Dict]:
        with self._lock:
            return self._last_validation.to_dict() if self._last_validation else None

    def clear(self):
        with self._lock:
            self._df = None
            self._last_validation = None
            self._source_filename = None

    # ------------------------------------------------------------------
    # Convenience lookups used across analytics / intent detection
    # ------------------------------------------------------------------
    def list_products(self) -> List[Dict]:
        df = self.get_dataframe()
        out = (
            df[["product_id", "product_name", "category"]]
            .drop_duplicates()
            .sort_values("product_name")
        )
        return out.to_dict(orient="records")

    def list_stores(self) -> List[Dict]:
        df = self.get_dataframe()
        out = df[["store_id", "store_name"]].drop_duplicates().sort_values("store_id")
        return out.to_dict(orient="records")

    def list_categories(self) -> List[str]:
        df = self.get_dataframe()
        return sorted(df["category"].dropna().unique().tolist())

    def date_range(self) -> Dict[str, str]:
        df = self.get_dataframe()
        return {
            "start": df["date"].min().strftime("%Y-%m-%d"),
            "end": df["date"].max().strftime("%Y-%m-%d"),
        }

    def days_of_history(self) -> int:
        df = self.get_dataframe()
        return int((df["date"].max() - df["date"].min()).days) + 1

    def summary(self) -> Dict:
        df = self.get_dataframe()
        return {
            "source_filename": self._source_filename,
            "row_count": int(len(df)),
            "date_range": self.date_range(),
            "days_of_history": self.days_of_history(),
            "n_products": int(df["product_id"].nunique()),
            "n_stores": int(df["store_id"].nunique()),
            "n_categories": int(df["category"].nunique()),
        }


# A single process-wide instance. Imported by app.py and analytics callers.
data_engine = DataEngine()
