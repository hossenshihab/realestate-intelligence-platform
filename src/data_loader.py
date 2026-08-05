"""
data_loader.py

This module provides a reusable DataLoader class for loading datasets
and performing initial data quality checks.

Author: Shihab Hossen
Project: Real Estate Intelligence Platform
"""

from functools import wraps
from pathlib import Path
import pandas as pd


def ensure_data_loaded(method):
    """Decorator to ensure data is loaded before running quality checks."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.df is None:
            raise ValueError(f"Data not loaded yet. Please call '.load_data()' first.")
        return method(self, *args, **kwargs)

    return wrapper


class DataLoader:
    """
    A utility class for loading datasets and performing
    basic data quality assessments.
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.df: pd.DataFrame | None = None

    def load_data(self, **kwargs) -> pd.DataFrame:
        """
        Load a dataset based on its file extension.

        Supports CSV, Excel, and Parquet. Pass extra pandas arguments via kwargs.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.file_path}")

        ext = self.file_path.suffix.lower()
        if ext == ".csv":
            self.df = pd.read_csv(self.file_path, **kwargs)
        elif ext in [".xlsx", ".xls"]:
            self.df = pd.read_excel(self.file_path, **kwargs)
        elif ext == ".parquet":
            self.df = pd.read_parquet(self.file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format extension: {ext}")

        return self.df

    @ensure_data_loaded
    def dataset_info(self) -> dict:
        """Return general dataset information."""
        return {
            "Rows": self.df.shape[0],
            "Columns": self.df.shape[1],
            "Memory (MB)": round(self.df.memory_usage(deep=True).sum() / 1024**2, 2),
        }

    @ensure_data_loaded
    def missing_summary(self) -> pd.DataFrame:
        """Return missing value statistics."""
        missing = self.df.isnull().sum()

        # Avoid division-by-zero error if the dataset is completely empty
        total_rows = len(self.df) or 1

        summary = pd.DataFrame(
            {
                "Missing Count": missing,
                "Missing Percentage": (missing / total_rows * 100).round(2),
            }
        )

        return summary.sort_values(by="Missing Percentage", ascending=False)

    @ensure_data_loaded
    def duplicate_summary(self) -> dict:
        """Return duplicate statistics."""
        duplicates = self.df.duplicated().sum()
        total_rows = len(self.df) or 1

        return {
            "Duplicate Rows": duplicates,
            "Duplicate Percentage": round((duplicates / total_rows) * 100, 2),
        }

    @ensure_data_loaded
    def statistical_summary(self, include_numeric: bool = True) -> pd.DataFrame:
        """
        Return descriptive statistics separated by data type to avoid NaN mixing.
        """
        if include_numeric:
            return self.df.describe(include=[object, "category"]).T
        return self.df.describe(exclude=[object, "category"]).T
