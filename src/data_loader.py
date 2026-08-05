"""
data_loader.py

This module provides a reusable DataLoader class for loading datasets
and performing initial data quality checks.

Author: Shihab Hossen
Project: Real Estate Intelligence Platform
"""
from pathlib import Path
import pandas as pd

class DataLoader:
    """
    A utility class for loading datasets and performing
    basic data quality assessments.
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """
        Load a CSV dataset.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.

        Raises
        ------
        FileNotFoundError
            If the dataset file does not exist.
        """

        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.file_path}")

        self.df = pd.read_csv(self.file_path)

        return self.df

    def dataset_info(self) -> dict:
        """
        Return general dataset information.
        """

        return {
            "Rows": self.df.shape[0],
            "Columns": self.df.shape[1],
            "Memory (MB)": round(
                self.df.memory_usage(deep=True).sum() / 1024**2,
                2,
            ),
        }

    def missing_summary(self) -> pd.DataFrame:
        """
        Return missing value statistics.
        """

        missing = self.df.isnull().sum()

        summary = pd.DataFrame(
            {
                "Missing Count": missing,
                "Missing Percentage": (missing / len(self.df) * 100).round(2),
            }
        )

        return summary.sort_values(
            by="Missing Percentage",
            ascending=False,
        )

    def duplicate_summary(self) -> dict:
        """
        Return duplicate statistics.
        """

        duplicates = self.df.duplicated().sum()

        return {
            "Duplicate Rows": duplicates,
            "Duplicate Percentage": round(
                duplicates / len(self.df) * 100,
                2,
            ),
        }

    def statistical_summary(self) -> pd.DataFrame:
        """
        Return descriptive statistics.
        """

        return self.df.describe(include="all").T
