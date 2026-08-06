"""
visualization.py

Reusable visualization module for exploratory data analysis.

Author: Shihab Hossen
Project: Real Estate Intelligence Platform
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class DataVisualizer:
    """
    Reusable visualization class for exploratory data analysis.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        # Global plotting style
        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["axes.titlesize"] = 16
        plt.rcParams["axes.labelsize"] = 13

    def _validate_column(self, column: str):
        if column not in self.df.columns:
            raise ValueError(f"'{column}' does not exist in the dataset.")

    def histogram(
        self,
        column: str,
        bins: int = 30,
        kde: bool = True,
    ):
        """
        Plot histogram of a numerical feature.
        """

        self._validate_column(column)

        plt.figure()

        sns.histplot(
            data=self.df,
            x=column,
            bins=bins,
            kde=kde,
        )

        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    def boxplot(
        self,
        column: str,
    ):
        """
        Plot boxplot for detecting outliers.
        """

        self._validate_column(column)

        plt.figure()

        sns.boxplot(y=self.df[column])

        plt.title(f"Boxplot of {column}")
        plt.tight_layout()
        plt.show()

    def scatter(
        self,
        x: str,
        y: str,
    ):
        """
        Scatter plot between two numerical variables.
        """

        self._validate_column(x)
        self._validate_column(y)

        plt.figure()

        sns.scatterplot(
            data=self.df,
            x=x,
            y=y,
            alpha=0.6,
        )

        plt.title(f"{y} vs {x}")
        plt.tight_layout()
        plt.show()

    def bar_chart(
        self,
        column: str,
    ):
        """
        Plot frequency bar chart.
        """

        self._validate_column(column)

        plt.figure()

        self.df[column].value_counts().sort_index().plot(kind="bar")

        plt.title(f"{column} Distribution")
        plt.xlabel(column)
        plt.ylabel("Count")

        plt.tight_layout()
        plt.show()

    def correlation_heatmap(self):
        """
        Plot correlation heatmap.
        """

        corr = self.df.corr(numeric_only=True)

        plt.figure(figsize=(14, 10))

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
        )

        plt.title("Correlation Heatmap")

        plt.tight_layout()
        plt.show()

    def correlation_with_target(
        self,
        target: str,
    ) -> pd.DataFrame:
        """
        Return correlations with target.
        """

        self._validate_column(target)

        corr = (
            self.df.corr(numeric_only=True)[target]
            .drop(target)
            .sort_values(ascending=False)
            .to_frame(name="Correlation")
        )

        return corr

    def pairplot(
        self,
        columns: list[str],
    ):
        """
        Plot pairwise relationships.
        """

        sns.pairplot(self.df[columns])

        plt.show()
