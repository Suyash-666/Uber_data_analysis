

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_data(file_path: str) -> pd.DataFrame:
    """Load Uber ride data from a CSV file."""
    return pd.read_csv(file_path)


def show_basic_info(df: pd.DataFrame) -> None:
    """Display shape, column info, and first rows."""
    print("\n=== Dataset Shape ===")
    print(df.shape)

    print("\n=== Dataset Info ===")
    print(df.info())

    print("\n=== First 5 Rows ===")
    print(df.head())


def check_data_types_and_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table with data types and missing values."""
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_values": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
        }
    )
    return summary.sort_values("missing_values", ascending=False)


def handle_missing_values(df: pd.DataFrame, drop_threshold: float = 0.6) -> pd.DataFrame:
    """Handle missing values using drop/fill strategies.

    - Drop columns where missing ratio exceeds `drop_threshold`.
    - Convert `datetime` and drop rows where datetime is invalid.
    - Fill numeric columns with median (fallback mean).
    - Fill categorical columns with mode (fallback "Unknown").
    """
    cleaned = df.copy()

    missing_ratio = cleaned.isna().mean()
    cols_to_drop = missing_ratio[missing_ratio > drop_threshold].index.tolist()
    if cols_to_drop:
        cleaned = cleaned.drop(columns=cols_to_drop)

    if "datetime" in cleaned.columns:
        cleaned["datetime"] = pd.to_datetime(cleaned["datetime"], errors="coerce")
        cleaned = cleaned.dropna(subset=["datetime"])

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = cleaned.select_dtypes(exclude=[np.number, "datetime"]).columns.tolist()

    for col in numeric_cols:
        if cleaned[col].isna().any():
            fill_value = cleaned[col].median()
            if pd.isna(fill_value):
                fill_value = cleaned[col].mean()
            cleaned[col] = cleaned[col].fillna(fill_value)

    for col in categorical_cols:
        if cleaned[col].isna().any():
            mode_vals = cleaned[col].mode(dropna=True)
            fill_value = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
            cleaned[col] = cleaned[col].fillna(fill_value)

    return cleaned


def parse_datetime_column(df: pd.DataFrame, column: str = "datetime") -> pd.DataFrame:
    """Convert datetime column to proper datetime format."""
    converted = df.copy()
    if column in converted.columns:
        converted[column] = pd.to_datetime(converted[column], errors="coerce")
    return converted


def extract_datetime_features(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    """Extract hour, day, month, year, day_of_week and time_of_day."""
    transformed = df.copy()
    if datetime_col not in transformed.columns:
        return transformed

    transformed[datetime_col] = pd.to_datetime(transformed[datetime_col], errors="coerce")
    transformed = transformed.dropna(subset=[datetime_col])

    transformed["hour"] = transformed[datetime_col].dt.hour
    transformed["day"] = transformed[datetime_col].dt.day
    transformed["month"] = transformed[datetime_col].dt.month
    transformed["year"] = transformed[datetime_col].dt.year
    transformed["day_of_week"] = transformed[datetime_col].dt.dayofweek

    bins = [-1, 5, 11, 17, 23]
    labels = ["Night", "Morning", "Afternoon", "Evening"]
    transformed["time_of_day"] = pd.cut(transformed["hour"], bins=bins, labels=labels)
    transformed["time_of_day"] = transformed["time_of_day"].astype(str)

    return transformed


def handle_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Remove outliers in specified columns using IQR method."""
    cleaned = df.copy()
    if columns is None:
        columns = [col for col in ["fare_amount", "distance"] if col in cleaned.columns]

    for col in columns:
        if col not in cleaned.columns:
            continue

        series = cleaned[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        cleaned = cleaned[(cleaned[col] >= lower) & (cleaned[col] <= upper)]

    return cleaned


def scale_numerical_features(
    df: pd.DataFrame,
    method: str = "standardize",
    exclude_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Standardize/normalize numerical features.

    method:
    - standardize: StandardScaler
    - normalize: MinMaxScaler
    """
    transformed = df.copy()

    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = transformed.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    if not numeric_cols:
        return transformed

    scaler = MinMaxScaler() if method == "normalize" else StandardScaler()
    transformed[numeric_cols] = scaler.fit_transform(transformed[numeric_cols])

    return transformed


def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """Save cleaned dataset to CSV."""
    df.to_csv(output_path, index=False)


def clean_uber_data(
    df: pd.DataFrame,
    missing_drop_threshold: float = 0.6,
    outlier_iqr_multiplier: float = 1.5,
    scaling_method: str = "standardize",
) -> pd.DataFrame:
    """Comprehensive cleaning pipeline for Uber data."""
    cleaned = df.copy()

    # 1) Handle missing values
    cleaned = handle_missing_values(cleaned, drop_threshold=missing_drop_threshold)

    # 2) Remove duplicates
    cleaned = cleaned.drop_duplicates()

    # 3) Convert datetime + 4/6) Feature extraction
    cleaned = parse_datetime_column(cleaned, column="datetime")
    cleaned = extract_datetime_features(cleaned, datetime_col="datetime")

    # 5) IQR outlier handling for fare_amount and distance
    cleaned = handle_outliers_iqr(
        cleaned,
        columns=["fare_amount", "distance"],
        iqr_multiplier=outlier_iqr_multiplier,
    )

    # Keep sensible non-negative values
    if "fare_amount" in cleaned.columns:
        cleaned = cleaned[cleaned["fare_amount"] >= 0]
    if "distance" in cleaned.columns:
        cleaned = cleaned[cleaned["distance"] >= 0]
    if "passenger_count" in cleaned.columns:
        cleaned = cleaned[cleaned["passenger_count"] > 0]

    # 7) Scale numerical features
    cleaned = scale_numerical_features(cleaned, method=scaling_method)

    return cleaned


def clean_and_save_uber_data(
    input_path: str,
    output_path: str = "uber_rides_cleaned.csv",
    missing_drop_threshold: float = 0.6,
    outlier_iqr_multiplier: float = 1.5,
    scaling_method: str = "standardize",
) -> pd.DataFrame:
    """Load dataset, clean it, and save cleaned output to CSV."""
    df = load_data(input_path)
    cleaned = clean_uber_data(
        df,
        missing_drop_threshold=missing_drop_threshold,
        outlier_iqr_multiplier=outlier_iqr_multiplier,
        scaling_method=scaling_method,
    )
    save_cleaned_data(cleaned, output_path)
    return cleaned
