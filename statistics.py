"""Statistical analysis helpers for Uber ride data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""
    return df.describe(include="all").transpose()


def numerical_statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean, median, mode, std, and quartiles for numeric columns."""
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return pd.DataFrame()

    mode_vals = numeric_df.mode(dropna=True)
    first_mode = mode_vals.iloc[0] if not mode_vals.empty else pd.Series(index=numeric_df.columns, dtype=float)

    summary = pd.DataFrame(
        {
            "mean": numeric_df.mean(),
            "median": numeric_df.median(),
            "mode": first_mode,
            "std": numeric_df.std(),
            "variance": numeric_df.var(),
            "min": numeric_df.min(),
            "q1_25%": numeric_df.quantile(0.25),
            "q2_50%": numeric_df.quantile(0.50),
            "q3_75%": numeric_df.quantile(0.75),
            "max": numeric_df.max(),
            "iqr": numeric_df.quantile(0.75) - numeric_df.quantile(0.25),
        }
    )

    mean_nonzero = summary["mean"].replace(0, np.nan).abs()
    summary["cv"] = (summary["std"] / mean_nonzero).replace([np.inf, -np.inf], np.nan)
    return summary


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return Pearson correlation matrix for numeric features."""
    numeric_df = df.select_dtypes(include=["number"])
    return numeric_df.corr(numeric_only=True)


def correlation_analysis(
    df: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Return correlation matrix for selected features or all numeric features."""
    if features is None:
        subset = df.select_dtypes(include=["number"])
    else:
        valid = [col for col in features if col in df.columns]
        subset = df[valid].select_dtypes(include=["number"])

    if subset.empty:
        return pd.DataFrame()
    return subset.corr(numeric_only=True)


def ride_time_distribution(df: pd.DataFrame, datetime_col: str = "datetime") -> dict[str, pd.Series]:
    """Distribution of rides by hour, day-of-week, and month."""
    if datetime_col not in df.columns:
        raise ValueError(f"Column '{datetime_col}' not found in DataFrame.")

    temp = df.copy()
    temp[datetime_col] = pd.to_datetime(temp[datetime_col], errors="coerce")
    temp = temp.dropna(subset=[datetime_col])

    by_hour = temp[datetime_col].dt.hour.value_counts().sort_index()

    day_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    by_day_idx = temp[datetime_col].dt.dayofweek.value_counts().sort_index()
    by_day = by_day_idx.rename(index=day_map)

    by_month = temp[datetime_col].dt.month.value_counts().sort_index()

    return {
        "by_hour": by_hour,
        "by_day": by_day,
        "by_month": by_month,
    }


def identify_peak_periods(df: pd.DataFrame, datetime_col: str = "datetime") -> dict:
    """Identify peak hour and day for Uber rides."""
    dist = ride_time_distribution(df, datetime_col=datetime_col)
    by_hour = dist["by_hour"]
    by_day = dist["by_day"]

    peak_hour = int(by_hour.idxmax()) if not by_hour.empty else None
    peak_hour_rides = int(by_hour.max()) if not by_hour.empty else 0

    peak_day = str(by_day.idxmax()) if not by_day.empty else None
    peak_day_rides = int(by_day.max()) if not by_day.empty else 0

    return {
        "peak_hour": peak_hour,
        "peak_hour_rides": peak_hour_rides,
        "peak_day": peak_day,
        "peak_day_rides": peak_day_rides,
    }


def fare_patterns(
    df: pd.DataFrame,
    fare_col: str = "fare_amount",
    datetime_col: str = "datetime",
    location_col: str = "base",
) -> dict[str, pd.DataFrame]:
    """Analyze fare patterns across times and locations."""
    if fare_col not in df.columns:
        raise ValueError(f"Column '{fare_col}' not found in DataFrame.")

    temp = df.copy()

    if datetime_col in temp.columns:
        temp[datetime_col] = pd.to_datetime(temp[datetime_col], errors="coerce")
        temp = temp.dropna(subset=[datetime_col])
        temp["hour"] = temp[datetime_col].dt.hour
        temp["day_of_week"] = temp[datetime_col].dt.day_name()
        temp["month"] = temp[datetime_col].dt.month

    output: dict[str, pd.DataFrame] = {}

    if "hour" in temp.columns:
        output["fare_by_hour"] = (
            temp.groupby("hour")[fare_col]
            .agg(["count", "mean", "median", "std"])
            .rename(columns={"count": "rides", "mean": "avg_fare", "std": "fare_std"})
            .reset_index()
        )

    if "day_of_week" in temp.columns:
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_stats = (
            temp.groupby("day_of_week")[fare_col]
            .agg(["count", "mean", "median", "std"])
            .rename(columns={"count": "rides", "mean": "avg_fare", "std": "fare_std"})
        )
        output["fare_by_day"] = day_stats.reindex(day_order).dropna(how="all").reset_index()

    if location_col in temp.columns:
        output["fare_by_location"] = (
            temp.groupby(location_col)[fare_col]
            .agg(["count", "mean", "median", "std"])
            .rename(columns={"count": "rides", "mean": "avg_fare", "std": "fare_std"})
            .sort_values("rides", ascending=False)
            .reset_index()
        )

    if all(col in temp.columns for col in ["lat", "lon"]):
        # coarse location grid to analyze spatial fare behavior
        temp["lat_bin"] = temp["lat"].round(2)
        temp["lon_bin"] = temp["lon"].round(2)
        output["fare_by_geo_grid"] = (
            temp.groupby(["lat_bin", "lon_bin"])[fare_col]
            .agg(["count", "mean"])
            .rename(columns={"count": "rides", "mean": "avg_fare"})
            .sort_values("rides", ascending=False)
            .reset_index()
        )

    return output


def central_tendency_variability(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate central tendency and variability measures for numeric columns."""
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return pd.DataFrame()

    # pandas removed DataFrame.mad(); compute mean absolute deviation manually.
    mad_vals = (numeric_df.sub(numeric_df.mean())).abs().mean()

    result = pd.DataFrame(
        {
            "mean": numeric_df.mean(),
            "median": numeric_df.median(),
            "mode": numeric_df.mode(dropna=True).iloc[0] if not numeric_df.mode(dropna=True).empty else np.nan,
            "range": numeric_df.max() - numeric_df.min(),
            "variance": numeric_df.var(),
            "std": numeric_df.std(),
            "iqr": numeric_df.quantile(0.75) - numeric_df.quantile(0.25),
            "mad": mad_vals,
        }
    )
    return result


def categorical_frequency_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create frequency tables for all categorical/object columns."""
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    tables: dict[str, pd.DataFrame] = {}

    for col in categorical_cols:
        counts = df[col].value_counts(dropna=False)
        table = pd.DataFrame(
            {
                col: counts.index.astype(str),
                "frequency": counts.values,
                "percentage": (counts.values / len(df) * 100).round(2),
            }
        )
        tables[col] = table

    return tables


def fare_passenger_anova(
    df: pd.DataFrame,
    fare_col: str = "fare_amount",
    passenger_col: str = "passenger_count",
) -> dict:
    """Run one-way ANOVA: fare differences across passenger_count groups."""
    if fare_col not in df.columns or passenger_col not in df.columns:
        raise ValueError(f"Columns '{fare_col}' and/or '{passenger_col}' not found in DataFrame.")

    grouped = [
        group[fare_col].dropna().values
        for _, group in df.groupby(passenger_col)
        if len(group[fare_col].dropna()) > 1
    ]

    if len(grouped) < 2:
        return {"f_statistic": None, "p_value": None, "message": "Not enough valid groups for ANOVA."}

    f_stat, p_value = stats.f_oneway(*grouped)
    return {"f_statistic": float(f_stat), "p_value": float(p_value)}


def ttest_weekday_vs_weekend_fares(
    df: pd.DataFrame,
    fare_col: str = "fare_amount",
    datetime_col: str = "datetime",
    equal_var: bool = False,
) -> dict:
    """Welch/Student t-test for fare difference between weekdays and weekends."""
    if fare_col not in df.columns or datetime_col not in df.columns:
        raise ValueError(f"Columns '{fare_col}' and/or '{datetime_col}' not found in DataFrame.")

    temp = df[[fare_col, datetime_col]].copy()
    temp[datetime_col] = pd.to_datetime(temp[datetime_col], errors="coerce")
    temp = temp.dropna(subset=[fare_col, datetime_col])
    temp["is_weekend"] = temp[datetime_col].dt.dayofweek >= 5

    weekday_fares = temp.loc[~temp["is_weekend"], fare_col]
    weekend_fares = temp.loc[temp["is_weekend"], fare_col]

    if len(weekday_fares) < 2 or len(weekend_fares) < 2:
        return {
            "t_statistic": None,
            "p_value": None,
            "weekday_mean": float(weekday_fares.mean()) if len(weekday_fares) else None,
            "weekend_mean": float(weekend_fares.mean()) if len(weekend_fares) else None,
            "message": "Not enough observations in weekday/weekend groups.",
        }

    t_stat, p_value = stats.ttest_ind(weekday_fares, weekend_fares, equal_var=equal_var, nan_policy="omit")
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "weekday_mean": float(weekday_fares.mean()),
        "weekend_mean": float(weekend_fares.mean()),
        "weekday_n": int(len(weekday_fares)),
        "weekend_n": int(len(weekend_fares)),
    }


def anova_fare_by_hour(
    df: pd.DataFrame,
    fare_col: str = "fare_amount",
    datetime_col: str = "datetime",
) -> dict:
    """One-way ANOVA: fare varies across hour of day."""
    if fare_col not in df.columns or datetime_col not in df.columns:
        raise ValueError(f"Columns '{fare_col}' and/or '{datetime_col}' not found in DataFrame.")

    temp = df[[fare_col, datetime_col]].copy()
    temp[datetime_col] = pd.to_datetime(temp[datetime_col], errors="coerce")
    temp = temp.dropna(subset=[fare_col, datetime_col])
    temp["hour"] = temp[datetime_col].dt.hour

    groups = [grp[fare_col].values for _, grp in temp.groupby("hour") if len(grp) > 1]
    if len(groups) < 2:
        return {"f_statistic": None, "p_value": None, "message": "Not enough hourly groups for ANOVA."}

    f_stat, p_value = stats.f_oneway(*groups)
    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "num_groups": int(len(groups)),
    }


def chi_square_time_of_day_passenger(
    df: pd.DataFrame,
    time_col: str = "time_of_day",
    passenger_col: str = "passenger_count",
    datetime_col: str = "datetime",
) -> dict:
    """Chi-square test of association between time_of_day and passenger_count."""
    temp = df.copy()
    if time_col not in temp.columns:
        if datetime_col not in temp.columns:
            raise ValueError(f"Need '{time_col}' or '{datetime_col}' in DataFrame.")
        temp[datetime_col] = pd.to_datetime(temp[datetime_col], errors="coerce")
        temp = temp.dropna(subset=[datetime_col])
        temp["hour"] = temp[datetime_col].dt.hour
        bins = [-1, 5, 11, 17, 23]
        labels = ["Night", "Morning", "Afternoon", "Evening"]
        temp[time_col] = pd.cut(temp["hour"], bins=bins, labels=labels).astype(str)

    if passenger_col not in temp.columns:
        raise ValueError(f"Column '{passenger_col}' not found in DataFrame.")

    test_df = temp[[time_col, passenger_col]].dropna()
    contingency = pd.crosstab(test_df[time_col].astype(str), test_df[passenger_col].astype(str))

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {
            "chi2": None,
            "p_value": None,
            "dof": None,
            "message": "Insufficient contingency table shape for chi-square test.",
            "contingency_table": contingency,
        }

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "contingency_table": contingency,
        "expected_frequencies": pd.DataFrame(expected, index=contingency.index, columns=contingency.columns),
    }


def z_test_mean_fare(
    df: pd.DataFrame,
    hypothesized_mean: float,
    fare_col: str = "fare_amount",
) -> dict:
    """One-sample z-test for mean fare against a hypothesized population mean."""
    if fare_col not in df.columns:
        raise ValueError(f"Column '{fare_col}' not found in DataFrame.")

    sample = df[fare_col].dropna().values
    n = len(sample)
    if n < 2:
        return {"z_statistic": None, "p_value": None, "message": "Not enough observations for z-test."}

    sample_mean = float(np.mean(sample))
    sample_std = float(np.std(sample, ddof=1))
    if sample_std == 0:
        return {"z_statistic": None, "p_value": None, "message": "Zero variance in sample."}

    z_stat = (sample_mean - hypothesized_mean) / (sample_std / np.sqrt(n))
    p_value_two_tailed = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return {
        "z_statistic": float(z_stat),
        "p_value": float(p_value_two_tailed),
        "sample_mean": sample_mean,
        "hypothesized_mean": float(hypothesized_mean),
        "n": int(n),
    }


def correlation_coefficients(df: pd.DataFrame, features: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Compute Pearson, Spearman, and Kendall correlation matrices for numeric variables."""
    if features is None:
        data = df.select_dtypes(include=["number"])
    else:
        valid = [col for col in features if col in df.columns]
        data = df[valid].select_dtypes(include=["number"])

    if data.empty:
        return {"pearson": pd.DataFrame(), "spearman": pd.DataFrame(), "kendall": pd.DataFrame()}

    return {
        "pearson": data.corr(method="pearson", numeric_only=True),
        "spearman": data.corr(method="spearman", numeric_only=True),
        "kendall": data.corr(method="kendall", numeric_only=True),
    }


def linear_regression_fare_determinants(
    df: pd.DataFrame,
    target_col: str = "fare_amount",
    feature_cols: list[str] | None = None,
) -> dict:
    """Fit linear regression and return coefficient-based fare determinant summary."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    if feature_cols is None:
        feature_cols = [
            col for col in ["distance", "passenger_count", "hour", "day", "month", "year", "day_of_week", "lat", "lon"]
            if col in df.columns
        ]

    feature_cols = [col for col in feature_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if not feature_cols:
        return {"message": "No valid numeric feature columns for linear regression.", "coefficients": pd.DataFrame()}

    model_df = df[feature_cols + [target_col]].dropna()
    if len(model_df) < 2:
        return {"message": "Insufficient rows for regression.", "coefficients": pd.DataFrame()}

    X = model_df[feature_cols]
    y = model_df[target_col]

    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)

    coef_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient": model.coef_,
            "abs_coefficient": np.abs(model.coef_),
        }
    ).sort_values("abs_coefficient", ascending=False)

    return {
        "intercept": float(model.intercept_),
        "r2": float(r2),
        "n_samples": int(len(model_df)),
        "coefficients": coef_df,
    }


def pca_eigen_decomposition(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    standardize: bool = True,
) -> dict:
    """Calculate covariance-matrix eigenvalues/eigenvectors for PCA preparation."""
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=["number"]).columns.tolist()
    else:
        feature_cols = [col for col in feature_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    if len(feature_cols) < 2:
        return {
            "message": "Need at least two numeric features for PCA preparation.",
            "eigenvalues": np.array([]),
            "eigenvectors": pd.DataFrame(),
            "explained_variance_ratio": np.array([]),
        }

    X = df[feature_cols].dropna().to_numpy(dtype=float)
    if X.shape[0] < 2:
        return {
            "message": "Not enough observations for covariance estimation.",
            "eigenvalues": np.array([]),
            "eigenvectors": pd.DataFrame(),
            "explained_variance_ratio": np.array([]),
        }

    if standardize:
        means = X.mean(axis=0)
        stds = X.std(axis=0, ddof=1)
        stds[stds == 0] = 1.0
        X = (X - means) / stds

    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # sort descending by eigenvalue
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    total = eigvals.sum()
    explained = eigvals / total if total != 0 else np.zeros_like(eigvals)

    eigvec_df = pd.DataFrame(eigvecs, index=feature_cols, columns=[f"PC{i+1}" for i in range(len(feature_cols))])

    return {
        "eigenvalues": eigvals,
        "eigenvectors": eigvec_df,
        "explained_variance_ratio": explained,
        "features": feature_cols,
    }
