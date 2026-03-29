"""Visualization utilities for Uber ride analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")


def _prepare_temporal_features(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    """Return a copy with parsed datetime and common temporal features."""
    if datetime_col not in df.columns:
        raise ValueError(f"Column '{datetime_col}' not found in DataFrame.")

    temp = df.copy()
    temp[datetime_col] = pd.to_datetime(temp[datetime_col], errors="coerce")
    temp = temp.dropna(subset=[datetime_col])
    temp["date"] = temp[datetime_col].dt.date
    temp["hour"] = temp[datetime_col].dt.hour
    temp["month_period"] = temp[datetime_col].dt.to_period("M").astype(str)
    temp["day_name"] = temp[datetime_col].dt.day_name()
    temp["day_of_week"] = temp[datetime_col].dt.dayofweek

    if "time_of_day" not in temp.columns:
        bins = [-1, 5, 11, 17, 23]
        labels = ["Night", "Morning", "Afternoon", "Evening"]
        temp["time_of_day"] = pd.cut(temp["hour"], bins=bins, labels=labels)

    return temp


# ==========================
# Matplotlib visualizations
# ==========================
def matplotlib_line_rides_over_time(df: pd.DataFrame, datetime_col: str = "datetime") -> None:
    """Line plots for rides over daily and monthly timelines."""
    temp = _prepare_temporal_features(df, datetime_col=datetime_col)

    daily = temp.groupby("date").size()
    monthly = temp.groupby("month_period").size()

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
    axes[0].plot(daily.index, daily.values, color="tab:blue", linewidth=1.6)
    axes[0].set_title("Daily Ride Trend")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Rides")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].plot(monthly.index, monthly.values, color="tab:orange", marker="o")
    axes[1].set_title("Monthly Ride Trend")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Rides")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.show()


def matplotlib_bar_rides_by_hour_and_day(df: pd.DataFrame, datetime_col: str = "datetime") -> None:
    """Bar plots for rides by hour of day and day of week."""
    temp = _prepare_temporal_features(df, datetime_col=datetime_col)

    hour_counts = temp["hour"].value_counts().sort_index()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_counts = temp["day_name"].value_counts().reindex(day_order)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(hour_counts.index, hour_counts.values, color="steelblue")
    axes[0].set_title("Rides by Hour of Day")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Rides")

    axes[1].bar(day_counts.index, day_counts.values, color="seagreen")
    axes[1].set_title("Rides by Day of Week")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Rides")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.show()


def matplotlib_hist_fare_distribution(df: pd.DataFrame, fare_col: str = "fare_amount", bins: int = 40) -> None:
    """Histogram for fare amount distribution."""
    if fare_col not in df.columns:
        raise ValueError(f"Column '{fare_col}' not found in DataFrame.")

    plt.figure(figsize=(10, 5))
    plt.hist(df[fare_col].dropna(), bins=bins, color="mediumpurple", edgecolor="black", alpha=0.8)
    plt.title("Distribution of Fare Amount")
    plt.xlabel("Fare Amount")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def matplotlib_scatter_fare_vs_distance(
    df: pd.DataFrame,
    fare_col: str = "fare_amount",
    distance_col: str = "distance",
) -> None:
    """Scatter plot for fare vs distance relationship."""
    if fare_col not in df.columns or distance_col not in df.columns:
        raise ValueError(f"Columns '{fare_col}' and/or '{distance_col}' not found in DataFrame.")

    plt.figure(figsize=(10, 6))
    plt.scatter(df[distance_col], df[fare_col], alpha=0.35, s=12, c="teal")
    plt.title("Fare vs Distance")
    plt.xlabel("Distance")
    plt.ylabel("Fare Amount")
    plt.tight_layout()
    plt.show()


def matplotlib_pie_rides_by_base(df: pd.DataFrame, base_col: str = "base", top_n: int = 10) -> None:
    """Pie chart for ride distribution by base/company."""
    if base_col not in df.columns:
        raise ValueError(f"Column '{base_col}' not found in DataFrame.")

    counts = df[base_col].astype(str).value_counts()
    if top_n > 0 and len(counts) > top_n:
        top = counts.head(top_n)
        other = counts.iloc[top_n:].sum()
        counts = pd.concat([top, pd.Series({"Other": other})])

    plt.figure(figsize=(8, 8))
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140)
    plt.title("Ride Distribution by Base/Company")
    plt.tight_layout()
    plt.show()


def matplotlib_3d_lat_lon_fare(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    fare_col: str = "fare_amount",
    sample_size: int = 5000,
) -> None:
    """3D scatter: latitude, longitude, and fare amount."""
    required = [lat_col, lon_col, fare_col]
    if any(col not in df.columns for col in required):
        raise ValueError(f"Required columns not found: {required}")

    temp = df[required].dropna().copy()
    if sample_size and len(temp) > sample_size:
        temp = temp.sample(sample_size, random_state=42)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        temp[lon_col],
        temp[lat_col],
        temp[fare_col],
        c=temp[fare_col],
        cmap="viridis",
        s=10,
        alpha=0.7,
    )
    ax.set_title("3D View: Longitude, Latitude, Fare")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Fare Amount")
    fig.colorbar(scatter, ax=ax, pad=0.12, label="Fare Amount")
    plt.tight_layout()
    plt.show()


def matplotlib_heatmap_rides_hour_day(df: pd.DataFrame, datetime_col: str = "datetime") -> None:
    """Heatmap (Matplotlib) for rides by hour and day of week."""
    temp = _prepare_temporal_features(df, datetime_col=datetime_col)

    pivot = pd.crosstab(temp["day_of_week"], temp["hour"]).reindex(index=range(7), fill_value=0)
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    plt.figure(figsize=(12, 4.8))
    plt.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    plt.colorbar(label="Rides")
    plt.title("Heatmap: Rides by Hour and Day of Week")
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.yticks(ticks=np.arange(7), labels=day_labels)
    plt.xticks(ticks=np.arange(0, 24, 1))
    plt.tight_layout()
    plt.show()


# =======================
# Seaborn visualizations
# =======================
def seaborn_box_fare_by_time_of_day(df: pd.DataFrame, fare_col: str = "fare_amount") -> None:
    """Box plot for fare distribution by time_of_day."""
    if fare_col not in df.columns:
        raise ValueError(f"Column '{fare_col}' not found in DataFrame.")

    temp = df.copy()
    if "time_of_day" not in temp.columns:
        temp = _prepare_temporal_features(temp, datetime_col="datetime")

    order = ["Night", "Morning", "Afternoon", "Evening"]
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=temp, x="time_of_day", y=fare_col, order=order)
    plt.title("Fare Distribution by Time of Day")
    plt.xlabel("Time of Day")
    plt.ylabel("Fare Amount")
    plt.tight_layout()
    plt.show()


def seaborn_violin_distance_by_passenger(
    df: pd.DataFrame,
    distance_col: str = "distance",
    passenger_col: str = "passenger_count",
) -> None:
    """Violin plot for distance distribution by passenger count."""
    if distance_col not in df.columns or passenger_col not in df.columns:
        raise ValueError(f"Columns '{distance_col}' and/or '{passenger_col}' not found in DataFrame.")

    plt.figure(figsize=(10, 5))
    sns.violinplot(data=df, x=passenger_col, y=distance_col, inner="quartile", cut=0)
    plt.title("Distance Distribution by Passenger Count")
    plt.xlabel("Passenger Count")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()


def seaborn_swarm_fare_by_day_of_week(
    df: pd.DataFrame,
    fare_col: str = "fare_amount",
    datetime_col: str = "datetime",
    sample_size: int = 4000,
) -> None:
    """Swarm plot for fare by day of week."""
    if fare_col not in df.columns:
        raise ValueError(f"Column '{fare_col}' not found in DataFrame.")

    temp = _prepare_temporal_features(df, datetime_col=datetime_col)
    plot_df = temp[["day_name", fare_col]].dropna()
    if sample_size and len(plot_df) > sample_size:
        plot_df = plot_df.sample(sample_size, random_state=42)

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    plt.figure(figsize=(12, 5))
    sns.swarmplot(data=plot_df, x="day_name", y=fare_col, order=order, size=3, alpha=0.7)
    plt.title("Fare by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Fare Amount")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def seaborn_correlation_heatmap(df: pd.DataFrame, cols: list[str] | None = None) -> None:
    """Seaborn heatmap for correlation matrix."""
    if cols:
        data = df[[c for c in cols if c in df.columns]].select_dtypes(include=["number"])
    else:
        data = df.select_dtypes(include=["number"])

    if data.empty:
        raise ValueError("No numeric columns available for correlation heatmap.")

    corr = data.corr(numeric_only=True)
    plt.figure(figsize=(9, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


def seaborn_pairplot_numeric(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    hue: str | None = None,
    sample_size: int = 3000,
) -> None:
    """Pair plot for selected/all numerical features."""
    temp = df.copy()
    if sample_size and len(temp) > sample_size:
        temp = temp.sample(sample_size, random_state=42)

    if cols:
        numeric_cols = [c for c in cols if c in temp.columns and pd.api.types.is_numeric_dtype(temp[c])]
    else:
        numeric_cols = temp.select_dtypes(include=["number"]).columns.tolist()

    if len(numeric_cols) < 2:
        raise ValueError("Need at least two numerical columns for pair plot.")

    pairplot_kwargs = {"data": temp[numeric_cols + ([hue] if hue and hue in temp.columns else [])], "vars": numeric_cols}
    if hue and hue in temp.columns:
        pairplot_kwargs["hue"] = hue

    sns.pairplot(**pairplot_kwargs)
    plt.show()


def seaborn_countplot_categorical(df: pd.DataFrame, columns: list[str] | None = None, top_n: int = 15) -> None:
    """Count plots for categorical variables."""
    if columns is None:
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

    columns = [col for col in columns if col in df.columns]
    if not columns:
        raise ValueError("No categorical columns found for count plot.")

    for col in columns:
        plt.figure(figsize=(12, 5))
        order = df[col].astype(str).value_counts().head(top_n).index
        sns.countplot(data=df, x=col, order=order)
        plt.title(f"Rides by {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# =====================================
# Backward-compatible function wrappers
# =====================================
def plot_fare_distribution(df: pd.DataFrame, fare_col: str = "fare_amount") -> None:
    matplotlib_hist_fare_distribution(df, fare_col=fare_col)


def plot_trips_by_hour(df: pd.DataFrame, datetime_col: str = "datetime") -> None:
    temp = _prepare_temporal_features(df, datetime_col=datetime_col)
    trips_per_hour = temp["hour"].value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    sns.barplot(x=trips_per_hour.index, y=trips_per_hour.values)
    plt.title("Trips by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Trips")
    plt.tight_layout()
    plt.show()


def plot_trips_by_day(df: pd.DataFrame, datetime_col: str = "datetime") -> None:
    temp = _prepare_temporal_features(df, datetime_col=datetime_col)
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    trips_per_day = temp["day_name"].value_counts().reindex(day_order)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=trips_per_day.index, y=trips_per_day.values)
    plt.title("Trips by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Number of Trips")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_trips_by_month(df: pd.DataFrame, datetime_col: str = "datetime") -> None:
    temp = _prepare_temporal_features(df, datetime_col=datetime_col)
    temp["month"] = pd.to_datetime(temp[datetime_col]).dt.month
    trips_per_month = temp["month"].value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    sns.barplot(x=trips_per_month.index, y=trips_per_month.values)
    plt.title("Trips by Month")
    plt.xlabel("Month")
    plt.ylabel("Number of Trips")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, cols: list[str] | None = None) -> None:
    seaborn_correlation_heatmap(df, cols=cols)


def plot_fare_by_time_of_day(df: pd.DataFrame, fare_col: str = "fare_amount") -> None:
    seaborn_box_fare_by_time_of_day(df, fare_col=fare_col)


def plot_fare_by_location(df: pd.DataFrame, fare_col: str = "fare_amount", location_col: str = "base") -> None:
    if fare_col not in df.columns or location_col not in df.columns:
        raise ValueError(f"Columns '{fare_col}' and/or '{location_col}' not found in DataFrame.")

    fare_by_location = df.groupby(location_col)[fare_col].mean().sort_values(ascending=False).head(15)

    plt.figure(figsize=(12, 5))
    sns.barplot(x=fare_by_location.index, y=fare_by_location.values)
    plt.title("Average Fare by Location/Base (Top 15)")
    plt.xlabel("Location/Base")
    plt.ylabel("Average Fare")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_pickup_locations(df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon") -> None:
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Columns '{lat_col}' and/or '{lon_col}' not found in DataFrame.")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=lon_col, y=lat_col, alpha=0.3, s=12)
    plt.title("Pickup Locations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.histplot(df[fare_col].dropna(), bins=40, kde=True)
    plt.title("Fare Amount Distribution")
    plt.xlabel("Fare Amount")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_trips_by_hour(df: pd.DataFrame, datetime_col: str = "datetime") -> None:
    if datetime_col not in df.columns:
        raise ValueError(f"Column '{datetime_col}' not found in DataFrame.")

    temp = df.copy()
    temp[datetime_col] = pd.to_datetime(temp[datetime_col], errors="coerce")
    trips_per_hour = temp[datetime_col].dt.hour.value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    sns.barplot(x=trips_per_hour.index, y=trips_per_hour.values)
    plt.title("Trips by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Trips")
    plt.tight_layout()
    plt.show()


def plot_trips_by_day(df: pd.DataFrame, datetime_col: str = "datetime") -> None:
    if datetime_col not in df.columns:
        raise ValueError(f"Column '{datetime_col}' not found in DataFrame.")

    temp = df.copy()
    temp[datetime_col] = pd.to_datetime(temp[datetime_col], errors="coerce")
    days = temp[datetime_col].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    trips_per_day = days.value_counts().reindex(order)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=trips_per_day.index, y=trips_per_day.values)
    plt.title("Trips by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Number of Trips")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_trips_by_month(df: pd.DataFrame, datetime_col: str = "datetime") -> None:
    if datetime_col not in df.columns:
        raise ValueError(f"Column '{datetime_col}' not found in DataFrame.")

    temp = df.copy()
    temp[datetime_col] = pd.to_datetime(temp[datetime_col], errors="coerce")
    trips_per_month = temp[datetime_col].dt.month.value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    sns.barplot(x=trips_per_month.index, y=trips_per_month.values)
    plt.title("Trips by Month")
    plt.xlabel("Month")
    plt.ylabel("Number of Trips")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, cols: list[str] | None = None) -> None:
    if cols:
        data = df[[c for c in cols if c in df.columns]].select_dtypes(include=["number"])
    else:
        data = df.select_dtypes(include=["number"])

    if data.empty:
        raise ValueError("No numeric columns available for correlation heatmap.")

    corr = data.corr(numeric_only=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def plot_fare_by_time_of_day(df: pd.DataFrame, fare_col: str = "fare_amount") -> None:
    if fare_col not in df.columns:
        raise ValueError(f"Column '{fare_col}' not found in DataFrame.")

    temp = df.copy()
    if "time_of_day" not in temp.columns:
        if "datetime" not in temp.columns:
            raise ValueError("Need either 'time_of_day' or 'datetime' column.")
        temp["datetime"] = pd.to_datetime(temp["datetime"], errors="coerce")
        temp["hour"] = temp["datetime"].dt.hour
        bins = [-1, 5, 11, 17, 23]
        labels = ["Night", "Morning", "Afternoon", "Evening"]
        temp["time_of_day"] = pd.cut(temp["hour"], bins=bins, labels=labels)

    order = ["Night", "Morning", "Afternoon", "Evening"]
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=temp, x="time_of_day", y=fare_col, order=order)
    plt.title("Fare Distribution by Time of Day")
    plt.xlabel("Time of Day")
    plt.ylabel("Fare Amount")
    plt.tight_layout()
    plt.show()


def plot_fare_by_location(df: pd.DataFrame, fare_col: str = "fare_amount", location_col: str = "base") -> None:
    if fare_col not in df.columns or location_col not in df.columns:
        raise ValueError(f"Columns '{fare_col}' and/or '{location_col}' not found in DataFrame.")

    fare_by_location = (
        df.groupby(location_col)[fare_col]
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )

    plt.figure(figsize=(12, 5))
    sns.barplot(x=fare_by_location.index, y=fare_by_location.values)
    plt.title("Average Fare by Location/Base (Top 15)")
    plt.xlabel("Location/Base")
    plt.ylabel("Average Fare")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_pickup_locations(df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon") -> None:
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Columns '{lat_col}' and/or '{lon_col}' not found in DataFrame.")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=lon_col, y=lat_col, alpha=0.3, s=12)
    plt.title("Pickup Locations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()
