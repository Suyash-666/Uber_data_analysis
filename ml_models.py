"""Machine learning models for Uber ride prediction and clustering tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    silhouette_score,
)
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor, plot_tree

sns.set_theme(style="whitegrid")


def _resolve_distance_column(df: pd.DataFrame, requested_col: str = "distance") -> str | None:
    """Resolve distance column name from common aliases."""
    if requested_col in df.columns:
        return requested_col

    aliases = [
        "distance",
        "trip_distance",
        "trip_miles",
        "miles",
        "trip_km",
        "km",
        "ride_distance",
    ]

    cols_lower = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias in cols_lower:
            return cols_lower[alias]

    return None


def engineer_features(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    """Create engineered features for ML models."""
    data = df.copy()

    # Normalize distance column name if a known alias exists
    resolved_distance_col = _resolve_distance_column(data, requested_col="distance")
    if resolved_distance_col is not None and resolved_distance_col != "distance":
        data["distance"] = data[resolved_distance_col]

    # Derive distance if unavailable but pickup/dropoff coordinates exist.
    if "distance" not in data.columns:
        coord_sets = [
            ("pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"),
            ("pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon"),
            ("start_lat", "start_lon", "end_lat", "end_lon"),
            ("lat", "lon", "dropoff_lat", "dropoff_lon"),
        ]
        selected = next((cols for cols in coord_sets if all(c in data.columns for c in cols)), None)

        if selected is not None:
            lat1_col, lon1_col, lat2_col, lon2_col = selected

            lat1 = pd.to_numeric(data[lat1_col], errors="coerce")
            lon1 = pd.to_numeric(data[lon1_col], errors="coerce")
            lat2 = pd.to_numeric(data[lat2_col], errors="coerce")
            lon2 = pd.to_numeric(data[lon2_col], errors="coerce")

            # Haversine distance (km)
            r = 6371.0
            phi1 = np.radians(lat1)
            phi2 = np.radians(lat2)
            dphi = np.radians(lat2 - lat1)
            dlambda = np.radians(lon2 - lon1)

            a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            data["distance"] = r * c

    if datetime_col in data.columns:
        data[datetime_col] = pd.to_datetime(data[datetime_col], errors="coerce")
        data["hour"] = data[datetime_col].dt.hour
        data["day"] = data[datetime_col].dt.day
        data["month"] = data[datetime_col].dt.month
        data["year"] = data[datetime_col].dt.year
        data["day_of_week"] = data[datetime_col].dt.dayofweek
        data["is_weekend"] = (data["day_of_week"] >= 5).astype(float)

    if {"distance", "passenger_count"}.issubset(data.columns):
        data["distance_per_passenger"] = data["distance"] / data["passenger_count"].replace(0, np.nan)

    if {"fare_amount", "distance"}.issubset(data.columns):
        data["fare_per_distance"] = data["fare_amount"] / data["distance"].replace(0, np.nan)

    return data


def create_ride_category(
    df: pd.DataFrame,
    distance_col: str = "distance",
    short_q: float = 0.33,
    long_q: float = 0.66,
) -> pd.DataFrame:
    """Create categorical ride label: short/medium/long based on distance quantiles."""
    resolved_distance_col = _resolve_distance_column(df, requested_col=distance_col)
    if resolved_distance_col is None:
        raise ValueError(f"Column '{distance_col}' not found in DataFrame.")

    data = df.copy()
    if resolved_distance_col != "distance":
        data["distance"] = data[resolved_distance_col]
    active_distance_col = "distance"

    valid_distance = data[active_distance_col].dropna()
    if valid_distance.empty:
        raise ValueError("Distance column has no valid values.")

    q1 = valid_distance.quantile(short_q)
    q2 = valid_distance.quantile(long_q)

    data["ride_category"] = pd.cut(
        data[active_distance_col],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["short", "medium", "long"],
    )
    return data


def _default_feature_candidates(df: pd.DataFrame) -> list[str]:
    preferred = [
        "distance",
        "passenger_count",
        "lat",
        "lon",
        "hour",
        "day",
        "month",
        "year",
        "day_of_week",
        "is_weekend",
        "distance_per_passenger",
    ]
    return [col for col in preferred if col in df.columns]


def _build_numeric_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray, labels: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": pd.DataFrame(
            confusion_matrix(y_true, y_pred, labels=labels),
            index=[f"true_{c}" for c in labels],
            columns=[f"pred_{c}" for c in labels],
        ),
    }


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str = "fare_amount",
    model_name: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Backward-compatible single-model regression training helper."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    data = engineer_features(df).dropna(subset=[target_col]).copy()
    feature_cols = _default_feature_candidates(data)
    if not feature_cols:
        raise ValueError("No valid numeric feature columns available for training.")

    X = data[feature_cols]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if model_name == "linear_regression":
        model = _build_numeric_pipeline(LinearRegression())
    elif model_name == "svr":
        model = _build_numeric_pipeline(SVR(kernel="rbf", C=10.0, epsilon=0.1))
    else:
        rf = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", rf),
            ]
        )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = _regression_metrics(y_test, preds)

    return {
        "model": model_name,
        "features": feature_cols,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
    }


def train_regression_models(
    df: pd.DataFrame,
    target_col: str = "fare_amount",
    test_size: float = 0.2,
    random_state: int = 42,
    feature_cols: list[str] | None = None,
    svr_max_train_rows: int | None = 20000,
) -> dict:
    """Train Linear Regression and SVR for fare prediction and compare performance.

    Notes:
    - RBF `SVR` has high computational complexity on large datasets.
    - `svr_max_train_rows` subsamples training rows for SVR only to keep runtime practical.
      Set to `None` to train SVR on all training rows.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    data = engineer_features(df).dropna(subset=[target_col]).copy()
    if feature_cols is None:
        feature_cols = _default_feature_candidates(data)
    feature_cols = [col for col in feature_cols if col in data.columns]

    if not feature_cols:
        raise ValueError("No valid feature columns available for regression.")

    model_df = data[feature_cols + [target_col]].dropna()
    X = model_df[feature_cols]
    y = model_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models = {
        "Linear Regression": _build_numeric_pipeline(LinearRegression()),
        "SVR": _build_numeric_pipeline(SVR(kernel="rbf", C=10.0, epsilon=0.1)),
    }

    results = []
    trained_models = {}
    training_info: dict[str, dict[str, int | bool]] = {}

    for name, model in models.items():
        fit_X = X_train
        fit_y = y_train
        was_subsampled = False

        if name == "SVR" and svr_max_train_rows is not None and len(X_train) > svr_max_train_rows:
            sampled_idx = X_train.sample(n=svr_max_train_rows, random_state=random_state).index
            fit_X = X_train.loc[sampled_idx]
            fit_y = y_train.loc[sampled_idx]
            was_subsampled = True

        model.fit(fit_X, fit_y)
        preds = model.predict(X_test)
        metrics = _regression_metrics(y_test, preds)
        results.append({"model": name, **metrics})
        trained_models[name] = model
        training_info[name] = {
            "fit_rows": int(len(fit_X)),
            "full_train_rows": int(len(X_train)),
            "was_subsampled": bool(was_subsampled),
        }

    comparison = pd.DataFrame(results).sort_values("rmse", ascending=True).reset_index(drop=True)
    return {
        "features": feature_cols,
        "comparison": comparison,
        "models": trained_models,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "training_info": training_info,
    }


def train_classification_models(
    df: pd.DataFrame,
    distance_col: str = "distance",
    test_size: float = 0.2,
    random_state: int = 42,
    feature_cols: list[str] | None = None,
    svm_max_train_rows: int | None = 25000,
    rf_n_estimators: int = 200,
) -> dict:
    """Train Logistic Regression, SVM, and Random Forest for ride category classification.

    Notes:
    - RBF SVM can be expensive on large training sets.
    - `svm_max_train_rows` subsamples SVM training rows only when needed.
    - `rf_n_estimators` controls random-forest training cost.
    """
    data = engineer_features(df)

    # Resolve/normalize distance column before target construction.
    resolved_distance_col = _resolve_distance_column(data, requested_col=distance_col)
    if resolved_distance_col is None:
        raise ValueError(
            f"Column '{distance_col}' not found in DataFrame. "
            f"Available columns: {', '.join(data.columns.astype(str).tolist())}"
        )

    if resolved_distance_col != "distance":
        data["distance"] = data[resolved_distance_col]

    data = create_ride_category(data, distance_col="distance")
    distance_col = "distance"

    target_col = "ride_category"
    data = data.dropna(subset=[target_col]).copy()

    if feature_cols is None:
        feature_cols = _default_feature_candidates(data)
    feature_cols = [col for col in feature_cols if col in data.columns and col != distance_col]

    if not feature_cols:
        raise ValueError("No valid feature columns available for classification.")

    model_df = data[feature_cols + [target_col]].dropna()
    X = model_df[feature_cols]
    y = model_df[target_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    models = {
        "Logistic Regression": _build_numeric_pipeline(LogisticRegression(max_iter=2000, random_state=random_state)),
        "SVM": _build_numeric_pipeline(SVC(kernel="rbf", C=2.0, probability=False, random_state=random_state)),
        "Random Forest Classifier": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=rf_n_estimators, random_state=random_state, n_jobs=-1)),
            ]
        ),
    }

    class_labels = sorted(y.unique().tolist())
    results = []
    confusion_matrices: dict[str, pd.DataFrame] = {}
    trained_models = {}
    training_info: dict[str, dict[str, int | bool]] = {}

    for name, model in models.items():
        fit_X = X_train
        fit_y = y_train
        was_subsampled = False

        if name == "SVM" and svm_max_train_rows is not None and len(X_train) > svm_max_train_rows:
            sampled_idx = X_train.sample(n=svm_max_train_rows, random_state=random_state).index
            fit_X = X_train.loc[sampled_idx]
            fit_y = y_train.loc[sampled_idx]
            was_subsampled = True

        model.fit(fit_X, fit_y)
        preds = model.predict(X_test)
        metrics = _classification_metrics(y_test, preds, labels=class_labels)

        results.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "precision_weighted": metrics["precision_weighted"],
                "recall_weighted": metrics["recall_weighted"],
                "f1_weighted": metrics["f1_weighted"],
            }
        )
        confusion_matrices[name] = metrics["confusion_matrix"]
        trained_models[name] = model
        training_info[name] = {
            "fit_rows": int(len(fit_X)),
            "full_train_rows": int(len(X_train)),
            "was_subsampled": bool(was_subsampled),
        }

    comparison = pd.DataFrame(results).sort_values("f1_weighted", ascending=False).reset_index(drop=True)

    # feature importance from random forest
    rf_model = models["Random Forest Classifier"].named_steps["model"]
    feature_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": rf_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return {
        "features": feature_cols,
        "comparison": comparison,
        "confusion_matrices": confusion_matrices,
        "feature_importance": feature_importance,
        "models": trained_models,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "training_info": training_info,
    }


def kmeans_clustering_analysis(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    k_range: tuple[int, int] = (2, 10),
    random_state: int = 42,
) -> dict:
    """Run K-Means with elbow and silhouette selection, then assign clusters."""
    data = engineer_features(df)

    if feature_cols is None:
        feature_cols = [col for col in ["lat", "lon", "fare_amount", "distance"] if col in data.columns]
    feature_cols = [col for col in feature_cols if col in data.columns]

    if len(feature_cols) < 2:
        raise ValueError("Need at least two clustering features.")

    cluster_df = data[feature_cols].dropna().copy()
    if cluster_df.empty:
        raise ValueError("No rows available for clustering after dropping missing values.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df)

    k_min, k_max = k_range
    k_values = list(range(k_min, k_max + 1))
    inertia_list = []
    silhouette_list = []

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = km.fit_predict(X_scaled)
        inertia_list.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels) if k > 1 else np.nan
        silhouette_list.append(sil)

    eval_df = pd.DataFrame(
        {
            "k": k_values,
            "inertia": inertia_list,
            "silhouette_score": silhouette_list,
        }
    )

    best_row = eval_df.loc[eval_df["silhouette_score"].idxmax()]
    best_k = int(best_row["k"])

    best_model = KMeans(n_clusters=best_k, random_state=random_state, n_init=20)
    best_labels = best_model.fit_predict(X_scaled)

    labeled = cluster_df.copy()
    labeled["cluster"] = best_labels

    return {
        "features": feature_cols,
        "best_k": best_k,
        "evaluation": eval_df,
        "clustered_data": labeled,
        "model": best_model,
    }


def plot_kmeans_elbow_silhouette(evaluation_df: pd.DataFrame) -> None:
    """Plot elbow curve and silhouette curve for K selection."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(evaluation_df["k"], evaluation_df["inertia"], marker="o")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(evaluation_df["k"], evaluation_df["silhouette_score"], marker="o", color="tab:green")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Score")

    plt.tight_layout()
    plt.show()


def plot_kmeans_clusters_map(
    clustered_df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    cluster_col: str = "cluster",
) -> None:
    """Visualize K-Means clusters on pickup coordinates."""
    required = [lat_col, lon_col, cluster_col]
    if any(col not in clustered_df.columns for col in required):
        raise ValueError(f"Required columns not found: {required}")

    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=clustered_df,
        x=lon_col,
        y=lat_col,
        hue=cluster_col,
        palette="tab10",
        s=18,
        alpha=0.7,
    )
    plt.title("K-Means Clusters on Pickup Locations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()


def hierarchical_clustering_analysis(
    df: pd.DataFrame,
    location_cols: list[str] | None = None,
    n_clusters: int = 4,
    linkage_method: str = "ward",
    sample_size: int = 3000,
) -> dict:
    """Perform hierarchical clustering on pickup locations and return linkage + labels."""
    data = engineer_features(df)
    if location_cols is None:
        location_cols = [col for col in ["lat", "lon"] if col in data.columns]

    if len(location_cols) < 2:
        raise ValueError("Need latitude/longitude columns for hierarchical clustering.")

    loc_df = data[location_cols].dropna().copy()
    if loc_df.empty:
        raise ValueError("No location rows available for clustering.")

    if sample_size and len(loc_df) > sample_size:
        loc_df = loc_df.sample(sample_size, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(loc_df)
    Z = linkage(X_scaled, method=linkage_method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    clustered = loc_df.copy()
    clustered["cluster"] = labels

    return {
        "features": location_cols,
        "linkage_matrix": Z,
        "clustered_data": clustered,
        "n_clusters": n_clusters,
    }


def plot_hierarchical_dendrogram(
    linkage_matrix: np.ndarray,
    max_display_levels: int = 30,
) -> None:
    """Create dendrogram for hierarchical clustering."""
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, truncate_mode="lastp", p=max_display_levels, leaf_rotation=45)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Clustered Points")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()


def decision_tree_regressor_analysis(
    df: pd.DataFrame,
    target_col: str = "fare_amount",
    feature_cols: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_depth: int = 4,
) -> dict:
    """Train Decision Tree Regressor, evaluate metrics, and expose feature importance."""
    data = engineer_features(df)
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    if feature_cols is None:
        feature_cols = _default_feature_candidates(data)
    feature_cols = [col for col in feature_cols if col in data.columns]

    if not feature_cols:
        raise ValueError("No valid feature columns for decision tree regression.")

    model_df = data[feature_cols + [target_col]].dropna()
    X = model_df[feature_cols]
    y = model_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)),
        ]
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    metrics = _regression_metrics(y_test, preds)

    dt_model = pipe.named_steps["model"]
    feature_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": dt_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return {
        "model": pipe,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "feature_cols": feature_cols,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
    }


def plot_decision_tree_structure(
    trained_pipeline: Pipeline,
    feature_names: list[str],
    max_depth: int = 3,
) -> None:
    """Visualize trained Decision Tree Regressor structure."""
    dt_model = trained_pipeline.named_steps["model"]

    plt.figure(figsize=(18, 8))
    plot_tree(
        dt_model,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=9,
    )
    plt.title("Decision Tree Regressor Structure")
    plt.tight_layout()
    plt.show()


def association_rule_mining(
    df: pd.DataFrame,
    min_support: float = 0.05,
    min_threshold: float = 1.0,
) -> dict:
    """Run Apriori and association rules on discretized ride characteristics."""
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except ImportError as exc:
        raise ImportError("mlxtend is required for association rule mining. Install it first.") from exc

    data = engineer_features(df)

    basket = pd.DataFrame(index=data.index)

    if "time_of_day" in data.columns:
        basket["time_of_day"] = data["time_of_day"].astype(str)
    elif "hour" in data.columns:
        basket["time_of_day"] = pd.cut(
            data["hour"],
            bins=[-1, 5, 11, 17, 23],
            labels=["Night", "Morning", "Afternoon", "Evening"],
        ).astype(str)

    if "passenger_count" in data.columns:
        basket["passenger_group"] = pd.cut(
            data["passenger_count"],
            bins=[-np.inf, 1, 3, np.inf],
            labels=["single", "small_group", "large_group"],
        ).astype(str)

    if "distance" in data.columns:
        basket["distance_group"] = pd.qcut(
            data["distance"].rank(method="first"),
            q=3,
            labels=["short", "medium", "long"],
        ).astype(str)

    if "fare_amount" in data.columns:
        basket["fare_group"] = pd.qcut(
            data["fare_amount"].rank(method="first"),
            q=3,
            labels=["low_fare", "mid_fare", "high_fare"],
        ).astype(str)

    if "base" in data.columns:
        top_bases = data["base"].astype(str).value_counts().head(8).index
        basket["base_group"] = np.where(data["base"].astype(str).isin(top_bases), data["base"].astype(str), "Other")

    basket = basket.dropna()
    if basket.empty or basket.shape[1] < 2:
        return {
            "frequent_itemsets": pd.DataFrame(),
            "rules": pd.DataFrame(),
            "message": "Not enough categorical ride characteristics for association mining.",
        }

    # one-hot basket representation with prefixed feature=value labels
    encoded = pd.get_dummies(basket.astype(str), prefix=basket.columns, prefix_sep="=")
    encoded = encoded.astype(bool)

    frequent_itemsets = apriori(encoded, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return {
            "frequent_itemsets": frequent_itemsets,
            "rules": pd.DataFrame(),
            "message": "No frequent itemsets found at current support threshold.",
        }

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
    if not rules.empty:
        rules = rules.sort_values(["lift", "confidence", "support"], ascending=False)

    return {
        "frequent_itemsets": frequent_itemsets,
        "rules": rules,
        "basket_columns": basket.columns.tolist(),
    }
