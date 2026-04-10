"""
Data Preprocessing Pipeline
=============================
Handles feature engineering, encoding, scaling, and train/test splitting
for the shipment delay prediction models.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ── Feature definitions ──────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "distance_km",
    "departure_hour",
    "day_of_week",
    "is_weekend",
    "carrier_reliability_score",
    "weather_severity",
    "traffic_congestion",
    "has_news_disruption",
]

CATEGORICAL_FEATURES = [
    "origin",
    "destination",
    "route_type",
]

TARGET = "delayed"

# Columns to drop (not features)
DROP_COLUMNS = ["shipment_id", "carrier_id", "delay_probability"]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def load_data(filepath: str = None) -> pd.DataFrame:
    """Load the shipment dataset from CSV."""
    if filepath is None:
        filepath = os.path.join(DATA_DIR, "shipments.csv")
    df = pd.read_csv(filepath)
    print(f"📦 Loaded dataset: {len(df):,} rows, {df.shape[1]} columns")
    return df


def print_data_source(df: pd.DataFrame):
    """Identify and print whether the data is real or synthetic."""
    # Real data from Kaggle has exactly 10,999 rows (before live additions)
    # and city distributions that differ from synthetic uniform random
    n_rows = len(df)
    unique_origins = df["origin"].nunique()
    has_live_ids = df["shipment_id"].str.contains("LIVE").any() if "shipment_id" in df.columns else False

    if n_rows == 10_999 or (n_rows > 10_990 and n_rows <= 11_100 and not has_live_ids):
        source = "REAL (Kaggle E-Commerce Shipping Dataset)"
    elif n_rows == 10_000 and unique_origins <= 14:
        source = "SYNTHETIC (numpy.random generated)"
    else:
        source = f"UNKNOWN ({n_rows:,} rows, {unique_origins} origins)"

    print(f"📋 Data source: {source}")
    return source


def validate_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Validate data quality: check nulls, invalid ranges, duplicates."""
    print("🔍 Validating data quality...")
    issues = []

    # Check for null values in features
    null_counts = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]].isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        for col, count in null_cols.items():
            issues.append(f"   ⚠️  {col}: {count} null values")
        # Fill numeric nulls with median, categorical with mode
        for col in NUMERIC_FEATURES:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        for col in CATEGORICAL_FEATURES:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
        if df[TARGET].isnull().any():
            df = df.dropna(subset=[TARGET])

    # Check for invalid ranges
    if (df["distance_km"] <= 0).any():
        bad = (df["distance_km"] <= 0).sum()
        issues.append(f"   ⚠️  distance_km: {bad} non-positive values")
        df = df[df["distance_km"] > 0]

    if not df["departure_hour"].between(0, 23).all():
        bad = (~df["departure_hour"].between(0, 23)).sum()
        issues.append(f"   ⚠️  departure_hour: {bad} out-of-range values")
        df["departure_hour"] = df["departure_hour"].clip(0, 23)

    if not df["day_of_week"].between(0, 6).all():
        bad = (~df["day_of_week"].between(0, 6)).sum()
        issues.append(f"   ⚠️  day_of_week: {bad} out-of-range values")
        df["day_of_week"] = df["day_of_week"].clip(0, 6)

    if not df["delayed"].isin([0, 1]).all():
        bad = (~df["delayed"].isin([0, 1])).sum()
        issues.append(f"   ⚠️  delayed: {bad} non-binary values")
        df = df[df["delayed"].isin([0, 1])]

    # Check for duplicate shipment IDs
    if "shipment_id" in df.columns:
        dupes = df["shipment_id"].duplicated().sum()
        if dupes > 0:
            issues.append(f"   ⚠️  {dupes} duplicate shipment_id values")
            df = df.drop_duplicates(subset=["shipment_id"], keep="first")

    if issues:
        print(f"   Found {len(issues)} issue(s):")
        for issue in issues:
            print(issue)
        print(f"   ✅ Auto-fixed. Clean records: {len(df):,}")
    else:
        print("   ✅ All data quality checks passed!")

    return df


def create_preprocessor() -> ColumnTransformer:
    """Create a sklearn ColumnTransformer for feature preprocessing."""
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    save_models: bool = True,
) -> tuple:
    """
    Prepare data for model training.
    
    Returns:
        X_train, X_test, y_train, y_test, preprocessor (fitted)
    """
    # Data quality validation
    df = validate_data_quality(df)

    # Identify data source
    print_data_source(df)

    # Drop non-feature columns
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df_clean = df.drop(columns=cols_to_drop)

    # Separate features and target
    X = df_clean.drop(columns=[TARGET])
    y = df_clean[TARGET]

    # Train/test split (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"📊 Train set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")
    print(f"   Train delay rate: {y_train.mean():.1%} | Test delay rate: {y_test.mean():.1%}")

    # Fit preprocessor on training data
    preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Save preprocessor
    if save_models:
        os.makedirs(MODELS_DIR, exist_ok=True)
        preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
        joblib.dump(preprocessor, preprocessor_path)
        print(f"💾 Preprocessor saved: {preprocessor_path}")

    # Get feature names after transformation
    feature_names = get_feature_names(preprocessor)
    print(f"   Total features after preprocessing: {len(feature_names)}")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Extract feature names from the preprocessor."""
    feature_names = []

    # Numeric features (unchanged names)
    feature_names.extend(NUMERIC_FEATURES)

    # Categorical features (one-hot encoded names)
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names.extend(cat_feature_names.tolist())

    return feature_names


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df)
    print(f"\n✅ Preprocessing complete!")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test shape:  {X_test.shape}")
    print(f"   Feature names: {feature_names[:10]}... ({len(feature_names)} total)")
