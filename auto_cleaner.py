#!/usr/bin/env python3
"""
auto_cleaner.py

Usage:
    python auto_cleaner.py input.csv
    python auto_cleaner.py input.xlsx

Outputs:
    cleaned_<filename>.csv
    cleaning_log_<filename>.md
"""

import os
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List


# --------------------------------------------------
# Load data
# --------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format")


# --------------------------------------------------
# Column standardization
# --------------------------------------------------
def clean_columns(df: pd.DataFrame, log: List[str]) -> pd.DataFrame:
    original_cols = df.columns.tolist()

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_")
        .str.replace(r"[^\w]", "", regex=True)
        .str.lower()
    )

    df = df.loc[:, ~df.columns.duplicated()]

    log.append("## Column Cleaning")
    for o, n in zip(original_cols, df.columns):
        if o != n:
            log.append(f"- `{o}` → `{n}`")

    return df


# --------------------------------------------------
# Detect column types
# --------------------------------------------------
def detect_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    datetime = df.select_dtypes(include="datetime").columns.tolist()

    categorical = [
        c for c in df.columns
        if c not in numeric and c not in datetime
    ]

    return {
        "numeric": numeric,
        "categorical": categorical,
        "datetime": datetime
    }


# --------------------------------------------------
# Convert data types safely
# --------------------------------------------------
def fix_dtypes(df: pd.DataFrame, log: List[str]) -> pd.DataFrame:
    log.append("\n## Data Type Fixes")

    for col in df.columns:
        if df[col].dtype == object:
            # Try numeric
            converted = pd.to_numeric(df[col], errors="ignore")
            if converted.dtype != object:
                df[col] = converted
                log.append(f"- `{col}` converted to numeric")
                continue

            # Try datetime
            dt = pd.to_datetime(df[col], errors="ignore")
            if dt.dtype != object:
                df[col] = dt
                log.append(f"- `{col}` converted to datetime")

    return df


# --------------------------------------------------
# Handle missing values
# --------------------------------------------------
def handle_missing(df: pd.DataFrame, types: Dict, log: List[str]) -> pd.DataFrame:
    log.append("\n## Missing Value Handling")

    for col in types["numeric"]:
        if df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
            log.append(f"- `{col}` → filled numeric missing with median ({median:.3f})")

    for col in types["categorical"]:
        if df[col].isna().any():
            mode = df[col].mode()
            fill_val = mode.iloc[0] if not mode.empty else "unknown"
            df[col] = df[col].fillna(fill_val)
            log.append(f"- `{col}` → filled categorical missing with mode (`{fill_val}`)")

    for col in types["datetime"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(method="ffill")
            log.append(f"- `{col}` → forward-filled datetime missing")

    return df


# --------------------------------------------------
# Outlier treatment (IQR capping)
# --------------------------------------------------
def handle_outliers(df: pd.DataFrame, numeric_cols: List[str], log: List[str]) -> pd.DataFrame:
    log.append("\n## Outlier Handling (IQR Capping)")

    for col in numeric_cols:
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((series < lower) | (series > upper)).sum()
        if outliers > 0:
            df[col] = series.clip(lower, upper)
            log.append(f"- `{col}` → capped {outliers} outliers")

    return df


# --------------------------------------------------
# Normalize categorical text
# --------------------------------------------------
def normalize_categories(df: pd.DataFrame, cat_cols: List[str], log: List[str]) -> pd.DataFrame:
    log.append("\n## Categorical Normalization")

    for col in cat_cols:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )
            log.append(f"- `{col}` → normalized text")

    return df


# --------------------------------------------------
# Remove duplicates
# --------------------------------------------------
def remove_duplicates(df: pd.DataFrame, log: List[str]) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)

    log.append("\n## Duplicate Removal")
    log.append(f"- Removed {removed} duplicate rows")

    return df


# --------------------------------------------------
# Main cleaning pipeline
# --------------------------------------------------
def clean_data(df: pd.DataFrame) -> (pd.DataFrame, str):
    log = ["# Data Cleaning Report\n"]

    df = clean_columns(df, log)
    df = fix_dtypes(df, log)

    types = detect_types(df)

    df = handle_missing(df, types, log)
    df = handle_outliers(df, types["numeric"], log)
    df = normalize_categories(df, types["categorical"], log)
    df = remove_duplicates(df, log)

    log.append("\n## Final Dataset Summary")
    log.append(f"- Rows: {df.shape[0]}")
    log.append(f"- Columns: {df.shape[1]}")

    return df, "\n".join(log)


# --------------------------------------------------
# CLI
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Automatic Data Cleaner")
    parser.add_argument("file", help="Input CSV or Excel file")
    args = parser.parse_args()

    path = args.file
    base = os.path.splitext(os.path.basename(path))[0]

    print(f"📥 Loading: {path}")
    df = load_data(path)

    print("🧹 Cleaning data...")
    cleaned_df, log = clean_data(df)

    out_data = f"cleaned_{base}.csv"
    out_log = f"cleaning_log_{base}.md"

    cleaned_df.to_csv(out_data, index=False)
    with open(out_log, "w", encoding="utf-8") as f:
        f.write(log)

    print(f"✅ Cleaned data saved: {out_data}")
    print(f"🧾 Cleaning report saved: {out_log}")


if __name__ == "__main__":
    main()
