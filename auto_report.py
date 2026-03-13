#!/usr/bin/env python3
"""
auto_report.py

Usage:
    python auto_report.py path/to/data.csv
    python auto_report.py path/to/data.xlsx

Generates a detailed Markdown report about the dataset:
    data_report_<original_filename>.md
"""

import os
import argparse
import textwrap
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from scipy import stats
                                                                     

def load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use CSV or Excel.")
    return df


def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns.tolist()

    # Categorical: objects, bools, or numerics with low number of unique values
    potential_cat = df.select_dtypes(include=["object", "bool"]).columns.tolist()
    for col in df.columns:
        if col in numeric_cols:
            unique_vals = df[col].nunique(dropna=True)
            if 1 < unique_vals <= max(20, 0.05 * len(df)):  # heuristic
                potential_cat.append(col)

    # Remove duplicates and avoid overlaps
    potential_cat = list(dict.fromkeys(potential_cat))  # unique while preserving order
    categorical_cols = [c for c in potential_cat if c not in datetime_cols]

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
    }


def summarize_basic_info(df: pd.DataFrame) -> str:
    lines = []
    lines.append("## 1. Basic Dataset Information\n")
    lines.append(f"- Number of rows: **{df.shape[0]}**")
    lines.append(f"- Number of columns: **{df.shape[1]}**\n")

    lines.append("### 1.1 Column Overview\n")
    lines.append("| Column | Dtype | Non-Null Count | Missing (%) | Unique Values | Example Values |")
    lines.append("|--------|-------|----------------|-------------|---------------|----------------|")

    for col in df.columns:
        non_null = df[col].notna().sum()
        missing_pct = 100 * (1 - non_null / len(df)) if len(df) > 0 else 0
        unique_vals = df[col].nunique(dropna=True)
        # Get up to 3 example values
        ex_vals = df[col].dropna().unique()[:3]
        ex_vals_str = ", ".join(map(str, ex_vals))
        lines.append(
            f"| {col} | {df[col].dtype} | {non_null} | "
            f"{missing_pct:.2f}% | {unique_vals} | {ex_vals_str} |"
        )

    lines.append("")
    return "\n".join(lines)


def summarize_missing_values(df: pd.DataFrame) -> str:
    lines = []
    lines.append("## 2. Missing Values Analysis\n")

    missing_counts = df.isna().sum()
    missing_pct = 100 * missing_counts / len(df) if len(df) > 0 else missing_counts

    if (missing_counts > 0).any():
        lines.append("| Column | Missing Count | Missing (%) |")
        lines.append("|--------|---------------|-------------|")
        for col in df.columns:
            if missing_counts[col] > 0:
                lines.append(
                    f"| {col} | {missing_counts[col]} | {missing_pct[col]:.2f}% |"
                )
        lines.append("\n**Insights:**")
        lines.append(
            "- Columns with high missing percentage might need imputation or removal."
        )
        lines.append(
            "- Patterns of missingness can indicate data quality issues or important structure."
        )
    else:
        lines.append("No missing values detected in this dataset.")

    lines.append("")
    return "\n".join(lines)


def summarize_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> str:
    lines = []
    lines.append("## 3. Numeric Features Summary\n")

    if not numeric_cols:
        lines.append("No numeric columns detected.\n")
        return "\n".join(lines)

    desc = df[numeric_cols].describe().T  # index = columns

    lines.append("| Column | Count | Mean | Std | Min | 25% | 50% | 75% | Max |")
    lines.append("|--------|-------|------|-----|-----|-----|-----|-----|-----|")

    for col in desc.index:
        row = desc.loc[col]
        lines.append(
            f"| {col} | {row['count']:.0f} | {row['mean']:.3f} | {row['std']:.3f} | "
            f"{row['min']:.3f} | {row['25%']:.3f} | {row['50%']:.3f} | {row['75%']:.3f} | {row['max']:.3f} |"
        )

    lines.append("")
    return "\n".join(lines)


def summarize_categorical(df: pd.DataFrame, cat_cols: List[str]) -> str:
    lines = []
    lines.append("## 4. Categorical Features Summary\n")

    if not cat_cols:
        lines.append("No categorical columns detected.\n")
        return "\n".join(lines)

    for col in cat_cols:
        lines.append(f"### 4.{cat_cols.index(col) + 1} Column: `{col}`")
        vc = df[col].value_counts(dropna=False)
        total = len(df)
        lines.append("| Category | Count | Percentage |")
        lines.append("|----------|-------|------------|")
        for val, cnt in vc.items():
            pct = 100 * cnt / total if total > 0 else 0
            lines.append(f"| {val} | {cnt} | {pct:.2f}% |")
        lines.append("")

    return "\n".join(lines)


def analyze_correlations(df: pd.DataFrame, numeric_cols: List[str], threshold: float = 0.7) -> str:
    lines = []
    lines.append("## 5. Correlation Analysis (Numeric Columns)\n")

    if len(numeric_cols) < 2:
        lines.append("Not enough numeric columns to compute correlations.\n")
        return "\n".join(lines)

    corr = df[numeric_cols].corr()

    lines.append("### 5.1 Correlation Matrix (top-level view)\n")
    lines.append("Correlation values range from -1 (perfect negative) to +1 (perfect positive).\n")

    # Show a truncated matrix (round)
    lines.append("| | " + " | ".join(numeric_cols) + " |")
    lines.append("|" + "|".join(["---"] * (len(numeric_cols) + 1)) + "|")
    for col in numeric_cols:
        row_vals = [f"{corr.loc[col, c]:.2f}" for c in numeric_cols]
        lines.append(f"| {col} | " + " | ".join(row_vals) + " |")

    # Strong correlations
    lines.append("\n### 5.2 Strongly Correlated Pairs\n")
    strong_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            c1, c2 = numeric_cols[i], numeric_cols[j]
            val = corr.loc[c1, c2]
            if abs(val) >= threshold:
                strong_pairs.append((c1, c2, val))

    if strong_pairs:
        lines.append(f"Pairs with absolute correlation ≥ {threshold}:")
        lines.append("| Feature 1 | Feature 2 | Correlation |")
        lines.append("|-----------|-----------|-------------|")
        for c1, c2, v in sorted(strong_pairs, key=lambda x: -abs(x[2])):
            lines.append(f"| {c1} | {c2} | {v:.3f} |")
        lines.append(
            "\n**Insights:** Strong correlations may indicate redundancy or potential predictive relationships."
        )
    else:
        lines.append(
            f"No pairs of numeric variables found with correlation ≥ {threshold} in absolute value."
        )

    lines.append("")
    return "\n".join(lines)


def analyze_cat_numeric_relationships(
    df: pd.DataFrame, cat_cols: List[str], numeric_cols: List[str]
) -> str:
    lines = []
    lines.append("## 6. Relationships Between Categorical and Numeric Variables\n")

    if not cat_cols or not numeric_cols:
        lines.append("Not enough categorical or numeric columns to analyze relationships.\n")
        return "\n".join(lines)

    # Ensure no duplicate column names in this function either
    df = df.loc[:, ~df.columns.duplicated()]

    for i, cat in enumerate(cat_cols, start=1):
        if cat not in df.columns:
            continue

        lines.append(f"### 6.{i} Categorical Feature: `{cat}`")

        for num in numeric_cols:
            # 🔹 Skip if it's the same column used as both cat & num
            if num == cat:
                continue
            if num not in df.columns:
                continue

            lines.append(f"#### Relation: `{cat}` → `{num}`")

            sub = df[[cat, num]].dropna()

            # In case we still create duplicated columns due to same name
            sub = sub.loc[:, ~sub.columns.duplicated()]

            if num not in sub.columns:
                lines.append("Not enough data to analyze.\n")
                continue

            grouped = sub.groupby(cat)[num]
            summary = grouped.agg(["count", "mean", "std"])

            if summary.empty:
                lines.append("Not enough data to analyze.\n")
                continue

            lines.append("| Category | Count | Mean | Std |")
            lines.append("|----------|-------|------|-----|")
            for idx, row in summary.iterrows():
                lines.append(
                    f"| {idx} | {row['count']:.0f} | {row['mean']:.3f} | {row['std']:.3f} |"
                )

            # One-way ANOVA if there are at least 2 groups
            if summary.shape[0] >= 2:
                groups = [g[1].values for g in grouped]
                try:
                    f_stat, p_val = stats.f_oneway(*groups)
                    lines.append(
                        f"\n- ANOVA F-statistic: **{f_stat:.3f}**, p-value: **{p_val:.3e}**"
                    )
                    if p_val < 0.05:
                        lines.append(
                            "  → Statistically significant difference in means between categories."
                        )
                    else:
                        lines.append(
                            "  → No strong statistical evidence of different means between categories."
                        )
                except Exception as e:
                    lines.append(f"- ANOVA could not be computed: {e}")
            lines.append("")

    return "\n".join(lines)



def detect_outliers_iqr(df: pd.DataFrame, numeric_cols: List[str]) -> str:
    lines = []
    lines.append("## 7. Outlier Analysis (IQR Method)\n")

    if not numeric_cols:
        lines.append("No numeric columns to detect outliers.\n")
        return "\n".join(lines)

    lines.append("| Column | Outlier Count | Outlier (%) |")
    lines.append("|--------|---------------|-------------|")

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            out_count = 0
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = series[(series < lower) | (series > upper)]
            out_count = len(outliers)
        pct = 100 * out_count / len(series) if len(series) > 0 else 0
        lines.append(f"| {col} | {out_count} | {pct:.2f}% |")

    lines.append(
        "\n**Note:** Outliers are values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]. They may be "
        "errors, rare events, or important edge cases depending on the context."
    )
    lines.append("")
    return "\n".join(lines)


def generate_conclusion(df: pd.DataFrame, types: Dict[str, List[str]]) -> str:
    lines = []
    lines.append("## 8. High-Level Summary & Suggestions\n")

    n_rows, n_cols = df.shape
    num_numeric = len(types["numeric"])
    num_cat = len(types["categorical"])
    num_dt = len(types["datetime"])

    lines.append(
        textwrap.dedent(
            f"""
            - The dataset has **{n_rows}** rows and **{n_cols}** columns.
            - Detected **{num_numeric} numeric**, **{num_cat} categorical**, and **{num_dt} datetime** columns.

            **Suggested next steps (depending on your use case):**
            - If you're building a **prediction model**, choose a target column (e.g., label, churn, price) and:
              - Check which features are most correlated with the target.
              - Handle missing values via imputation.
              - Encode categorical variables (one-hot, label encoding, etc.).
            - For **data cleaning**:
              - Investigate columns with high missing percentages.
              - Review outliers to decide if they should be capped, removed, or kept.
            - For **business insights**:
              - Focus on strong relationships between categorical features and key numeric metrics.
              - Use the correlation analysis to identify drivers of your main KPIs.
            """
        ).strip()
    )

    lines.append("")
    return "\n".join(lines)


def build_report(df: pd.DataFrame, path: str) -> str:
    types = detect_column_types(df)

    report_sections = [
        f"# Automated Data Analysis Report\n\nSource file: `{os.path.basename(path)}`\n",
        summarize_basic_info(df),
        summarize_missing_values(df),
        summarize_numeric(df, types["numeric"]),
        summarize_categorical(df, types["categorical"]),
        analyze_correlations(df, types["numeric"]),
        analyze_cat_numeric_relationships(df, types["categorical"], types["numeric"]),
        detect_outliers_iqr(df, types["numeric"]),
        generate_conclusion(df, types),
    ]

    return "\n\n".join(report_sections)


def build_short_summary(
    df: pd.DataFrame,
    types: Dict[str, List[str]],
    path: str,
    max_lines: int = 100,
) -> str:
    """
    Build a compact textual summary of the dataset in <= max_lines (roughly).
    """
    lines = []
    n_rows, n_cols = df.shape
    numeric_cols = types.get("numeric", [])
    cat_cols = types.get("categorical", [])
    dt_cols = types.get("datetime", [])

    lines.append(f"# Short Data Summary\n")
    lines.append(f"Source file: `{os.path.basename(path)}`\n")

    # 1. Basic shape
    lines.append("## 1. Dataset Snapshot")
    lines.append(f"- Rows: **{n_rows}**")
    lines.append(f"- Columns: **{n_cols}**")
    lines.append(f"- Numeric columns: **{len(numeric_cols)}**")
    lines.append(f"- Categorical columns: **{len(cat_cols)}**")
    lines.append(f"- Datetime columns: **{len(dt_cols)}**\n")

    # 2. Missing values (top 5)
    lines.append("## 2. Missing Values (Top Columns)")
    missing_counts = df.isna().sum()
    total = len(df) if len(df) > 0 else 1
    missing_pct = 100 * missing_counts / total
    mv = missing_pct[missing_pct > 0].sort_values(ascending=False).head(5)

    if not mv.empty:
        for col, pct in mv.items():
            lines.append(f"- `{col}` → {pct:.2f}% missing")
    else:
        lines.append("- No missing values detected.")
    lines.append("")

    # 3. Key numeric stats (top 3 numeric columns)
    if numeric_cols:
        lines.append("## 3. Numeric Columns (Quick Stats)")
        desc = df[numeric_cols].describe().T
        # pick up to 3 numeric columns with most non-null values
        top_num = desc.sort_values("count", ascending=False).head(3)
        for col, row in top_num.iterrows():
            lines.append(
                f"- `{col}` → mean: {row['mean']:.3f}, std: {row['std']:.3f}, "
                f"min: {row['min']:.3f}, max: {row['max']:.3f}"
            )
        lines.append("")
    else:
        lines.append("## 3. Numeric Columns (Quick Stats)")
        lines.append("- No numeric columns detected.\n")

    # 4. Strong correlations (top 5)
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                c1, c2 = numeric_cols[i], numeric_cols[j]
                val = corr.loc[c1, c2]
                if not np.isnan(val):
                    pairs.append((c1, c2, val, abs(val)))
        lines.append("## 4. Strong Correlations (|corr| ≥ 0.6)")

        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()

            strong_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    c1, c2 = numeric_cols[i], numeric_cols[j]
                    val = corr.loc[c1, c2]
                    if not np.isnan(val) and abs(val) >= 0.6:
                        strong_pairs.append((c1, c2, float(val), abs(float(val))))

            # sort by absolute correlation strongest first
            strong_pairs = sorted(strong_pairs, key=lambda x: -x[3])

            if strong_pairs:
                for c1, c2, v, _ in strong_pairs:
                    lines.append(f"- `{c1}` ↔ `{c2}` → corr = {v:.3f}")
            else:
                lines.append("- No correlation pairs ≥ 0.6 found.")
        else:
            lines.append("- Not enough numeric columns for correlation analysis.")

        lines.append("")


    # 5. Outlier-heavy columns (top 5)
    lines.append("## 5. Outlier Overview (IQR Method)")
    if numeric_cols:
        outlier_info = []
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            out_count = ((series < lower) | (series > upper)).sum()
            pct = 100 * out_count / len(series) if len(series) > 0 else 0
            outlier_info.append((col, pct))
        outlier_info = sorted(outlier_info, key=lambda x: -x[1])[:5]
        if outlier_info:
            for col, pct in outlier_info:
                lines.append(f"- `{col}` → approx. {pct:.2f}% outliers")
        else:
            lines.append("- No notable outliers detected.")
    else:
        lines.append("- No numeric columns for outlier detection.")
    lines.append("")

    # 6. Quick suggestions
    lines.append("## 6. Quick Suggestions")
    lines.append("- Handle missing values in high-missing columns (impute or drop).")
    lines.append("- Review outlier-heavy numeric fields to decide whether to cap, remove, or keep them.")
    lines.append("- Use strongly correlated numeric features carefully to avoid redundancy.")
    if cat_cols and numeric_cols:
        lines.append("- Explore how key categorical features (like department/role/location) affect numeric KPIs (like CTC).")
    lines.append("")

    # Enforce max_lines (just in case)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + ["\n…(truncated to keep summary short)…\n"]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Automatic data analysis & report generator.")
    parser.add_argument("file", help="Path to the CSV or Excel file")
    args = parser.parse_args()

    path = args.file
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    print(f"📥 Loading data from: {path}")
    df = load_data(path)
    print(f"✅ Loaded dataset with shape: {df.shape}")

    # Optional cleaning you already added:
    df = df.loc[:, ~df.columns.duplicated()]
    df.columns = df.columns.astype(str).str.strip()
    print(f"✅ Final columns used ({len(df.columns)}): {list(df.columns)}")

    print("🧠 Analyzing data and generating report...")
    # Full detailed report
    report = build_report(df, path)

    base_name = os.path.splitext(os.path.basename(path))[0]
    out_name = f"data_report_{base_name}.md"
    with open(out_name, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"📄 Full report generated: {out_name}")

    # 🔹 Short summary (<= 100 lines)
    types = detect_column_types(df)
    summary = build_short_summary(df, types, path, max_lines=100)
    summary_name = f"data_summary_{base_name}.md"
    with open(summary_name, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"📝 Short summary generated: {summary_name}")


if __name__ == "__main__":
    main()
