#!/usr/bin/env python3
"""
auto_trainer.py

Usage:
    python auto_trainer.py cleaned_data.csv

Outputs:
    trained_model_<algorithm>.pkl
"""

import os
import argparse
import difflib
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import r2_score, mean_squared_error


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

def train_model_ui(
    df: pd.DataFrame,
    target: str,
    features: list,
    task: str,
    model_choice: str
):
    X = df[features]
    y = df[target]

    algorithms = get_algorithms(task)
    # Handle auto model selection
    if model_choice == "auto":
        suggested = suggest_algorithm(task, len(df), len(features))
        for k, (name, mdl) in algorithms.items():
            if suggested.lower().startswith(name.lower().split()[0]):
                model_choice = k
                break

    algo_name, model = algorithms.get(model_choice)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {}
    if task == "classification":
        metrics["accuracy"] = accuracy_score(y_test, y_pred)
    else:
        metrics["r2"] = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        metrics["rmse"] = mse ** 0.5


    model_file = f"trained_model_{algo_name.replace(' ', '_').lower()}.pkl"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(BASE_DIR, "..", "outputs")


    # ✅ Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, model_file)
    joblib.dump(pipeline, output_path)

    return model_file, metrics


# --------------------------------------------------
# Suggest algorithm
# --------------------------------------------------
def suggest_algorithm(task: str, n_samples: int, n_features: int) -> str:
    if task == "classification":
        return "Random Forest Classifier" if n_samples > 1000 else "Logistic Regression"
    else:
        return "Random Forest Regressor" if n_features > 5 else "Linear Regression"


# --------------------------------------------------
# Available algorithms
# --------------------------------------------------
def get_algorithms(task: str):
    if task == "classification":
        return {
            "1": ("Logistic Regression", LogisticRegression(max_iter=1000)),
            "2": ("Decision Tree", DecisionTreeClassifier()),
            "3": ("Random Forest", RandomForestClassifier()),
            "4": ("SVM", SVC())
        }
    else:
        return {
            "1": ("Linear Regression", LinearRegression()),
            "2": ("Decision Tree", DecisionTreeRegressor()),
            "3": ("Random Forest", RandomForestRegressor())
        }


# --------------------------------------------------
# Main training logic
# --------------------------------------------------
def train_model(df: pd.DataFrame):
    print("\n📌 Available columns:")
    for c in df.columns:
        print(" -", c)

    target = input("\n🎯 Enter dependent (target) column: ").strip()

    if target not in df.columns:
        print("❌ Target column not found.")
        return

    features = input(
        "📥 Enter independent columns (comma-separated) "
        "or press Enter for auto-select: "
    ).strip()

    # -------- Feature selection (SAFE) --------
    if features:
        requested = [c.strip() for c in features.split(",") if c.strip()]
        valid, invalid = [], []

        for col in requested:
            if col in df.columns:
                valid.append(col)
            else:
                invalid.append(col)

        if invalid:
            print("\n❌ Invalid column names:")
            for col in invalid:
                suggestion = difflib.get_close_matches(col, df.columns, n=1)
                if suggestion:
                    print(f" - {col} → did you mean `{suggestion[0]}`?")
                else:
                    print(f" - {col} → no close match")
            print("\n⚠️ Re-run and enter valid column names.")
            return

        X = df[valid]
    else:
        X = df.drop(columns=[target])

    y = df[target]

    # -------- Task selection --------
    print("\n📊 Select task type:")
    print("1. Classification")
    print("2. Regression")

    task_choice = input("Enter choice (1/2): ").strip()
    task = "classification" if task_choice == "1" else "regression"

    algorithms = get_algorithms(task)

    suggested = suggest_algorithm(task, len(df), X.shape[1])
    print(f"\n🤖 Suggested Algorithm: {suggested}")

    print("\n📚 Available Algorithms:")
    for k, v in algorithms.items():
        print(f"{k}. {v[0]}")

    choice = input("\n🔢 Choose algorithm number: ").strip()
    algo_name, model = algorithms.get(choice, list(algorithms.values())[0])

    print(f"\n🚀 Training using: {algo_name}")

    # -------- Train-test split --------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------- Auto preprocessing --------
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    print("\n🔎 Feature types detected:")
    print("Numeric:", num_cols)
    print("Categorical:", cat_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # -------- Train --------
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    # -------- Metrics --------
    print("\n📈 Model Performance")

    if task == "classification":
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Accuracy: {acc:.4f}")
        print("\n📄 Classification Report:")
        print(classification_report(y_test, y_pred))
    else:
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"✅ R² Score: {r2:.4f}")
        print(f"📉 RMSE: {rmse:.4f}")

    # -------- Save model --------
    model_file = f"trained_model_{algo_name.replace(' ', '_').lower()}.pkl"

    output_path = os.path.join("outputs", model_file)
    os.makedirs("outputs", exist_ok=True)

    joblib.dump(pipeline, output_path)

    print(f"\n💾 Model saved as: {model_file}")


# --------------------------------------------------
# CLI
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Automatic ML Model Trainer")
    parser.add_argument("file", help="Cleaned CSV or Excel file")
    args = parser.parse_args()

    print(f"\n📥 Loading dataset: {args.file}")
    df = load_data(args.file)

    train_model(df)


if __name__ == "__main__":
    main()
