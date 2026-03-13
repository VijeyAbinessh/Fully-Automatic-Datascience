#!/usr/bin/env python3
"""
resource_finder.py — CLEAN VERSION (no Kaggle errors)

This version:
 - Does NOT import Kaggle unless --use-kaggle flag is used
 - Prevents 'Could not find kaggle.json' warnings
 - Works out-of-the-box with HuggingFace + GitHub only
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional
import requests
from huggingface_hub import HfApi

hf_api = HfApi()
from google import genai

# ---------------------------
# Gemini LLM setup
# ---------------------------
GEMINI_MODEL = "models/gemini-2.5-flash"

_gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

SYSTEM_PROMPT = """
You are an AI research assistant whose PRIMARY goal is dataset discovery.

You MUST ALWAYS output:
- dataset_search_queries: a NON-EMPTY list of short, noun-based dataset search phrases
  that would work on HuggingFace, Kaggle, or UCI.

Rules:
- Never return an empty dataset_search_queries list
- Avoid sentences
- Avoid words like "vs", "want", "find"
- Use dataset-style phrases only
- If the problem is vague, infer the MOST LIKELY dataset interpretation

Example:
Problem: fertilizer usage vs crop yield
dataset_search_queries:
- crop yield dataset
- fertilizer recommendation dataset
- soil nutrients crop yield
- agriculture tabular dataset

Return ONLY valid JSON.
"""



def analyze_problem_llm(problem: str) -> Dict[str, Any]:
    prompt = f"""
Problem Statement:
\"\"\"{problem}\"\"\"

Return JSON in this exact format:
{{
  "task_type": "",
  "domain": "",
  "keywords": [],
  "dataset_search_queries": [],
  "known_datasets": [],
  "reasoning": ""
}}
"""

    response = _gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[SYSTEM_PROMPT, prompt],
        config={"temperature": 0.2}
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)


# ---------------------------
# Task heuristics
# ---------------------------
TASK_KEYWORDS = {
    "text-classification": ["sentiment", "classification", "review", "spam", "intent"],
    "object-detection": ["detect", "object", "bounding box", "yolo", "bbox"],
    "image-classification": ["image", "photo", "classify"],
    "segmentation": ["segmentation", "mask"],
    "qa": ["question answering", "qa"],
    "translation": ["translate"],
    "timeseries-forecasting": ["time series", "forecast"],
    "anomaly-detection": ["anomaly", "outlier"],
}

DEFAULT_METRICS = {
    "object-detection": ["mAP", "IoU"],
    "image-classification": ["accuracy", "top-5"],
    "text-classification": ["accuracy", "f1"],
    "timeseries-forecasting": ["RMSE", "MAE"],
}


def infer_task_type(problem: str) -> str:
    p = problem.lower()

    # 🔥 FINANCIAL / PRICE / FORECASTING
    if any(x in p for x in [
        "price", "forecast", "prediction", "predict",
        "time series", "stock", "gold", "commodity",
        "market", "rate"
    ]):
        return "timeseries-forecasting"

    for task, kws in TASK_KEYWORDS.items():
        for kw in kws:
            if kw in p:
                return task

    return "general"



def minimal_data_requirements(task: str) -> Dict[str, Any]:
    return {
        "modalities": ["image"] if "image" in task else ["text"],
        "labels_needed": ["class_label"],
        "min_examples_per_class": 500,
        "privacy_constraints": "Check for PII and licensing conditions"
    }

# ---------------------------
# Search HuggingFace
# ---------------------------
def search_hf_datasets(query: str, limit: int = 5):
    try:
        results = hf_api.list_datasets(search=query, limit=limit)
        return [
            {
                "name": ds.id.split("/")[-1],
                "id": ds.id,
                "link": f"https://huggingface.co/datasets/{ds.id}",
                "description": getattr(ds, "description", "")[:200]
            }
            for ds in results
        ]
    except Exception as e:
        print("[WARN] HuggingFace query failed:", e)
        return []

# ---------------------------
# Search GitHub
# ---------------------------
def search_github(query: str, limit: int = 5):
    url = "https://api.github.com/search/repositories"
    params = {"q": query, "sort": "stars", "order": "desc", "per_page": limit}

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        items = r.json().get("items", [])
        return [
            {
                "name": item["full_name"],
                "link": item["html_url"],
                "description": item.get("description", ""),
                "stars": item.get("stargazers_count", 0)
            }
            for item in items
        ]
    except Exception as e:
        print("[WARN] GitHub search failed:", e)
        return []

# ---------------------------
# Kaggle (optional)
# ---------------------------
def search_kaggle(query: str, limit: int = 5):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[INFO] Kaggle not installed, skipping.")
        return []

    try:
        api = KaggleApi()
        api.authenticate()
        results = api.dataset_list(search=query, max_size=limit)  # <-- FIXED
        return [
            {
                "name": d.title,
                "link": f"https://www.kaggle.com/datasets/{d.ref}"
            }
            for d in results
        ]
    except Exception as e:
        print("[WARN] Kaggle search failed:", e)
        return []


# ---------------------------
# Build output JSON
# ---------------------------
def build_output(problem, task, hf_results, gh_results, kaggle_results):
    p = problem.lower()

    # ✅ DEFAULTS (ALWAYS DEFINED)
    models = [
        {"name": "LinearRegression", "framework": "scikit-learn", "when": "Safe baseline"}
    ]
    metrics = ["RMSE"]

    # 🌱 AGRICULTURE / TABULAR REGRESSION
    if any(x in p for x in ["fertilizer", "crop", "yield", "soil", "agriculture"]):
        models = [
            {"name": "LinearRegression", "framework": "scikit-learn", "when": "Baseline"},
            {"name": "RandomForestRegressor", "framework": "scikit-learn", "when": "Strong tabular model"},
            {"name": "XGBoostRegressor", "framework": "xgboost", "when": "High performance"}
        ]
        metrics = ["RMSE", "MAE", "R²"]

    # 📈 TIME SERIES / PRICE FORECASTING
    elif task == "timeseries-forecasting":
        models = [
            {"name": "ARIMA", "framework": "statsmodels", "when": "Statistical baseline"},
            {"name": "Prophet", "framework": "prophet", "when": "Seasonality-aware"},
            {"name": "LSTM", "framework": "PyTorch / TensorFlow", "when": "Deep learning"},
            {"name": "XGBoost", "framework": "xgboost", "when": "Tabular forecasting"}
        ]
        metrics = ["RMSE", "MAE", "MAPE"]

    # 🖼️ COMPUTER VISION
    elif task == "object-detection":
        models = [
            {"name": "YOLOv8", "framework": "PyTorch", "when": "Real-time detection"}
        ]
        metrics = ["mAP", "IoU"]

    return {
        "problem_summary": problem,
        "task_type": task,
        "candidate_datasets": hf_results + kaggle_results,
        "baseline_code_repos": gh_results,
        "recommended_models": models,
        "evaluation_metrics": metrics,
        "next_steps": [
            "Download datasets",
            "Create train/val/test split",
            "Train baseline model",
            "Calibrate hyperparameters",
            "Deploy model via FastAPI"
        ]
    }

# ---------------------------
def find_resources_ui(problem: str, limit: int = 5):
    llm_result = {}  # ✅ FIX: always defined

    try:
        llm_result = analyze_problem_llm(problem)
        task = llm_result.get("task_type", "general")
        keywords = llm_result.get("keywords", [])
        domain = llm_result.get("domain", "")
    except Exception as e:
        print("[WARN] LLM failed, falling back to heuristics:", e)
        task = infer_task_type(problem)
        keywords = [problem]
        domain = "unknown"

    # ✅ SAFE: llm_result always exists now
    dataset_queries = llm_result.get("dataset_search_queries", [])

    # 🔥 STRONG SAFETY NET FOR GENERIC AGRICULTURE TERMS (UI + API)
    if not dataset_queries or len(" ".join(dataset_queries)) < 12:
        if any(x in problem.lower() for x in ["fertilizer", "crop", "agriculture", "soil"]):
            dataset_queries = [
                "fertilizer recommendation dataset",
                "soil nutrients dataset",
                "crop yield dataset",
                "agriculture tabular dataset"
            ]
        else:
            dataset_queries = keywords


    if not dataset_queries:
        dataset_queries = keywords  # fallback

    search_query = " ".join(dataset_queries[:5])

    hf_data = search_hf_datasets(search_query, limit)
    gh_query = f"{search_query} {task} forecasting language:python stars:>20"
    gh_data = search_github(gh_query, limit)

    output = build_output(
        problem=problem,
        task=task,
        hf_results=hf_data,
        gh_results=gh_data,
        kaggle_results=[]
    )

    output["llm_analysis"] = {
        "keywords": keywords,
        "domain": domain,
        "reasoning": llm_result.get("reasoning", "")
    }

    return output


# ---------------------------
# CLI Entry Point
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", nargs="+", help="Problem statement")
    parser.add_argument("--use-kaggle", action="store_true", help="Enable Kaggle search")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    problem = " ".join(args.problem)

    # 🔥 LLM-first analysis
    try:
        llm_result = analyze_problem_llm(problem)
        task = llm_result.get("task_type", infer_task_type(problem))
        keywords = llm_result.get("keywords", [problem])
        domain = llm_result.get("domain", "")
    except Exception as e:
        print("[WARN] LLM failed, using heuristics:", e)
        task = infer_task_type(problem)
        keywords = [problem]
        domain = "unknown"

    # 🔑 USE LLM KEYWORDS FOR SEARCH
    # 🔑 USE GEMINI DATASET SEARCH QUERIES
    dataset_queries = llm_result.get("dataset_search_queries", [])

    # 🔥 STRONG SAFETY NET FOR GENERIC AGRICULTURE TERMS
    if not dataset_queries or len(" ".join(dataset_queries)) < 12:
        if any(x in problem.lower() for x in ["fertilizer", "crop", "agriculture", "soil"]):
            dataset_queries = [
                "fertilizer recommendation dataset",
                "soil nutrients dataset",
                "crop yield dataset",
                "agriculture tabular dataset"
            ]
        else:
            dataset_queries = keywords


    if not dataset_queries:
        dataset_queries = keywords  # safe fallback

    search_query = " ".join(dataset_queries[:5])


    print(f"\n🔍 Problem: {problem}")
    print(f"📌 Inferred Task: {task}")
    print(f"🔑 Keywords: {keywords}\n")

    print("➡ Searching HuggingFace…")
    hf_data = search_hf_datasets(search_query, args.limit)

    print("➡ Searching GitHub…")
    gh_data = search_github(search_query + " " + task, args.limit)

    kaggle_data = []
    if args.use_kaggle:
        print("➡ Searching Kaggle…")
        kaggle_data = search_kaggle(search_query, args.limit)

    output = build_output(problem, task, hf_data, gh_data, kaggle_data)

    # 🔥 Attach LLM analysis
    output["llm_analysis"] = {
        "keywords": keywords,
        "domain": domain,
        "reasoning": llm_result.get("reasoning", "") if "llm_result" in locals() else ""
    }

    print("\n✅ RESULTS:\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))

    with open("resource_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n📁 Saved as resource_output.json")


if __name__ == "__main__":
    main()

