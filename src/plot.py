import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Path setup
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.basename(BASE_DIR) == "src":
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
else:
    PROJECT_ROOT = BASE_DIR

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CSV_DIR = os.path.join(RESULTS_DIR, "csv")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)


# =========================
# Helpers
# =========================
def find_existing_file(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def clean_columns(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df


def standardize_model_name(name: str) -> str:
    name = str(name).strip().lower()

    if "most frequent" in name:
        return "Most Frequent Baseline"
    if "naive bayes" in name:
        return "Naive Bayes"
    if "logistic regression" in name:
        return "Logistic Regression"
    if "bert" in name:
        return "BERT"

    return str(name).strip()


def find_summary_file_paths():
    baseline_candidates = [
        os.path.join(CSV_DIR, "model_results_summary.csv"),
        os.path.join(RESULTS_DIR, "model_results_summary.csv"),
        os.path.join(PROJECT_ROOT, "model_results_summary.csv"),
    ]

    bert_candidates = [
        os.path.join(CSV_DIR, "bert_results_summary.csv"),
        os.path.join(RESULTS_DIR, "bert_results_summary.csv"),
        os.path.join(PROJECT_ROOT, "bert_results_summary.csv"),
    ]

    baseline_path = find_existing_file(baseline_candidates)
    bert_path = find_existing_file(bert_candidates)

    if baseline_path is None:
        raise FileNotFoundError("Cannot find model_results_summary.csv")
    if bert_path is None:
        raise FileNotFoundError("Cannot find bert_results_summary.csv")

    return baseline_path, bert_path


# =========================
# Summary parsing
# =========================
def parse_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected columns: Metric, Value
    Example rows:
      Most Frequent Accuracy (Test), 0.2500
      Naive Bayes Precision (Test), 0.9024
      BERT F1 (Test), 0.9487
    """
    df = clean_columns(df)

    if "Metric" not in df.columns or "Value" not in df.columns:
        raise ValueError("Not long format")

    test_df = df[df["Metric"].astype(str).str.contains(r"\(Test\)", regex=True, na=False)].copy()

    rows = []
    for _, row in test_df.iterrows():
        metric_name = str(row["Metric"]).replace("(Test)", "").strip()
        value = float(row["Value"])

        model = standardize_model_name(metric_name)

        if "Accuracy" in metric_name:
            metric = "Accuracy"
        elif "Precision" in metric_name:
            metric = "Precision"
        elif "Recall" in metric_name:
            metric = "Recall"
        elif "F1" in metric_name:
            metric = "F1"
        else:
            continue

        rows.append({"Model": model, "Metric": metric, "Value": value})

    if not rows:
        raise ValueError("No usable test rows in long format")

    out = pd.DataFrame(rows)
    out = out.pivot(index="Model", columns="Metric", values="Value").reset_index()
    return out


def parse_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected columns like:
      Model, Accuracy, Precision, Recall, F1
    or lowercase variants.
    """
    df = clean_columns(df)

    col_map = {c.lower(): c for c in df.columns}

    model_col = col_map.get("model")
    accuracy_col = col_map.get("accuracy")
    precision_col = col_map.get("precision")
    recall_col = col_map.get("recall")
    f1_col = col_map.get("f1") or col_map.get("f1 score") or col_map.get("f1_score")

    if not model_col or not accuracy_col or not precision_col or not recall_col or not f1_col:
        raise ValueError("Not wide format")

    out = pd.DataFrame({
        "Model": df[model_col].apply(standardize_model_name),
        "Accuracy": pd.to_numeric(df[accuracy_col], errors="coerce"),
        "Precision": pd.to_numeric(df[precision_col], errors="coerce"),
        "Recall": pd.to_numeric(df[recall_col], errors="coerce"),
        "F1": pd.to_numeric(df[f1_col], errors="coerce"),
    })

    out = out.dropna(subset=["Model", "Accuracy", "Precision", "Recall", "F1"])
    return out


def parse_summary_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = clean_columns(df)

    # Try long format first
    try:
        return parse_long_format(df)
    except Exception:
        pass

    # Then try wide format
    try:
        return parse_wide_format(df)
    except Exception:
        pass

    raise ValueError(
        f"Unsupported CSV format in {path}. "
        f"Columns found: {list(df.columns)}"
    )


def load_summary_results() -> pd.DataFrame:
    baseline_path, bert_path = find_summary_file_paths()

    baseline_df = parse_summary_file(baseline_path)
    bert_df = parse_summary_file(bert_path)

    combined = pd.concat([baseline_df, bert_df], ignore_index=True)

    # Remove duplicate models if the same model appears twice
    combined = combined.groupby("Model", as_index=False)[["Accuracy", "Precision", "Recall", "F1"]].max()

    model_order = [
        "Most Frequent Baseline",
        "Naive Bayes",
        "Logistic Regression",
        "BERT",
    ]

    combined["Model"] = pd.Categorical(combined["Model"], categories=model_order, ordered=True)
    combined = combined.sort_values("Model").reset_index(drop=True)

    return combined


# =========================
# Error parsing
# =========================
def get_error_file_paths():
    return {
        "Most Frequent Baseline": find_existing_file([
            os.path.join(CSV_DIR, "errors_most_frequent_test.csv"),
            os.path.join(RESULTS_DIR, "errors_most_frequent_test.csv"),
            os.path.join(PROJECT_ROOT, "errors_most_frequent_test.csv"),
        ]),
        "Naive Bayes": find_existing_file([
            os.path.join(CSV_DIR, "errors_naive_bayes_test.csv"),
            os.path.join(RESULTS_DIR, "errors_naive_bayes_test.csv"),
            os.path.join(PROJECT_ROOT, "errors_naive_bayes_test.csv"),
        ]),
        "Logistic Regression": find_existing_file([
            os.path.join(CSV_DIR, "errors_logistic_regression_test.csv"),
            os.path.join(RESULTS_DIR, "errors_logistic_regression_test.csv"),
            os.path.join(PROJECT_ROOT, "errors_logistic_regression_test.csv"),
        ]),
        "BERT": find_existing_file([
            os.path.join(CSV_DIR, "errors_bert_test.csv"),
            os.path.join(RESULTS_DIR, "errors_bert_test.csv"),
            os.path.join(PROJECT_ROOT, "errors_bert_test.csv"),
        ]),
    }


def load_error_counts() -> pd.DataFrame:
    error_paths = get_error_file_paths()

    records = []
    for model_name, path in error_paths.items():
        if path is None:
            continue

        df = pd.read_csv(path)
        records.append({
            "Model": model_name,
            "ErrorCount": len(df),
        })

    if not records:
        return pd.DataFrame()

    error_df = pd.DataFrame(records)

    # if every file has exactly 100 rows, they are probably sample errors only
    if (error_df["ErrorCount"] == 100).all():
        print("All error CSV files contain exactly 100 rows. These appear to be sample errors only.")
        print("Skipping error_count_comparison.png to avoid misleading totals.")
        return pd.DataFrame()

    model_order = [
        "Most Frequent Baseline",
        "Naive Bayes",
        "Logistic Regression",
        "BERT",
    ]
    error_df["Model"] = pd.Categorical(error_df["Model"], categories=model_order, ordered=True)
    error_df = error_df.sort_values("Model").reset_index(drop=True)

    return error_df


# =========================
# Plot helpers
# =========================
def add_value_labels(ax, values, fmt="{:.4f}", y_offset=0.01):
    for i, v in enumerate(values):
        ax.text(i, v + y_offset, fmt.format(v), ha="center", va="bottom", fontsize=9)


# =========================
# Plot functions
# =========================
def plot_metric_comparison(summary_df: pd.DataFrame):
    plot_df = summary_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]]

    ax = plot_df.plot(kind="bar", figsize=(11, 6))
    plt.title("Model Performance Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)
    plt.legend(title="Metric")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "model_metric_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_f1_comparison(summary_df: pd.DataFrame):
    plot_df = summary_df[["Model", "F1"]].sort_values("F1", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["Model"], plot_df["F1"])
    plt.title("F1 Score Comparison")
    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)
    add_value_labels(plt.gca(), plot_df["F1"].tolist(), fmt="{:.4f}", y_offset=0.01)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "f1_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_accuracy_comparison(summary_df: pd.DataFrame):
    plot_df = summary_df[["Model", "Accuracy"]].sort_values("Accuracy", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["Model"], plot_df["Accuracy"])
    plt.title("Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)
    add_value_labels(plt.gca(), plot_df["Accuracy"].tolist(), fmt="{:.4f}", y_offset=0.01)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "accuracy_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_error_count_comparison(error_df: pd.DataFrame):
    if error_df.empty:
        return

    plt.figure(figsize=(9, 5))
    plt.bar(error_df["Model"], error_df["ErrorCount"])
    plt.title("Misclassification Count Comparison")
    plt.xlabel("Model")
    plt.ylabel("Number of Misclassified Test Samples")
    plt.xticks(rotation=15)

    max_val = error_df["ErrorCount"].max()
    for i, v in enumerate(error_df["ErrorCount"]):
        plt.text(i, v + max_val * 0.01, str(v), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "error_count_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# =========================
# Main
# =========================
def main():
    summary_df = load_summary_results()

    print("\nSummary results:")
    print(summary_df)

    plot_metric_comparison(summary_df)
    plot_f1_comparison(summary_df)
    plot_accuracy_comparison(summary_df)

    error_df = load_error_counts()
    if not error_df.empty:
        print("\nError counts:")
        print(error_df)
        plot_error_count_comparison(error_df)
    else:
        print("\nNo full error-count plot generated.")

    print(f"\nPlots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()