import matplotlib.pyplot as plt
import os
import pandas as pd


# =========================
# Path Settings
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 如果 plot.py 放在 src/ 裡面，就回到上一層找 results/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..")) if os.path.basename(BASE_DIR) == "src" else BASE_DIR
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)


# =========================
# Helper Functions
# =========================
def normalize_columns(df):
    """
    Normalize column names to lowercase with underscores.
    """
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


def find_metric_column(df, candidates):
    """
    Find the first matching column from a list of possible names.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def standardize_summary_dataframe(df, default_models=None):
    """
    Convert different summary csv formats into a standard dataframe:
    columns = [model, accuracy, precision, recall, f1]
    """
    df = normalize_columns(df)

    # possible names for model column
    model_col = find_metric_column(df, ["model", "model_name", "classifier"])
    accuracy_col = find_metric_column(df, ["accuracy", "acc"])
    precision_col = find_metric_column(df, ["precision", "macro_precision", "weighted_precision"])
    recall_col = find_metric_column(df, ["recall", "macro_recall", "weighted_recall"])
    f1_col = find_metric_column(df, ["f1", "f1_score", "macro_f1", "weighted_f1"])

    # case 1: already in row format
    if model_col is not None:
        out = pd.DataFrame()
        out["model"] = df[model_col]
        out["accuracy"] = df[accuracy_col] if accuracy_col else None
        out["precision"] = df[precision_col] if precision_col else None
        out["recall"] = df[recall_col] if recall_col else None
        out["f1"] = df[f1_col] if f1_col else None
        return out

    # case 2: models are columns, metrics are rows
    possible_metric_col = find_metric_column(df, ["metric", "metrics"])
    if possible_metric_col is not None:
        df = df.set_index(possible_metric_col)
        df.index = df.index.str.lower().str.strip()

        records = []
        for col in df.columns:
            records.append({
                "model": col,
                "accuracy": df.loc["accuracy", col] if "accuracy" in df.index else None,
                "precision": df.loc["precision", col] if "precision" in df.index else None,
                "recall": df.loc["recall", col] if "recall" in df.index else None,
                "f1": df.loc["f1", col] if "f1" in df.index else (
                    df.loc["f1_score", col] if "f1_score" in df.index else None
                )
            })
        return pd.DataFrame(records)

    # case 3: one row csv without model column (assign default model)
    if default_models is not None and len(df) == len(default_models):
        out = pd.DataFrame()
        out["model"] = default_models
        out["accuracy"] = df[accuracy_col] if accuracy_col else None
        out["precision"] = df[precision_col] if precision_col else None
        out["recall"] = df[recall_col] if recall_col else None
        out["f1"] = df[f1_col] if f1_col else None
        return out

    raise ValueError("Could not identify summary CSV format.")


def load_summary_results():
    """
    Load baseline summary and BERT summary, then merge.
    """
    baseline_path = os.path.join(RESULTS_DIR, "model_results_summary.csv")
    bert_path = os.path.join(RESULTS_DIR, "bert_results_summary.csv")

    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Missing file: {baseline_path}")
    if not os.path.exists(bert_path):
        raise FileNotFoundError(f"Missing file: {bert_path}")

    baseline_df = pd.read_csv(baseline_path)
    bert_df = pd.read_csv(bert_path)

    baseline_df = standardize_summary_dataframe(baseline_df)
    bert_df = standardize_summary_dataframe(bert_df)

    combined_df = pd.concat([baseline_df, bert_df], ignore_index=True)

    # normalize model names
    combined_df["model"] = combined_df["model"].astype(str).str.strip()

    # try to make model names prettier
    model_rename = {
        "most_frequent": "Most Frequent",
        "naive_bayes": "Naive Bayes",
        "logistic_regression": "Logistic Regression",
        "bert": "BERT",
        "bert_model": "BERT"
    }
    combined_df["model"] = combined_df["model"].replace(model_rename)

    # convert metrics to numeric
    for col in ["accuracy", "precision", "recall", "f1"]:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

    # drop empty rows
    combined_df = combined_df.dropna(subset=["model"], how="any")

    return combined_df


def count_errors_from_file(file_path):
    """
    Count number of prediction errors in an errors csv.
    """
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    return len(df)


def load_error_counts():
    """
    Load error counts from error csv files.
    """
    error_files = {
        "Most Frequent": "errors_most_frequent_test.csv",
        "Naive Bayes": "errors_naive_bayes_test.csv",
        "Logistic Regression": "errors_logistic_regression_test.csv",
        "BERT": "errors_bert_test.csv",
    }

    records = []
    for model_name, filename in error_files.items():
        path = os.path.join(RESULTS_DIR, filename)
        count = count_errors_from_file(path)
        if count is not None:
            records.append({"model": model_name, "error_count": count})

    if not records:
        return pd.DataFrame(columns=["model", "error_count"])

    return pd.DataFrame(records)


# =========================
# Plot Functions
# =========================
def plot_metric_comparison(summary_df):
    """
    Grouped bar chart for Accuracy / Precision / Recall / F1 across models.
    """
    metrics = ["accuracy", "precision", "recall", "f1"]
    available_metrics = [m for m in metrics if m in summary_df.columns and summary_df[m].notna().any()]

    if not available_metrics:
        print("No usable metrics found for comparison plot.")
        return

    plot_df = summary_df.set_index("model")[available_metrics]

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


def plot_f1_comparison(summary_df):
    """
    Simple bar chart for F1 only, useful for presentation.
    """
    if "f1" not in summary_df.columns or summary_df["f1"].isna().all():
        print("No F1 column found.")
        return

    plot_df = summary_df[["model", "f1"]].dropna().sort_values(by="f1", ascending=False)

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["model"], plot_df["f1"])
    plt.title("F1 Score Comparison")
    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)

    for i, value in enumerate(plot_df["f1"]):
        plt.text(i, value + 0.01, f"{value:.3f}", ha="center")

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "f1_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")


def plot_error_count_comparison(error_df):
    """
    Compare total number of misclassified samples for each model.
    """
    if error_df.empty:
        print("No error files found or no error data available.")
        return

    error_df = error_df.sort_values(by="error_count", ascending=False)

    plt.figure(figsize=(9, 5))
    plt.bar(error_df["model"], error_df["error_count"])
    plt.title("Misclassification Count by Model")
    plt.xlabel("Model")
    plt.ylabel("Number of Errors")
    plt.xticks(rotation=15)

    for i, value in enumerate(error_df["error_count"]):
        plt.text(i, value + max(error_df["error_count"]) * 0.01, str(value), ha="center")

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "error_count_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")


# =========================
# Main
# =========================
def main():
    print("Loading summary results...")
    summary_df = load_summary_results()
    print(summary_df)

    print("\nGenerating plots...")
    plot_metric_comparison(summary_df)
    plot_f1_comparison(summary_df)

    error_df = load_error_counts()
    if not error_df.empty:
        print("\nError counts:")
        print(error_df)
    plot_error_count_comparison(error_df)

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()