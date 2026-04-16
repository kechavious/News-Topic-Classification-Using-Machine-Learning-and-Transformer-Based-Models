import os
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_CSV_DIR = "results/csv"
PLOTS_DIR = "results/plots"


def ensure_plot_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def load_summary_results():
    baseline_path = os.path.join(RESULTS_CSV_DIR, "model_results_summary.csv")
    bert_path = os.path.join(RESULTS_CSV_DIR, "bert_results_summary.csv")

    baseline_df = pd.read_csv(baseline_path)
    bert_df = pd.read_csv(bert_path)

    return baseline_df, bert_df


def prepare_accuracy_dataframe():
    baseline_df, bert_df = load_summary_results()

    # 只保留 Test 結果
    baseline_test = baseline_df[baseline_df["Model"].str.contains("Test", case=False)].copy()
    bert_test = bert_df[bert_df["Model"].str.contains("Test", case=False)].copy()

    # 統一名稱
    baseline_test["Model"] = baseline_test["Model"].replace({
        "Most Frequent Baseline (Test)": "Most Frequent Baseline",
        "Naive Bayes (Test)": "Naive Bayes",
        "Logistic Regression (Test)": "Logistic Regression",
    })

    bert_test["Model"] = bert_test["Model"].replace({
        "BERT (Test)": "BERT"
    })

    combined = pd.concat([baseline_test, bert_test], ignore_index=True)
    combined = combined[["Model", "Accuracy", "Precision", "Recall", "F1"]]

    return combined


def plot_accuracy_comparison():
    df = prepare_accuracy_dataframe()

    plt.figure(figsize=(10, 6))
    plt.bar(df["Model"], df["Accuracy"])
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15)
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, "accuracy_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_metric_comparison():
    df = prepare_accuracy_dataframe()

    metrics = ["Accuracy", "Precision", "Recall", "F1"]

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.bar(df["Model"], df[metric])
        plt.title(f"{metric} Comparison")
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.ylim(0, 1.0)
        plt.xticks(rotation=15)
        plt.tight_layout()

        output_path = os.path.join(PLOTS_DIR, f"{metric.lower()}_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {output_path}")


def plot_confusion_matrix_from_csv(csv_filename, output_filename, title):
    csv_path = os.path.join(RESULTS_CSV_DIR, csv_filename)
    df = pd.read_csv(csv_path, index_col=0)

    plt.figure(figsize=(8, 6))
    plt.imshow(df.values, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    plt.xticks(range(len(df.columns)), df.columns, rotation=45)
    plt.yticks(range(len(df.index)), df.index)

    # 在格子中標數字
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            plt.text(j, i, str(df.iloc[i, j]), ha="center", va="center")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def main():
    ensure_plot_dir()

    plot_accuracy_comparison()
    plot_metric_comparison()

    plot_confusion_matrix_from_csv(
        csv_filename="confusion_logistic_regression_test.csv",
        output_filename="confusion_matrix_lr.png",
        title="Confusion Matrix - Logistic Regression",
    )

    plot_confusion_matrix_from_csv(
        csv_filename="confusion_bert_test.csv",
        output_filename="confusion_matrix_bert.png",
        title="Confusion Matrix - BERT",
    )

    plot_confusion_matrix_from_csv(
        csv_filename="confusion_naive_bayes_test.csv",
        output_filename="confusion_matrix_nb.png",
        title="Confusion Matrix - Naive Bayes",
    )

    plot_confusion_matrix_from_csv(
        csv_filename="confusion_most_frequent_test.csv",
        output_filename="confusion_matrix_baseline.png",
        title="Confusion Matrix - Most Frequent Baseline",
    )


if __name__ == "__main__":
    main()