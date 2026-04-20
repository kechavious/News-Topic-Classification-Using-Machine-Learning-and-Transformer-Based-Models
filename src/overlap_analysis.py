import os
import pandas as pd
from error_analysis import save_overlap_analysis, save_stratified_error_sample


def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def main():
    """
    Expected input files:
      results/csv/lr_test_predictions.csv
      results/csv/bert_test_predictions.csv

    Each file should contain:
      text, true_label_id, pred_label_id
    """

    lr_path = "results/csv/lr_test_predictions.csv"
    bert_path = "results/csv/bert_test_predictions.csv"

    if not os.path.exists(lr_path):
        raise FileNotFoundError(f"Missing file: {lr_path}")
    if not os.path.exists(bert_path):
        raise FileNotFoundError(f"Missing file: {bert_path}")

    lr_df = pd.read_csv(lr_path)
    bert_df = pd.read_csv(bert_path)

    required_cols = {"text", "true_label_id", "pred_label_id"}
    if not required_cols.issubset(lr_df.columns):
        raise ValueError(f"{lr_path} must contain columns: {required_cols}")
    if not required_cols.issubset(bert_df.columns):
        raise ValueError(f"{bert_path} must contain columns: {required_cols}")

    # align by text + gold label
    merged = lr_df.merge(
        bert_df,
        on=["text", "true_label_id"],
        suffixes=("_lr", "_bert")
    )

    save_overlap_analysis(
        texts=merged["text"].tolist(),
        y_true=merged["true_label_id"].tolist(),
        lr_pred=merged["pred_label_id_lr"].tolist(),
        bert_pred=merged["pred_label_id_bert"].tolist(),
        output_dir="results/csv/overlap_analysis",
    )

    # Stratified samples for the full error files
    save_stratified_error_sample(
        "results/csv/errors_logistic_regression_test.csv",
        "results/csv/errors_logistic_regression_test_sample100_stratified.csv",
        total_samples=100,
        random_seed=42,
        min_per_pair=5,
    )

    save_stratified_error_sample(
        "results/csv/errors_bert_test.csv",
        "results/csv/errors_bert_test_sample100_stratified.csv",
        total_samples=100,
        random_seed=42,
        min_per_pair=5,
    )


if __name__ == "__main__":
    main()