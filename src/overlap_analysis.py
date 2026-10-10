"""Compare saved predictions with strict example alignment."""

import argparse
from pathlib import Path

from error_analysis import save_overlap_analysis, save_stratified_error_sample
from research_analysis import load_predictions, align_predictions
from utils import prepare_output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr", default="results/csv/lr_test_predictions.csv")
    parser.add_argument("--bert", default="results/csv/bert_test_predictions.csv")
    parser.add_argument("--error-dir", default="results/csv")
    parser.add_argument("--output-dir", default="runs/overlap-analysis")
    args = parser.parse_args()
    lr, bert = align_predictions(load_predictions(args.lr), load_predictions(args.bert))
    output = Path(prepare_output_dir(args.output_dir))
    save_overlap_analysis(
        lr.text.tolist(),
        lr.true_label_id.tolist(),
        lr.pred_label_id.tolist(),
        bert.pred_label_id.tolist(),
        str(output),
    )
    for name in ("logistic_regression", "bert"):
        error_path = Path(args.error_dir) / f"errors_{name}_test.csv"
        if error_path.exists():
            save_stratified_error_sample(
                str(error_path), str(output / f"errors_{name}_sample100.csv")
            )


if __name__ == "__main__":
    main()
