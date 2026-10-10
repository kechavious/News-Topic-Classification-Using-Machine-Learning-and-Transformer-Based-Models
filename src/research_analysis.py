"""Auditable paired evaluation of saved predictions; no model downloads required."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utils import LABEL_NAMES, prepare_output_dir


def load_predictions(path):
    frame = pd.read_csv(path)
    required = {"text", "true_label_id", "pred_label_id"}
    if not required.issubset(frame):
        raise ValueError(f"{path}: missing columns {sorted(required - set(frame))}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError(
            f"{path}: predictions must be nonempty and contain no missing values"
        )
    if not frame.text.map(lambda text: isinstance(text, str)).all():
        raise ValueError(f"{path}: texts must be strings")
    for col in ("true_label_id", "pred_label_id"):
        if not frame[col].isin(LABEL_NAMES).all():
            raise ValueError(f"{path}: {col} must contain label IDs 0–3")
    if "example_id" in frame and (
        frame.example_id.isna().any() or frame.example_id.duplicated().any()
    ):
        raise ValueError(f"{path}: example IDs must be non-null and unique")
    return frame


def align_predictions(left, right):
    """Align by stable IDs, or require identical legacy row order. Never join on text."""
    if len(left) != len(right):
        raise ValueError("Prediction files contain different numbers of examples")
    has_ids = ["example_id" in frame for frame in (left, right)]
    if any(has_ids) and not all(has_ids):
        raise ValueError("Both files must have example_id, or neither file may have it")
    if all(has_ids):
        if any(
            frame.example_id.isna().any() or frame.example_id.duplicated().any()
            for frame in (left, right)
        ):
            raise ValueError("Example IDs must be non-null and unique")
        if set(left.example_id) != set(right.example_id):
            raise ValueError("Prediction files contain different example IDs")
        right = right.set_index("example_id").loc[left.example_id].reset_index()
    left, right = left.reset_index(drop=True), right.reset_index(drop=True)
    if not left[["text", "true_label_id"]].equals(right[["text", "true_label_id"]]):
        raise ValueError(
            "Texts/gold labels do not align; legacy files must have identical row order"
        )
    return left, right


def paired_statistics(left, right, seed=42, resamples=10000):
    left, right = align_predictions(left, right)
    if len(left) == 0 or resamples < 1:
        raise ValueError("Nonempty predictions and positive resamples are required")
    gold = left.true_label_id.to_numpy()
    a = left.pred_label_id.to_numpy() == gold
    b = right.pred_label_id.to_numpy() == gold
    counts = np.array([(a & b).sum(), (~a & b).sum(), (a & ~b).sum(), (~a & ~b).sum()])
    # For accuracy differences, multinomial draws over the four paired cases
    # are exactly equivalent to resampling paired example indices.
    draws = np.random.default_rng(seed).multinomial(
        len(a), counts / len(a), size=resamples
    )
    differences = (draws[:, 1] - draws[:, 2]) / len(a)
    discordant = int(counts[1] + counts[2])
    metrics = {}
    for name, frame in [("logistic_regression", left), ("bert", right)]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            gold,
            frame.pred_label_id,
            labels=list(LABEL_NAMES),
            average="macro",
            zero_division=0,
        )
        metrics[name] = dict(
            accuracy=float(accuracy_score(gold, frame.pred_label_id)),
            precision=float(precision),
            recall=float(recall),
            macro_f1=float(f1),
        )
    return {
        "n_examples": len(a),
        "metrics": metrics,
        "overlap": dict(
            zip(
                [
                    "both_correct",
                    "lr_wrong_bert_correct",
                    "bert_wrong_lr_correct",
                    "both_wrong",
                ],
                map(int, counts),
            )
        ),
        "accuracy_difference_bert_minus_lr": float((counts[1] - counts[2]) / len(a)),
        "paired_bootstrap_95_percent_ci": np.quantile(
            differences, [0.025, 0.975]
        ).tolist(),
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "mcnemar_exact_two_sided_p": float(
            binomtest(int(counts[1]), discordant, 0.5).pvalue
        )
        if discordant
        else 1.0,
        "inference_scope": "Conditional on these saved models and test examples; does not measure training-seed variance or establish a causal mechanism.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr", default="results/csv/lr_test_predictions.csv")
    parser.add_argument("--bert", default="results/csv/bert_test_predictions.csv")
    parser.add_argument("--output-dir", default="runs/research-analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()
    a, b = align_predictions(load_predictions(args.lr), load_predictions(args.bert))
    report = paired_statistics(a, b, args.seed, args.resamples)
    report["input_sha256"] = {
        name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name, path in [("lr", args.lr), ("bert", args.bert)]
    }
    output = Path(prepare_output_dir(args.output_dir))
    (output / "paired_statistics.json").write_text(json.dumps(report, indent=2) + "\n")
    gold = a.true_label_id.to_numpy()
    x = a.pred_label_id.to_numpy() == gold
    y = b.pred_label_id.to_numpy() == gold
    rows = []
    for name, mask in [
        ("all", np.ones(len(a), dtype=bool)),
        ("bert_correct", y),
        ("bert_wrong", ~y),
        ("lr_wrong_bert_correct", ~x & y),
    ]:
        lengths = a.loc[mask, "text"].str.split().str.len()
        rows.append(
            {
                "partition": name,
                "n": len(lengths),
                "mean_words": float(lengths.mean()) if len(lengths) else None,
                "median_words": float(lengths.median()) if len(lengths) else None,
            }
        )
    pd.DataFrame(rows).to_csv(output / "length_summary.csv", index=False)
    rows = []
    for label, name in LABEL_NAMES.items():
        mask = gold == label
        rows.append(
            {
                "label": name,
                "n": int(mask.sum()),
                "lr_correct": int(x[mask].sum()),
                "bert_correct": int(y[mask].sum()),
                "net_gain": int(y[mask].sum() - x[mask].sum()),
            }
        )
    pd.DataFrame(rows).to_csv(output / "per_class_gain.csv", index=False)
    # Directional confusion reductions distinguish gross corrections from net gains.
    rows = []
    for source in LABEL_NAMES:
        for target in LABEL_NAMES:
            if source == target:
                continue
            n_lr = int(
                ((gold == source) & (a.pred_label_id.to_numpy() == target)).sum()
            )
            n_bert = int(
                ((gold == source) & (b.pred_label_id.to_numpy() == target)).sum()
            )
            rows.append(
                {
                    "gold": LABEL_NAMES[source],
                    "predicted": LABEL_NAMES[target],
                    "lr_errors": n_lr,
                    "bert_errors": n_bert,
                    "net_reduction": n_lr - n_bert,
                }
            )
    pd.DataFrame(rows).to_csv(output / "confusion_reductions.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
