import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils import LABEL_NAMES


def save_error_analysis(texts, y_true, y_pred, output_file: str, max_examples=None):
    """
    Save misclassified examples.

    Args:
        texts: iterable of input texts
        y_true: iterable of gold labels
        y_pred: iterable of predicted labels
        output_file: output csv path
        max_examples: if None, save all errors; otherwise save first N
    """
    errors = []
    for text, true_label, pred_label in zip(texts, y_true, y_pred):
        if true_label != pred_label:
            errors.append({
                "text": text,
                "true_label": LABEL_NAMES[true_label],
                "pred_label": LABEL_NAMES[pred_label],
                "true_id": true_label,
                "pred_id": pred_label,
                "confusion_pair": f"{LABEL_NAMES[true_label]} -> {LABEL_NAMES[pred_label]}",
            })

    if max_examples is not None:
        errors = errors[:max_examples]

    df = pd.DataFrame(errors)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} misclassified examples to {output_file}")


def save_confusion_matrix_csv(y_true, y_pred, output_file: str):
    """
    Save confusion matrix as CSV.
    """
    labels = list(range(len(LABEL_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    class_names = [LABEL_NAMES[i] for i in labels]

    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, encoding="utf-8-sig")
    print(f"Saved confusion matrix CSV to {output_file}")


def save_confusion_matrix_plot(y_true, y_pred, output_file: str, title: str):
    """
    Save confusion matrix heatmap as PNG.
    """
    labels = list(range(len(LABEL_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    class_names = [LABEL_NAMES[i] for i in labels]

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    plt.title(title)
    plt.xticks(rotation=20)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix plot to {output_file}")


def save_stratified_error_sample(
    error_csv_path: str,
    output_file: str,
    total_samples: int = 100,
    random_seed: int = 42,
    min_per_pair: int = 5,
):
    """
    Create a stratified sample of errors by confusion pair.

    This avoids taking just the first 100 rows.
    """
    df = pd.read_csv(error_csv_path)

    if df.empty:
        print(f"No errors found in {error_csv_path}; skipping stratified sample.")
        return

    if "confusion_pair" not in df.columns:
        df["confusion_pair"] = df["true_label"].astype(str) + " -> " + df["pred_label"].astype(str)

    rng = random.Random(random_seed)

    pair_counts = df["confusion_pair"].value_counts()
    pairs = pair_counts.index.tolist()

    sampled_parts = []

    # first pass: ensure each pair gets a small representative sample
    for pair in pairs:
        group = df[df["confusion_pair"] == pair]
        n_take = min(len(group), min_per_pair)
        sampled_parts.append(group.sample(n=n_take, random_state=random_seed))

    sampled_df = pd.concat(sampled_parts, ignore_index=True).drop_duplicates()

    remaining = total_samples - len(sampled_df)
    if remaining > 0:
        remaining_df = df.drop(sampled_df.index, errors="ignore")
        if not remaining_df.empty:
            weights = remaining_df["confusion_pair"].map(pair_counts).astype(float)
            extra_n = min(remaining, len(remaining_df))
            extra_df = remaining_df.sample(
                n=extra_n,
                weights=weights,
                random_state=random_seed
            )
            sampled_df = pd.concat([sampled_df, extra_df], ignore_index=True).drop_duplicates()

    # if too many due to min_per_pair, trim randomly
    if len(sampled_df) > total_samples:
        sampled_df = sampled_df.sample(n=total_samples, random_state=random_seed)

    sampled_df = sampled_df.sort_values(["true_label", "pred_label"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    sampled_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Saved stratified sample ({len(sampled_df)} rows) to {output_file}")


def save_overlap_analysis(
    texts,
    y_true,
    lr_pred,
    bert_pred,
    output_dir: str,
):
    """
    Compare Logistic Regression and BERT errors on the same test set.
    Saves:
      - overlap_summary.csv
      - lr_wrong_bert_correct.csv
      - bert_wrong_lr_correct.csv
      - both_wrong.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    records = []

    for text, gold, lr, bert in zip(texts, y_true, lr_pred, bert_pred):
        lr_wrong = lr != gold
        bert_wrong = bert != gold

        if lr_wrong and not bert_wrong:
            case = "LR wrong / BERT correct"
        elif bert_wrong and not lr_wrong:
            case = "BERT wrong / LR correct"
        elif lr_wrong and bert_wrong:
            case = "Both wrong"
        else:
            case = "Both correct"

        records.append({
            "text": text,
            "true_label": LABEL_NAMES[gold],
            "lr_pred": LABEL_NAMES[lr],
            "bert_pred": LABEL_NAMES[bert],
            "case": case,
            "lr_confusion_pair": f"{LABEL_NAMES[gold]} -> {LABEL_NAMES[lr]}" if lr_wrong else "",
            "bert_confusion_pair": f"{LABEL_NAMES[gold]} -> {LABEL_NAMES[bert]}" if bert_wrong else "",
        })

    df = pd.DataFrame(records)

    summary = df["case"].value_counts().rename_axis("case").reset_index(name="count")
    summary.to_csv(os.path.join(output_dir, "overlap_summary.csv"), index=False, encoding="utf-8-sig")

    df[df["case"] == "LR wrong / BERT correct"].to_csv(
        os.path.join(output_dir, "lr_wrong_bert_correct.csv"), index=False, encoding="utf-8-sig"
    )
    df[df["case"] == "BERT wrong / LR correct"].to_csv(
        os.path.join(output_dir, "bert_wrong_lr_correct.csv"), index=False, encoding="utf-8-sig"
    )
    df[df["case"] == "Both wrong"].to_csv(
        os.path.join(output_dir, "both_wrong.csv"), index=False, encoding="utf-8-sig"
    )

    print(f"Saved overlap analysis files to {output_dir}")