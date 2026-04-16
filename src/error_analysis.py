import pandas as pd
from sklearn.metrics import confusion_matrix
from utils import LABEL_NAMES


def save_error_analysis(texts, y_true, y_pred, output_file: str, max_examples: int | None = None):
    """
    Save misclassified examples for manual error analysis.

    Args:
        texts: original input texts
        y_true: gold labels
        y_pred: predicted labels
        output_file: path to save csv
        max_examples: if None, save all errors; otherwise save only first N
    """
    errors = []
    for text, true_label, pred_label in zip(texts, y_true, y_pred):
        if true_label != pred_label:
            errors.append(
                {
                    "text": text,
                    "true_label": LABEL_NAMES[true_label],
                    "pred_label": LABEL_NAMES[pred_label],
                }
            )

    if max_examples is not None:
        errors = errors[:max_examples]

    df = pd.DataFrame(errors)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} misclassified examples to {output_file}")


def save_confusion_matrix_csv(y_true, y_pred, output_file: str):
    cm = confusion_matrix(y_true, y_pred)
    labels = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
    df = pd.DataFrame(cm, index=labels, columns=labels)
    df.to_csv(output_file, encoding="utf-8-sig")
    print(f"Saved confusion matrix to {output_file}")