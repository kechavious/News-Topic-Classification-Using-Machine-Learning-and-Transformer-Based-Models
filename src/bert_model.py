import os
import numpy as np
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from utils import (
    SEED,
    LABEL_NAMES,
    ensure_directories,
    load_ag_news_data,
    extract_text_and_labels,
)
from error_analysis import (
    save_error_analysis,
    save_confusion_matrix_csv,
    save_confusion_matrix_plot,
)

MODEL_NAME = "bert-base-uncased"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
    )


def evaluate_predictions(model_name: str, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print(f"{model_name}")
    print("=" * 60)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[LABEL_NAMES[i] for i in range(len(LABEL_NAMES))],
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def main():
    ensure_directories()
    os.makedirs("results/csv", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    train_set, dev_set, test_set = load_ag_news_data(dev_size=0.1)

    X_train, y_train = extract_text_and_labels(train_set)
    X_dev, y_dev = extract_text_and_labels(dev_set)
    X_test, y_test = extract_text_and_labels(test_set)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = train_set.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    dev_dataset = dev_set.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset = test_set.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    train_dataset = train_dataset.rename_column("label", "labels")
    dev_dataset = dev_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    keep_cols = ["input_ids", "attention_mask", "labels"]

    train_dataset = train_dataset.remove_columns(
        [col for col in train_dataset.column_names if col not in keep_cols]
    )
    dev_dataset = dev_dataset.remove_columns(
        [col for col in dev_dataset.column_names if col not in keep_cols]
    )
    test_dataset = test_dataset.remove_columns(
        [col for col in test_dataset.column_names if col not in keep_cols]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="results/bert_checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nTraining BERT...\n")
    trainer.train()

    dev_predictions = trainer.predict(dev_dataset)
    dev_pred_labels = np.argmax(dev_predictions.predictions, axis=1)

    test_predictions = trainer.predict(test_dataset)
    test_pred_labels = np.argmax(test_predictions.predictions, axis=1)

    results = []
    results.append(evaluate_predictions("BERT (Dev)", y_dev, dev_pred_labels))
    results.append(evaluate_predictions("BERT (Test)", y_test, test_pred_labels))

    # Save full BERT error analysis
    save_error_analysis(
        X_test,
        y_test,
        test_pred_labels,
        "results/csv/errors_bert_test.csv",
        max_examples=None,
    )

    # Save confusion matrix CSV + plot
    save_confusion_matrix_csv(
        y_test,
        test_pred_labels,
        "results/csv/confusion_bert_test.csv",
    )
    save_confusion_matrix_plot(
        y_test,
        test_pred_labels,
        "results/plots/confusion_bert_test.png",
        title="BERT Confusion Matrix (Test Set)",
    )

    # Save BERT test predictions for overlap analysis
    bert_pred_df = pd.DataFrame({
        "text": X_test,
        "true_label_id": y_test,
        "pred_label_id": test_pred_labels,
    })
    bert_pred_df.to_csv(
        "results/csv/bert_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("Saved results/csv/bert_test_predictions.csv")

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        "results/csv/bert_results_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSaved BERT summary to results/csv/bert_results_summary.csv")
    print(results_df)


if __name__ == "__main__":
    main()