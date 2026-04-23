import os
import pandas as pd
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
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
    get_package_versions,
    save_experiment_config,
)
from error_analysis import (
    save_error_analysis,
    save_confusion_matrix_csv,
    save_confusion_matrix_plot,
)


class MostFrequentBaseline:
    def __init__(self):
        self.most_common_label = None

    def fit(self, y_train):
        counter = Counter(y_train)
        self.most_common_label = counter.most_common(1)[0][0]

    def predict(self, n: int):
        return [self.most_common_label] * n


def evaluate_model(model_name: str, y_true, y_pred):
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

    dev_size = 0.1
    train_set, dev_set, test_set = load_ag_news_data(dev_size=dev_size)

    X_train, y_train = extract_text_and_labels(train_set)
    X_dev, y_dev = extract_text_and_labels(dev_set)
    X_test, y_test = extract_text_and_labels(test_set)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
    )

    save_experiment_config(
        "results/csv/baseline_experiment_config.json",
        {
            "script": "src/baseline_models.py",
            "seed": SEED,
            "dataset": {
                "name": "ag_news",
                "train_size": len(train_set),
                "dev_size": len(dev_set),
                "test_size": len(test_set),
                "dev_split_fraction": dev_size,
                "dev_split_strategy": "stratified by label",
            },
            "models": [
                "Most Frequent Baseline",
                "Naive Bayes",
                "Logistic Regression",
            ],
            "feature_extraction": {
                "type": "tfidf",
                "lowercase": True,
                "stop_words": "english",
                "max_features": 20000,
                "ngram_range": [1, 2],
            },
            "model_hyperparameters": {
                "naive_bayes": {},
                "logistic_regression": {
                    "max_iter": 2000,
                    "random_state": SEED,
                },
            },
            "package_versions": get_package_versions(
                ["numpy", "pandas", "scikit-learn", "datasets", "matplotlib"]
            ),
        },
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_dev_tfidf = vectorizer.transform(X_dev)
    X_test_tfidf = vectorizer.transform(X_test)

    results = []

    # 1. Most Frequent Baseline
    baseline = MostFrequentBaseline()
    baseline.fit(y_train)

    dev_pred_baseline = baseline.predict(len(y_dev))
    test_pred_baseline = baseline.predict(len(y_test))

    results.append(evaluate_model("Most Frequent Baseline (Dev)", y_dev, dev_pred_baseline))
    results.append(evaluate_model("Most Frequent Baseline (Test)", y_test, test_pred_baseline))

    save_error_analysis(
        X_test,
        y_test,
        test_pred_baseline,
        "results/csv/errors_most_frequent_test.csv",
        max_examples=None,
    )
    save_confusion_matrix_csv(
        y_test,
        test_pred_baseline,
        "results/csv/confusion_most_frequent_test.csv",
    )
    save_confusion_matrix_plot(
        y_test,
        test_pred_baseline,
        "results/plots/confusion_most_frequent_test.png",
        title="Most Frequent Baseline Confusion Matrix (Test Set)",
    )

    # 2. Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)

    dev_pred_nb = nb_model.predict(X_dev_tfidf)
    test_pred_nb = nb_model.predict(X_test_tfidf)

    results.append(evaluate_model("Naive Bayes (Dev)", y_dev, dev_pred_nb))
    results.append(evaluate_model("Naive Bayes (Test)", y_test, test_pred_nb))

    save_error_analysis(
        X_test,
        y_test,
        test_pred_nb,
        "results/csv/errors_naive_bayes_test.csv",
        max_examples=None,
    )
    save_confusion_matrix_csv(
        y_test,
        test_pred_nb,
        "results/csv/confusion_naive_bayes_test.csv",
    )
    save_confusion_matrix_plot(
        y_test,
        test_pred_nb,
        "results/plots/confusion_naive_bayes_test.png",
        title="Naive Bayes Confusion Matrix (Test Set)",
    )

    # 3. Logistic Regression
    lr_model = LogisticRegression(
        max_iter=2000,
        random_state=SEED,
    )
    lr_model.fit(X_train_tfidf, y_train)

    dev_pred_lr = lr_model.predict(X_dev_tfidf)
    test_pred_lr = lr_model.predict(X_test_tfidf)

    results.append(evaluate_model("Logistic Regression (Dev)", y_dev, dev_pred_lr))
    results.append(evaluate_model("Logistic Regression (Test)", y_test, test_pred_lr))

    save_error_analysis(
        X_test,
        y_test,
        test_pred_lr,
        "results/csv/errors_logistic_regression_test.csv",
        max_examples=None,
    )
    save_confusion_matrix_csv(
        y_test,
        test_pred_lr,
        "results/csv/confusion_logistic_regression_test.csv",
    )
    save_confusion_matrix_plot(
        y_test,
        test_pred_lr,
        "results/plots/confusion_logistic_regression_test.png",
        title="Logistic Regression Confusion Matrix (Test Set)",
    )

    # Save LR test predictions for overlap analysis
    lr_pred_df = pd.DataFrame({
        "text": X_test,
        "true_label_id": y_test,
        "pred_label_id": test_pred_lr,
    })
    lr_pred_df.to_csv(
        "results/csv/lr_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("Saved results/csv/lr_test_predictions.csv")

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        "results/csv/model_results_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSaved model summary to results/csv/model_results_summary.csv")
    print(results_df)


if __name__ == "__main__":
    main()
