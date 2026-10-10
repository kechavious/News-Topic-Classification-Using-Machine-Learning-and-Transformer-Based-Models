"""Regression tests for pairing, statistical inference, sampling and local training."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from research_analysis import align_predictions, paired_statistics, load_predictions
from error_analysis import save_stratified_error_sample, save_error_analysis
from utils import prepare_output_dir
from predict import predict


def frame(predictions, ids=False):
    result = pd.DataFrame(
        {
            "text": ["same text"] * len(predictions),
            "true_label_id": [0] * len(predictions),
            "pred_label_id": predictions,
        }
    )
    if ids:
        result["example_id"] = list(range(len(predictions)))
    return result


class EvaluationTests(unittest.TestCase):
    def test_pairing_does_not_multiply_duplicate_texts(self):
        a, b = align_predictions(frame([0, 1]), frame([1, 0]))
        self.assertEqual(len(a), 2)
        self.assertEqual(paired_statistics(a, b)["overlap"]["both_wrong"], 0)

    def test_stable_ids_align_reordered_predictions(self):
        a = frame([0, 1], True)
        b = frame([1, 0], True).iloc[::-1]
        _, aligned = align_predictions(a, b)
        self.assertEqual(aligned.pred_label_id.tolist(), [1, 0])

    def test_misaligned_legacy_rows_fail(self):
        a = frame([0, 1])
        b = a.copy()
        b.loc[1, "text"] = "different"
        with self.assertRaises(ValueError):
            align_predictions(a, b)

    def test_mismatched_ids_fail(self):
        a = frame([0, 1], True)
        b = a.copy()
        b.loc[1, "example_id"] = 5
        with self.assertRaises(ValueError):
            align_predictions(a, b)

    def test_duplicate_ids_fail(self):
        a = frame([0, 1], True)
        a.example_id = 0
        with self.assertRaises(ValueError):
            align_predictions(a, a)

    def test_one_sided_ids_fail(self):
        with self.assertRaises(ValueError):
            align_predictions(frame([0], True), frame([0]))

    def test_unequal_lengths_fail(self):
        with self.assertRaises(ValueError):
            align_predictions(frame([0]), frame([0, 1]))

    def test_exact_statistics_and_bootstrap_repeatability(self):
        a = frame([0, 1, 1, 1])
        b = frame([0, 0, 0, 1])
        s = paired_statistics(a, b)
        self.assertEqual(s["accuracy_difference_bert_minus_lr"], 0.5)
        self.assertEqual(s["mcnemar_exact_two_sided_p"], 0.5)
        self.assertEqual(s, paired_statistics(a, b))

    def test_no_discordances(self):
        s = paired_statistics(frame([0, 1]), frame([0, 1]))
        self.assertEqual(s["mcnemar_exact_two_sided_p"], 1)
        self.assertEqual(s["paired_bootstrap_95_percent_ci"], [0, 0])

    def test_invalid_labels_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            frame([4]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_predictions(path)

    def test_sampling_exact_count_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = pd.DataFrame(
                {
                    "text": [str(i) for i in range(180)],
                    "true_label": ["World"] * 180,
                    "pred_label": ["Sports"] * 90 + ["Business"] * 90,
                }
            )
            data.to_csv(tmp / "errors.csv", index=False)
            for name in ["a", "b"]:
                save_stratified_error_sample(
                    str(tmp / "errors.csv"), str(tmp / f"{name}.csv")
                )
            a = pd.read_csv(tmp / "a.csv")
            b = pd.read_csv(tmp / "b.csv")
            self.assertEqual(len(a), 100)
            self.assertEqual(a.text.nunique(), 100)
            self.assertTrue(a.equals(b))
            self.assertEqual(a.pred_label.nunique(), 2)

    def test_empty_error_csv_has_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "errors.csv"
            save_error_analysis(["text"], [0], [0], str(path))
            self.assertTrue(pd.read_csv(path).empty)

    def test_existing_results_are_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "existing.txt").write_text("keep")
            with self.assertRaises(ValueError):
                prepare_output_dir(tmp)
            self.assertEqual(Path(tmp, "existing.txt").read_text(), "keep")

    def test_dataset_loader_uses_canonical_repository_and_stratifies(self):
        from datasets import ClassLabel, DatasetDict
        import utils

        train = Dataset.from_dict(
            {"text": [str(i) for i in range(40)], "label": [i % 4 for i in range(40)]}
        ).cast_column("label", ClassLabel(names=list(utils.LABEL_NAMES.values())))
        dataset = DatasetDict(train=train, test=train.select(range(8)))
        with patch.object(utils, "load_dataset", return_value=dataset) as loader:
            a, b, c = utils.load_ag_news_data(revision="fixed-commit")
        loader.assert_called_once_with("fancyzhx/ag_news", revision="fixed-commit")
        self.assertEqual((len(a), len(b), len(c)), (36, 4, 8))
        self.assertEqual(sorted(b["label"]), [0, 1, 2, 3])

    def test_published_predictions_reproduce_scores(self):
        root = Path(__file__).resolve().parents[1]
        s = paired_statistics(
            load_predictions(root / "results/csv/lr_test_predictions.csv"),
            load_predictions(root / "results/csv/bert_test_predictions.csv"),
        )
        self.assertEqual(s["n_examples"], 7600)
        self.assertEqual(
            s["overlap"],
            dict(
                both_correct=6862,
                lr_wrong_bert_correct=301,
                bert_wrong_lr_correct=115,
                both_wrong=322,
            ),
        )
        self.assertAlmostEqual(s["metrics"]["bert"]["accuracy"], 0.9425)


class TrainingTests(unittest.TestCase):
    def splits(self):
        topics = [
            "diplomat world treaty",
            "sports football tournament",
            "business stock earnings",
            "technology computer software",
        ]

        def data(n):
            return Dataset.from_dict(
                {
                    "text": [topics[i % 4] + f" story {i}" for i in range(n)],
                    "label": [i % 4 for i in range(n)],
                }
            )

        return data(32), data(8), data(8)

    def test_baseline_cli_saves_reloadable_models_and_ids(self):
        import baseline_models

        previous = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with (
                    patch.object(
                        baseline_models, "load_ag_news_data", return_value=self.splits()
                    ),
                    patch.object(sys, "argv", ["baseline_models", "--output-dir", tmp]),
                ):
                    baseline_models.main()
                result = predict(
                    Path(tmp) / "models/logistic_regression.joblib",
                    ["sports football tournament"],
                )
                self.assertEqual(result[0]["label"], "Sports")
                self.assertEqual(
                    pd.read_csv(
                        Path(tmp) / "results/csv/lr_test_predictions.csv"
                    ).example_id.tolist(),
                    list(range(8)),
                )
                config = json.loads(
                    Path(tmp, "results/csv/baseline_experiment_config.json").read_text()
                )
                self.assertEqual(config["dataset"]["train_size"], 32)
        finally:
            os.chdir(previous)

    def test_bert_cli_trains_saves_and_predicts_without_network(self):
        import bert_model
        from transformers import (
            BertConfig,
            BertForSequenceClassification,
            BertTokenizer,
        )

        previous = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                vocab = root / "vocab.txt"
                vocab.write_text(
                    "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nsports\nfootball\nworld\nbusiness\ntechnology\n"
                )
                tokenizer = BertTokenizer(vocab_file=str(vocab))
                model = BertForSequenceClassification(
                    BertConfig(
                        vocab_size=10,
                        hidden_size=16,
                        num_hidden_layers=1,
                        num_attention_heads=2,
                        intermediate_size=32,
                        num_labels=4,
                    )
                )
                with (
                    patch.object(
                        bert_model, "load_ag_news_data", return_value=self.splits()
                    ),
                    patch.object(
                        bert_model.AutoTokenizer,
                        "from_pretrained",
                        return_value=tokenizer,
                    ),
                    patch.object(
                        bert_model.AutoModelForSequenceClassification,
                        "from_pretrained",
                        return_value=model,
                    ),
                    patch.object(
                        sys,
                        "argv",
                        [
                            "bert_model",
                            "--output-dir",
                            str(root / "run"),
                            "--epochs",
                            "1",
                            "--max-length",
                            "16",
                        ],
                    ),
                ):
                    bert_model.main()
                result = predict(root / "run/models/bert", ["sports football"])
                self.assertEqual(len(result), 1)
                self.assertAlmostEqual(
                    sum(result[0]["probabilities"].values()), 1, places=5
                )
                self.assertTrue((root / "run/models/bert/model.safetensors").exists())
                from transformers import AutoTokenizer

                saved_tokenizer = AutoTokenizer.from_pretrained(
                    root / "run/models/bert", local_files_only=True
                )
                self.assertEqual(saved_tokenizer.model_max_length, 16)
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
