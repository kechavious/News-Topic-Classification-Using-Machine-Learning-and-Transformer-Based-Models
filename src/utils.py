import os
import random
import json
import platform
import sys
import numpy as np
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from datasets import load_dataset

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

LABEL_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def ensure_directories() -> None:
    os.makedirs("results/csv", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("writeup", exist_ok=True)
    os.makedirs("presentation", exist_ok=True)


def load_ag_news_data(dev_size: float = 0.1):
    """
    Load AG News and split the original training set into train/dev.
    The official test set remains untouched.
    """
    dataset = load_dataset("ag_news")

    train_full = dataset["train"]
    test_set = dataset["test"]

    split = train_full.train_test_split(
        test_size=dev_size,
        seed=SEED,
        stratify_by_column="label",
    )
    train_set = split["train"]
    dev_set = split["test"]

    print("Dataset loaded.")
    print("Split strategy: stratified by label")
    print(f"Train size: {len(train_set)}")
    print(f"Dev size:   {len(dev_set)}")
    print(f"Test size:  {len(test_set)}")

    return train_set, dev_set, test_set


def extract_text_and_labels(dataset_split):
    return dataset_split["text"], dataset_split["label"]


def get_package_versions(package_names):
    versions = {}
    for package_name in package_names:
        try:
            versions[package_name] = version(package_name)
        except PackageNotFoundError:
            versions[package_name] = "not installed"
    return versions


def save_experiment_config(output_file: str, config: dict) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        **config,
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"Saved experiment config to {output_file}")
