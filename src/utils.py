import os
import random
import numpy as np
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

    split = train_full.train_test_split(test_size=dev_size, seed=SEED)
    train_set = split["train"]
    dev_set = split["test"]

    print("Dataset loaded.")
    print(f"Train size: {len(train_set)}")
    print(f"Dev size:   {len(dev_set)}")
    print(f"Test size:  {len(test_set)}")

    return train_set, dev_set, test_set


def extract_text_and_labels(dataset_split):
    return dataset_split["text"], dataset_split["label"]