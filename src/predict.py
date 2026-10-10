"""Classify a headline using a locally trained model."""

import argparse
import json
from pathlib import Path
from utils import LABEL_NAMES


def predict(model_path, texts):
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("Provide nonempty news texts")
    path = Path(model_path)
    if path.suffix == ".joblib":
        # joblib/pickle can execute code: load only artifacts you trained or trust.
        import joblib

        model = joblib.load(path)
        probabilities = model.predict_proba(texts)
        labels = model.classes_
    else:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True, use_safetensors=True
        )
        model.eval()
        limit = min(tokenizer.model_max_length, model.config.max_position_embeddings)
        tokens = tokenizer(
            texts, truncation=True, padding=True, max_length=limit, return_tensors="pt"
        )
        with torch.no_grad():
            probabilities = model(**tokens).logits.softmax(-1).cpu().numpy()
        labels = list(range(4))
    output = []
    for text, scores in zip(texts, probabilities):
        idx = int(scores.argmax())
        label = int(labels[idx])
        output.append(
            {
                "text": text,
                "label_id": label,
                "label": LABEL_NAMES[label],
                "probabilities": {
                    LABEL_NAMES[int(i)]: float(p) for i, p in zip(labels, scores)
                },
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Trusted local .joblib pipeline or BERT model directory",
    )
    parser.add_argument(
        "--text", action="append", required=True, help="Repeat for multiple headlines"
    )
    args = parser.parse_args()
    print(json.dumps(predict(args.model, args.text), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
