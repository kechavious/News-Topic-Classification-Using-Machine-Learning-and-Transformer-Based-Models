# 📰 News Topic Classification Using Machine Learning and Transformer-Based Models

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![HuggingFace](https://img.shields.io/badge/Transformers-BERT-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 👥 Team Members

- **George Liu**
- **Gordon Zau**
- **Louis Dong**
- **Zhiqi Zhou**

---

## 🔍 Motivation

Text classification is essential for organizing the massive volume of online news content published every day.

Traditional NLP methods such as **Bag-of-Words** and **TF-IDF** are computationally efficient and often strong baselines, but they may struggle to capture semantic meaning and contextual ambiguity.

Transformer-based models such as **BERT** use contextual embeddings that better model word meaning based on surrounding tokens.

This project evaluates how much performance improvement BERT provides over traditional machine learning approaches for multi-class news topic classification.

---

## 🧠 Research Question

> **How much improvement does a transformer-based model (BERT) provide over traditional TF-IDF-based classifiers for news topic classification?**

---

## 📂 Dataset

We use the **AG News Dataset**, a widely used benchmark for text classification.

| Label ID | Category |
|------|----------|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

### Dataset Split

- Training Set: 108,000
- Validation Set: 12,000
- Test Set: 7,600 (official AG News test set)

---

## 🧪 Methodology

### Data Split Procedure

The 120,000 AG News training examples were partitioned with `datasets.train_test_split(test_size=0.1, seed=42, stratify_by_column="label")`, producing a label-stratified split of 108,000 training and 12,000 validation examples. The official 7,600-example AG News test set was left untouched throughout development and used only once, for the final evaluation reported below.

The validation set was used for model selection and hyperparameter tuning. No test-set information influenced modeling decisions.

### Traditional Machine Learning Models

Text was converted into **TF-IDF unigram + bigram features**, then evaluated using:

- Most Frequent Baseline
- Multinomial Naive Bayes
- Logistic Regression

### Transformer-Based Model

We fine-tuned:

- `bert-base-uncased`

Final classification is produced using the `[CLS]` token representation passed through a softmax layer.

---

## 🤖 Models Compared

| Category | Models |
|---------|--------|
| Baselines | Most Frequent |
| Classical ML | Naive Bayes, Logistic Regression |
| Deep Learning | BERT |

---

## 📊 Final Results

All reported scores are evaluated on the held-out AG News **test set**.

| Model | Accuracy | Precision | Recall | F1 Score |
|------|----------|-----------|--------|----------|
| Most Frequent Baseline | 0.2500 | 0.0625 | 0.2500 | 0.1000 |
| Naive Bayes | 0.9024 | 0.9019 | 0.9024 | 0.9019 |
| Logistic Regression | 0.9180 | 0.9178 | 0.9180 | 0.9178 |
| **BERT** | **0.9425** | **0.9424** | **0.9425** | **0.9424** |

---

## 📈 Performance Visualizations

### Overall Model Comparison

![Model Comparison](results/plots/model_metric_comparison.png)

### Logistic Regression Confusion Matrix

![LR Confusion Matrix](results/plots/confusion_logistic_regression_test.png)

### BERT Confusion Matrix

![BERT Confusion Matrix](results/plots/confusion_bert_test.png)

---

## ⚔️ Where the BERT Advantage Lives

On the 7,600-example test set, both models were scored on the same inputs. Comparing their predictions gives a sharper picture than accuracy numbers alone:

| Case | Count | Share of test set |
|---|---|---|
| Both correct | 6,862 | 90.3% |
| LR wrong / BERT correct | 301 | 4.0% |
| BERT wrong / LR correct | 115 | 1.5% |
| Both wrong | 322 | 4.2% |

BERT flips 301 LR errors to correct predictions and loses 115 in the other direction. The **net +186 flips** account for ~2.45% of the test set — essentially the entire ~3% accuracy gap between the two models.

### Where BERT pulls ahead

The 301 LR→BERT wins cluster in semantically overlapping pairs:

| Confusion pair (LR's error) | Count |
|---|---|
| Sci/Tech → Business | 56 |
| Business → Sci/Tech | 44 |
| Business → World | 40 |
| World → Sports | 38 |
| World → Business | 31 |
| World → Sci/Tech | 25 |

Four real examples LR got wrong and BERT got right:

- **`IBM to hire even more new workers. By the end of the year, the computing giant plans to have its biggest headcount since 1991.`** — true *Sci/Tech*. LR → Business (fooled by "hire/workers/headcount"); BERT → Sci/Tech.
- **`Justices to debate mail-order wine...`** — true *Business* (commerce regulation). LR → Sci/Tech; BERT → Business.
- **`Live: Olympics day four. Richard Faulds and Stephen Parry are going for gold for Great Britain in Athens.`** — true *World* (international event). LR → Sports (keyword "gold/Olympics"); BERT → World.
- **`India's Tata expands regional footprint via NatSteel buyout.`** — true *World*. LR → Business (M&A vocabulary); BERT → World.

These failures have a common shape: LR latches onto lexical cues ("workers", "gold", "buyout") that co-occur with one class in training, while BERT's contextual embeddings weigh the surrounding words. This is the same reason contextual models separate *Apple (company)* from *apple (fruit)* — the meaning of a token depends on its neighbors, not just its identity.

### Where BERT still struggles

BERT is not immune to semantic overlap. Its remaining errors concentrate in the same pair LR struggles with:

| Both-wrong confusion pair | Count |
|---|---|
| Business ↔ Sci/Tech (combined) | 164 |
| World ↔ Business (combined) | 62 |
| Sci/Tech ↔ World (combined) | 57 |

Tech companies with earnings-driven coverage and geopolitical stories with economic framing remain genuinely ambiguous — the categories themselves overlap in the source news.

### An honest negative result

We checked whether BERT's wins were driven by headline length. They aren't. Median word count is ≈36 for the full test set, for BERT-correct items, for BERT-errors, and for the LR-wrong/BERT-right set. Length is not a useful discriminator here; BERT's advantage is semantic, not structural.

---

## 📌 Key Insights

* Logistic Regression is a strong traditional baseline.
* BERT achieved the best performance across all metrics.
* BERT improvements are concentrated in harder semantic cases.
* Contextual embeddings improve robustness on ambiguous headlines.
* Traditional TF-IDF models remain competitive with lower computational cost.

---

## ⚠️ Limitations

* **Single run, no seed variance.** All reported numbers come from one training run. We did not sweep seeds, so the reported BERT/LR gap does not include a confidence interval.
* **Clean, balanced benchmark.** AG News has evenly sized classes, curated headlines, and minimal noise. Real-world news streams are messier.
* **Short-headline regime.** Inputs are ~36 words on average. Conclusions may not transfer to longer articles where more context is available.
* **Same-distribution eval.** Training and test come from the same AG News distribution. We did not evaluate under domain shift (e.g., a different news source, a different time period).
* **No calibration analysis.** We report accuracy/F1 but did not study predicted-probability calibration.

---

## 🚀 Future Improvements

* Compare RoBERTa, DistilBERT, and other transformer models
* Broader hyperparameter tuning
* Evaluate robustness under domain shift
* Analyze which BERT layers carry classification signal
* Build an interactive Streamlit demo

---

## 📁 Project Structure

```text
News-Topic-Classification/
│
├── data/
├── src/
│   ├── baseline_models.py
│   ├── bert_model.py
│   ├── utils.py
│   ├── error_analysis.py
│   ├── overlap_analysis.py
│   └── plot.py
│
├── results/
│   ├── csv/
│   └── plots/
│
├── writeup/
├── presentation/
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/kechavious/News-Topic-Classification-Using-Machine-Learning-and-Transformer-Based-Models.git
cd News-Topic-Classification-Using-Machine-Learning-and-Transformer-Based-Models
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

**Mac / Linux**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running Experiments

### Train Baseline Models

```bash
python src/baseline_models.py
```

### Train BERT Model

```bash
python src/bert_model.py
```

### Run Error Overlap Analysis

```bash
python src/overlap_analysis.py
```

### Generate Visualizations

```bash
python src/plot.py
```

---

## ⚙️ BERT Training Configuration

| Parameter     | Value             |
| ------------- | ----------------- |
| Base Model    | bert-base-uncased |
| Epochs        | 2                 |
| Batch Size    | 8                 |
| Learning Rate | 2e-5              |
| Optimizer     | AdamW             |
| Max Length    | 128               |

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* PyTorch
* Hugging Face Transformers
* Matplotlib

---

## 🔁 Reproducibility Notes

This repository includes the full training pipeline, evaluation outputs, plots, and analysis scripts.

Trained BERT checkpoints are not included due to file size limitations.

Dependency versions are pinned in `requirements.txt`, and each training script writes an experiment configuration JSON into `results/csv/` so the dataset split, hyperparameters, and package versions used for a run are recorded alongside the metrics.

To reproduce BERT results locally:

```bash
python src/bert_model.py
```

---

## 📘 Documentation

This repository currently includes the README, source code, saved CSV results, and generated plots. Final report and presentation files are not included in the repository.

---

## 👤 Maintainer

**Gordon Zau**
GitHub: [https://github.com/kechavious](https://github.com/kechavious)

---

## 📚 References

* Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*
* Joulin et al. (2017). *Bag of Tricks for Efficient Text Classification.*
* Kim (2014). *CNN for Sentence Classification.*
* Yang et al. (2016). *Hierarchical Attention Networks.*

---

## 📄 License

This project is released under the MIT License. See `LICENSE` for details.











