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

| Label | Category |
|------|----------|
| 1 | World |
| 2 | Sports |
| 3 | Business |
| 4 | Sci/Tech |

### Dataset Split

- Training Set: ~108,000  
- Validation Set: ~12,000  
- Test Set: ~7,600  

The original AG News training split (120,000 samples) was partitioned into training and validation subsets using a fixed random seed.

The validation set was used for model selection and hyperparameter tuning, while the official test set remained untouched until final evaluation.

---

## 🧪 Methodology

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
| Naive Bayes | 0.9024 | 0.9024 | 0.9024 | 0.9024 |
| Logistic Regression | 0.9180 | 0.9180 | 0.9180 | 0.9180 |
| **BERT** | **0.9487** | **0.9487** | **0.9487** | **0.9487** |

---

## 📈 Performance Visualizations

### Overall Model Comparison

![Model Comparison](results/plots/model_metric_comparison.png)

### Logistic Regression Confusion Matrix

![LR Confusion Matrix](results/plots/confusion_logistic_regression_test.png)

### BERT Confusion Matrix

![BERT Confusion Matrix](results/plots/confusion_bert_test.png)

---

## ⚔️ Logistic Regression vs BERT

Although Logistic Regression is a strong TF-IDF baseline, BERT achieved a clear improvement on the same held-out test set.

The largest gains are concentrated in **semantically ambiguous categories**, rather than easy keyword-driven examples.

Most common improvements occurred in:

- **Business ↔ Sci/Tech**
- **World ↔ Sports**

This suggests that contextual embeddings help resolve ambiguity that sparse lexical features cannot fully capture.

---

## 🔍 Error Analysis

Rather than listing only the first 100 errors, we used a representative stratified sample grouped by confusion pair.

### Common Confusion Pairs

| True Class | Predicted Class | Typical Cause |
|-----------|----------------|--------------|
| Business | Sci/Tech | Company / product overlap |
| Sci/Tech | Business | Revenue / market wording |
| World | Sports | International event ambiguity |

---

### 1️⃣ Business ↔ Sci/Tech

These categories often share vocabulary such as:

- Apple
- AI
- chips
- launch
- revenue
- products

Example:

```text
Apple reported strong quarterly revenue driven by iPhone sales.

True Label: Business
Predicted Label: Sci/Tech
````

This mixes company, product, and financial signals.

---

### 2️⃣ World ↔ Sports

International teams, Olympic events, and national organizations may blur the line between sports coverage and world news.

---

### 3️⃣ Short Headline Ambiguity

Very short headlines often lack enough context, making them difficult even for BERT.

Examples:

```text
Champions advance after upset.
Markets react to shock move.
Leaders meet after crisis.
```

---

## 📌 Key Insights

* Logistic Regression is a strong traditional baseline.
* BERT achieved the best performance across all metrics.
* BERT improvements are concentrated in harder semantic cases.
* Contextual embeddings improve robustness on ambiguous headlines.
* Traditional TF-IDF models remain competitive with lower computational cost.

---

## ⚠️ Limitations

* AG News is a relatively clean and balanced benchmark dataset.
* Headlines are short and may lack sufficient context.
* Training and testing come from the same benchmark distribution.
* BERT results are based on a single run and may vary across random seeds.
* Real-world noisy news streams may be more challenging.

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

To reproduce BERT results locally:

```bash
python src/bert_model.py
```

---

## 📘 Documentation

* Final Report: `writeup/report.pdf`
* Presentation Slides: `presentation/slides.pptx`

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
















