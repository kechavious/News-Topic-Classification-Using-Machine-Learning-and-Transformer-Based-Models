```md id="y2m1zz"
# 📰 News Topic Classification Using Machine Learning and Transformer-Based Models

---

## 👥 Team Members

- George Liu  
- Gordon Zou  
- Louis Dong  
- Zhiqi Zhou  

---

## 📌 Motivation

Text classification is a core task in Natural Language Processing (NLP), widely used in:

- Document organization  
- Content recommendation systems  
- Spam filtering  
- Information retrieval  

Traditional models (e.g., Naive Bayes, Logistic Regression) rely on **sparse lexical representations** such as TF-IDF, which ignore context.

In contrast, transformer-based models like **BERT** learn **contextual embeddings**, enabling deeper semantic understanding.

👉 This project investigates the **performance gap between classical NLP methods and modern deep learning approaches**.

---

## ❓ Research Question

> To what extent do transformer-based models (BERT) outperform traditional TF-IDF-based classifiers in multi-class news topic classification?

---

## 📊 Dataset

We use the **AG News Dataset** (via Hugging Face), a benchmark dataset for text classification.

### Categories

- 🌍 World  
- ⚽ Sports  
- 💼 Business  
- 💻 Sci/Tech  

### Data Split

- Training set: ~108,000  
- Development set: ~12,000  
- Test set: ~7,600  

---

## ⚙️ Methods

We implement three tiers of models:

---

### 1️⃣ Baseline Model

- **Most Frequent Class**
  - Predicts the most common label  
  - Serves as a lower-bound benchmark (~25%)  

---

### 2️⃣ Traditional Machine Learning

- **Naive Bayes (MultinomialNB)**
  - Assumes feature independence  
  - Efficient and fast baseline  

- **Logistic Regression (TF-IDF)**
  - Uses weighted features  
  - Strong linear classifier baseline  

---

### 3️⃣ Transformer Model

- **BERT (`bert-base-uncased`)**
  - Pretrained transformer architecture  
  - Fine-tuned for classification  
  - Captures contextual word relationships  

---

## 🧪 Evaluation Metrics

We evaluate model performance using:

- **Accuracy** — overall correctness  
- **Precision (macro)** — class-wise exactness  
- **Recall (macro)** — class-wise coverage  
- **F1-score (macro)** — harmonic balance  

---

### 📊 Additional Analysis

- Confusion matrix visualization  
- Misclassification pattern analysis  
- Category-level performance breakdown  

---

## 📈 Results Summary

| Model | Accuracy |
|------|--------|
| Most Frequent Baseline | ~25% |
| Naive Bayes | ~90% |
| Logistic Regression | ~91% |
| BERT | ~92–95% |

---

### 🔑 Key Observations

- Logistic Regression slightly outperforms Naive Bayes due to better feature weighting  
- BERT achieves the highest accuracy through contextual understanding  
- Traditional models remain competitive despite simplicity  

---

## 🔍 Error Analysis

We identify several systematic error patterns:

---

### 1. Business ↔ Sci/Tech Confusion

- Overlapping vocabulary (e.g., "Apple", "AI", "market")  
- Mixed financial and technological context  

---

### 2. World ↔ Sports Confusion

- International sports events resemble geopolitical reporting  

---

### 3. Short Text Ambiguity

- Lack of contextual signals leads to misclassification  

---

### 📌 Example

```

Input:
"Apple reported strong quarterly revenue driven by iPhone sales."

True Label: Business
Predicted Label: Sci/Tech

````

---

## 🧱 Project Structure

```text
news-topic-classification/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── baseline_models.py        # Naive Bayes & Logistic Regression
│   ├── bert_model.py            # BERT fine-tuning
│   ├── utils.py                 # Data processing utilities
│   └── error_analysis.py        # Misclassification analysis
│
├── results/
│   └── csv/                     # Output metrics & error logs
│
├── analysis/                    # Additional analysis scripts
├── writeup/                     # Final report
├── presentation/                # Slides
````

---

## 🚀 Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-repo/news-topic-classification.git
cd news-topic-classification
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

---

### 3️⃣ Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

---

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

---

### 🔹 Run Baseline Models

```bash
python src/baseline_models.py
```

**Outputs**

* `baseline_results.csv`
* Error analysis CSV files

---

### 🔹 Run BERT Model

```bash
python src/bert_model.py
```

**Outputs**

* `bert_results_summary.csv`
* `errors_bert_test.csv`

---

## 📂 Output Files

All results are stored in:

```text
results/csv/
```

Including:

* Model performance summaries
* Misclassified examples
* Detailed error analysis

---

## 🔮 Future Work

* Hyperparameter tuning for all models
* Experiment with RoBERTa / DistilBERT
* Class imbalance handling
* Visualization (confusion matrix heatmaps)
* Extend to multi-domain datasets

---

## 📚 References

* Kim, Y. (2014). *CNN for Sentence Classification*
* Joulin et al. (2017). *FastText*
* Zhang et al. (2015). *Character-level CNN*
* Yang et al. (2016). *Hierarchical Attention Networks*
* Devlin et al. (2019). *BERT*

---

## 📜 License

This project is intended for academic and educational use only.

---

```
```













