```md
# 📰 News Topic Classification Using Machine Learning and Transformer-Based Models

---

## 👥 Team Members
- George Liu  
- Gordon Zou  
- Louis Dong  
- Zhiqi Zhou  

---

## 🔍 Motivation

Text classification is a fundamental task in Natural Language Processing (NLP) with applications in:

- Document organization  
- Content recommendation  
- Spam detection  
- Information retrieval  

Traditional models rely on **sparse lexical features** (e.g., TF-IDF), while modern transformer-based models like **BERT** capture **contextual semantics**.

This project investigates the performance gap between these two paradigms.

---

## 🧠 Research Question

**How much improvement does a transformer-based model (BERT) provide over traditional TF-IDF-based classifiers for news topic classification?**

---

## 📊 Dataset

We use the **AG News Dataset** from Hugging Face.

### **Categories**
- World  
- Sports  
- Business  
- Sci/Tech  

### **Data Split**
- Training: ~108,000  
- Development: ~12,000  
- Test: ~7,600  

---

## 🧪 Methodology

### **1. Baseline Model**

Most Frequent Class:

$$
\hat{y} = \arg\max_{c} \; \text{freq}(c)
$$

---

### **2. Traditional Machine Learning**

#### TF-IDF Representation

$$
\text{tfidf}(t,d) = \text{tf}(t,d) \cdot \log\left(\frac{N}{df(t)}\right)
$$

Models:
- Naive Bayes  
- Logistic Regression  

---

### **3. Transformer Model (BERT)**

We fine-tune:

```

bert-base-uncased

```

Prediction:

$$
\hat{y} = \arg\max \; \text{Softmax}(W \cdot h_{[CLS]})
$$

---

## 📏 Evaluation Metrics

We evaluate using:

- Accuracy  
- Precision (macro)  
- Recall (macro)  
- F1-score (macro)  

---

## 📊 Results Summary

| Model | Accuracy |
|------|--------|
| Most Frequent Baseline | ~25% |
| Naive Bayes | ~90% |
| Logistic Regression | ~91% |
| BERT | ~92–95% |

---

## 🔍 Error Analysis

We analyze misclassified samples to understand model limitations.

### **Observed Patterns**

- **Business ↔ Sci/Tech**
  - Shared vocabulary (companies, products, AI, revenue)

- **World ↔ Sports**
  - International competitions resemble geopolitical events

- **Short Text Ambiguity**
  - Lack of context leads to confusion

### **Example**

```

"Apple reported strong quarterly revenue driven by iPhone sales."
True: Business
Predicted: Sci/Tech

```

---

## 📁 Project Structure

```

news-topic-classification/
│
├── src/                         # Core implementation
│   ├── baseline_models.py       # TF-IDF + NB + LR
│   ├── bert_model.py            # BERT fine-tuning
│   ├── utils.py                 # Data loading & helpers
│   └── error_analysis.py        # Misclassification analysis
│
├── results/                     # Outputs
│   └── csv/
│       ├── model_results_summary.csv
│       ├── bert_results_summary.csv
│       └── error files
│
├── analysis/                    # Interpretation
├── writeup/                     # Paper
└── presentation/                # Slides

````

---

## ⚙️ Installation

Clone repository:

```bash
git clone https://github.com/your-repo/news-topic-classification.git
cd news-topic-classification
````

Create environment:

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running Experiments

### **1. Run Baseline Models**

```bash
python src/baseline_models.py
```

Outputs:

* Baseline performance CSV
* Error analysis CSV

---

### **2. Run BERT Model**

```bash
python src/bert_model.py
```

Outputs:

* BERT performance CSV
* Error analysis

---

## 📂 Output Directory

```
results/csv/
```

Contains:

* Model performance summaries
* Misclassified samples
* Evaluation outputs

---

## 📈 Key Insights

* Logistic Regression outperforms Naive Bayes due to better feature weighting
* BERT achieves highest accuracy by leveraging contextual embeddings
* Most errors arise from **semantic overlap between categories**

---

## 🔮 Future Work

* Hyperparameter tuning
* Alternative transformer models (RoBERTa, DistilBERT)
* Confusion matrix visualization
* Larger and more diverse datasets

---

## 📚 References

* Devlin et al. (2019) — BERT
* Kim (2014) — CNN for Text Classification
* Joulin et al. (2017) — FastText

---

## ✨ Author

**Gordon Zou**
New York University

---

## 📄 License

MIT License

This project was developed for academic purposes at NYU.

```
```



