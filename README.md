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

Text classification plays a critical role in organizing and understanding massive volumes of news content generated daily across digital platforms.

Traditional NLP approaches rely on **bag-of-words and TF-IDF representations**, which fail to capture contextual relationships between words.

With the emergence of transformer-based models like **BERT**, it is now possible to model deeper semantic meaning.

This project investigates **how contextual models improve classification performance compared to classical methods**.

---

## 🧠 Research Question

**How much improvement does a transformer-based model (BERT) provide over traditional TF-IDF-based classifiers for news topic classification?**

---

## 🧪 Methodology

### **1. Dataset**

We use the **AG News dataset**, a standard benchmark for text classification.

Categories:

- World  
- Sports  
- Business  
- Sci/Tech  

Dataset size:

- Training set: ~108,000  
- Development set: ~12,000  
- Test set: ~7,600  

---

### **2. Feature Representation**

#### **TF-IDF Representation**

$$
tfidf(t,d) = tf(t,d) \cdot \log \frac{N}{df(t)}
$$

Captures word importance using frequency-based weighting.

---

#### **Contextual Embedding (BERT)**

$$
H = \text{BERT}(X)
$$

Generates contextual embeddings for each input sequence.

---

### **3. Classification Models**

#### **Baseline**

- Most Frequent Class  

---

#### **Traditional Models**

- Naive Bayes (MultinomialNB)  
- Logistic Regression (TF-IDF)  

---

#### **Transformer Model**

- Fine-tuned BERT (`bert-base-uncased`)  

---

### **4. Evaluation Metrics**

We evaluate model performance using:

- Accuracy  
- Precision (macro)  
- Recall (macro)  
- F1-score (macro)  

---

### **5. Error Analysis**

We analyze misclassified examples to understand:

- Category confusion patterns  
- Ambiguity in short texts  
- Limitations of each model  

---

## 📁 Project Structure

```

news-topic-classification/
│
├── data/                           # Dataset files
│   ├── train.csv
│   ├── dev.csv
│   └── test.csv
│
├── src/                            # Core implementation
│   ├── baseline_models.py          # Naive Bayes & Logistic Regression
│   ├── bert_model.py               # BERT fine-tuning
│   ├── utils.py                    # Preprocessing & helpers
│   └── error_analysis.py           # Misclassification analysis
│
├── results/                        # Output results
│   ├── csv/
│   └── plots/
│
├── analysis/                       # Additional experiments
│
├── writeup/                        # Final report
│   └── report.pdf
│
├── presentation/                   # Slides
│   ├── slides.pptx
│   └── notes.md
│
├── requirements.txt
├── README.md
└── LICENSE

````

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/news-topic-classification.git
cd news-topic-classification
````

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running Experiments

### **1. Train Baseline Models**

```bash
python src/baseline_models.py
```

---

### **2. Train BERT Model**

```bash
python src/bert_model.py
```

---

### **3. Run Full Pipeline**

```bash
python src/run_experiments.py
```

---

## 📊 Results Summary

* Traditional models perform strongly with TF-IDF features
* Logistic Regression slightly outperforms Naive Bayes
* BERT achieves the best performance due to contextual understanding

| Model               | Accuracy |
| ------------------- | -------- |
| Baseline            | ~25%     |
| Naive Bayes         | ~90%     |
| Logistic Regression | ~91%     |
| BERT                | ~92–95%  |

---

## 🔍 Example

```
Input:
"Apple reported strong quarterly revenue driven by iPhone sales."

True Label: Business  
Predicted Label: Sci/Tech  
```

---

## 📘 Report

See:

```
writeup/report.pdf
```

---

## 🎤 Presentation

See:

```
presentation/slides.pptx
presentation/notes.md
```

---

## 📚 References

* Kim (2014) — CNN for Sentence Classification
* Joulin et al. (2017) — FastText
* Zhang et al. (2015) — Character-level CNN
* Yang et al. (2016) — Hierarchical Attention Networks
* Devlin et al. (2019) — BERT

---

## ✨ Author

**Gordon Zou**
New York University

---

## 📄 License

MIT License

This project was developed as part of coursework at New York University (NYU).
NYU does not claim ownership or endorsement of this software.

---

```
```














