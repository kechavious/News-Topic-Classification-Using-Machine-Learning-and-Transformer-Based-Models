```markdown
# 📰 News Topic Classification

---

## 🔍 Motivation

News content is growing exponentially across digital platforms. Automatically organizing and classifying news articles is essential for search, recommendation systems, and information retrieval.

Traditional methods rely on **TF-IDF and statistical models**, while modern NLP uses **transformers like BERT** to capture contextual meaning.

This project explores the performance gap between these approaches.

---

## 🧠 Research Question

**How do traditional machine learning models compare with transformer-based models in news topic classification?**

---

## 🧪 Methodology

### **1. Dataset**

We use the **AG News dataset**, containing labeled news articles across four categories:

- World  
- Sports  
- Business  
- Technology  

Dataset size:
- ~120,000 training samples  
- ~7,600 test samples  

---

### **2. Text Representation**

#### **TF-IDF Representation**

$$
tfidf(t,d) = tf(t,d) \cdot \log \frac{N}{df(t)}
$$

Captures word importance based on frequency.

---

#### **Contextual Embedding (BERT)**

$$
H = \text{BERT}(X)
$$

Learns deep contextual representations of text.

---

### **3. Classification Models**

#### **Baseline**

- Most Frequent Class

#### **Traditional Models**

- Naive Bayes  
- Logistic Regression (TF-IDF)

#### **Transformer Model**

- Fine-tuned BERT  

---

### **4. Evaluation Metrics**

Accuracy:

$$
Accuracy = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
$$

Precision / Recall / F1-score are also used for detailed evaluation.

---

### **5. Error Analysis**

We analyze misclassified examples to identify:

- Confusion between categories  
- Ambiguous wording  
- Model limitations  

---

## 📁 Project Structure

```

news_classification/
│
├── data/                           # Dataset (train/test splits)
│   ├── train.csv
│   ├── test.csv
│
├── preprocessing/                  # Text preprocessing
│   ├── clean_text.py
│   ├── tokenizer.py
│
├── models/                         # Model implementations
│   ├── naive_bayes.py
│   ├── logistic_regression.py
│   ├── bert_model.py
│
├── evaluation/                     # Metrics & evaluation
│   ├── metrics.py
│   ├── evaluate.py
│
├── experiments/                    # Experiment pipeline
│   ├── train.py
│   ├── run_experiments.py
│
├── results/                        # Outputs & visualizations
│   ├── logs/
│   ├── plots/
│
├── notebooks/                      # Jupyter experiments
│   └── analysis.ipynb
│
├── requirements.txt
├── README.md
└── LICENSE

````

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/news-classification.git
cd news-classification
pip install -r requirements.txt
````

---

## ▶️ Running Experiments

### **1. Train model**

```bash
python experiments/train.py --model nb
python experiments/train.py --model lr
python experiments/train.py --model bert
```

---

### **2. Evaluate model**

```bash
python evaluation/evaluate.py --model bert
```

---

### **3. Run full experiment pipeline**

```bash
python experiments/run_experiments.py
```

---

## 📊 Results Summary

* Traditional models perform well with TF-IDF features
* Logistic Regression outperforms Naive Bayes
* BERT achieves the highest accuracy due to contextual understanding

| Model               | Accuracy |
| ------------------- | -------- |
| Baseline            | ~25%     |
| Naive Bayes         | ~80%     |
| Logistic Regression | ~88%     |
| BERT                | ~93%     |

---

## 🔍 Example

Input:

```
Apple reports strong quarterly earnings driven by iPhone sales.
```

Output:

```
Business
```

---

## 🛠️ Tech Stack

* Python
* scikit-learn
* PyTorch
* HuggingFace Transformers
* NLTK / spaCy

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
```

---

## 📚 References

* Kim, Y. (2014). CNN for Sentence Classification
* Joulin et al. (2017). FastText
* Zhang et al. (2015). Character-level CNN
* Yang et al. (2016). Hierarchical Attention Networks
* Devlin et al. (2019). BERT

---

## ✨ Author

**Gordon Zou**
New York University

---

## 📄 License

MIT License

```
```






