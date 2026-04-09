
# 📰 News Topic Classification Using Machine Learning and Transformer-Based Models

---


## 👥 Team Members

* **George Liu**
* **Gordon Zou**
* **Louis Dong**
* **Zhiqi Zhou**


---


## 🔍 Motivation

Text classification plays a critical role in organizing and understanding massive volumes of news content generated daily across digital platforms. 

Traditional NLP approaches rely on **bag-of-words and TF-IDF representations**, which often fail to capture complex contextual relationships between words. With the emergence of transformer-based models like **BERT**, we can now model deeper semantic meaning. 

This project investigates the performance gap between frequency-based statistical models and deep contextual embeddings.


---


## 🧠 Research Question

> **How much improvement does a transformer-based model (BERT) provide over traditional TF-IDF-based classifiers for news topic classification?**


---


## 🧪 Methodology

### **1. Dataset**
We use the **AG News dataset**, a standard benchmark for text classification.
* **Categories:** World, Sports, Business, Sci/Tech.
* **Dataset size:** ~108,000 (Train) / ~12,000 (Dev) / ~7,600 (Test).

### **2. Feature Representation**

#### **TF-IDF Representation**
$$
tfidf(t,d) = tf(t,d) \cdot \log \frac{N}{df(t)}
$$

#### **Contextual Embedding (BERT)**
$$
H = \text{BERT}(X)
$$

### **3. Models Compared**
* **Baseline:** Most Frequent Class.
* **Traditional:** Naive Bayes (MultinomialNB), Logistic Regression.
* **Transformer:** Fine-tuned `bert-base-uncased`.


---


## 📁 Project Structure

```text
news-topic-classification/
│
├── data/                           # Dataset files (train/dev/test)
│   ├── train.csv
│   └── ...
│
├── src/                            # Core implementation
│   ├── baseline_models.py          # NB & Logistic Regression
│   ├── bert_model.py               # BERT fine-tuning logic
│   ├── utils.py                    # Preprocessing & helpers
│   └── error_analysis.py           # Misclassification analysis
│
├── results/                        # Output results
│   ├── csv/                        # Metric logs
│   └── plots/                      # Confusion matrices
│
├── writeup/                        # Final report
│   └── report.pdf
│
├── presentation/                   # Presentation materials
│   └── slides.pptx
│
├── requirements.txt                # Dependencies
└── README.md                       # Documentation
````

-----

## ⚙️ Installation

**1. Clone the repository:**

```bash
git clone [https://github.com/](https://github.com/)<your-username>/news-topic-classification.git
cd news-topic-classification
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

-----

## ▶️ Running Experiments

### **Step 1: Train Baseline Models**

```bash
python src/baseline_models.py
```

### **Step 2: Train BERT Model**

```bash
python src/bert_model.py
```

### **Step 3: Run Full Pipeline**

```bash
python src/run_experiments.py
```

-----

## 📊 Results Summary

  * **Non-linearity:** Transformer models effectively capture semantic nuances missed by TF-IDF.
  * **Efficiency:** Logistic Regression remains a highly competitive baseline for this dataset.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | \~25% | - | - | - |
| **Naive Bayes** | \~90% | 0.90 | 0.90 | 0.90 |
| **Logistic Regression** | \~91% | 0.91 | 0.91 | 0.91 |
| **BERT (Fine-tuned)** | **\~94%** | **0.94** | **0.94** | **0.94** |

-----

## 🔍 Classification Example

**Input Sequence:**

> "Apple reported strong quarterly revenue driven by iPhone sales."

**Analysis:**

  * **True Label:** `Business`
  * **Predicted Label:** `Sci/Tech`

-----

## 📘 Documentation & Presentation

  * **Final Report:** [View Report](https://www.google.com/search?q=./writeup/report.pdf)
  * **Slides:** [View Presentation](https://www.google.com/search?q=./presentation/slides.pptx)

-----

## 📚 References

  * Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. Proceedings of EMNLP.
  * Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2017).Bag of Tricks for Efficient Text Classification. Proceedings of EACL.
  * Zhang, X., Zhao, J., & LeCun, Y. (2015).Character-Level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems.
  * Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016).Hierarchical Attention Networks for Document Classification. Proceedings of NAACL.
  * Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019).
  * BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of NAACL.

-----

## ✨ Author

**Gordon Zou**
New York University

-----

## 📄 License

MIT License. Developed as part of coursework at **New York University (NYU)**.












