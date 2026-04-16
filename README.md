# 📰 News Topic Classification Using Machine Learning and Transformer-Based Models

---

## 👥 Team Members

- **George Liu**  
- **Gordon Zou**  
- **Louis Dong**  
- **Zhiqi Zhou**

---

## 🔍 Motivation

Text classification plays a critical role in organizing and understanding massive volumes of news content generated daily across digital platforms.

Traditional NLP approaches rely on **bag-of-words and TF-IDF representations**, which often fail to capture contextual relationships between words. With the emergence of transformer-based models such as **BERT**, it is now possible to model deeper semantic meaning through contextual embeddings.

This project investigates the performance gap between frequency-based statistical models and transformer-based neural models.

---

## 🧠 Research Question

> **How much improvement does a transformer-based model (BERT) provide over traditional TF-IDF-based classifiers for news topic classification?**

---

## 🧪 Methodology

### **1. Dataset**

We use the **AG News dataset**, a widely used benchmark for text classification.

- **Categories:** World, Sports, Business, Sci/Tech  
- **Dataset size:** ~108,000 (Train) / ~12,000 (Dev) / ~7,600 (Test)

---

### **2. Feature Representation**

#### 🔹 TF-IDF Representation

$$
\text{tfidf}(t,d) = \text{tf}(t,d) \cdot \log\left(\frac{N}{df(t)}\right)
$$

Where:

- \( t \): term  
- \( d \): document  
- \( N \): total number of documents  
- \( df(t) \): number of documents containing term \( t \)

---

#### 🔹 Transformer Representation (BERT)

Input sequence:

$$
X = (x_1, x_2, ..., x_n)
$$

Contextual encoding:

$$
H = \text{BERT}(X)
$$

Classification layer:

$$
\hat{y} = \arg\max \; \text{Softmax}(W \cdot h_{[CLS]})
$$

Where:

- \( h_{[CLS]} \) is the contextual representation of the entire sequence  
- \( W \) is the learned classification weight matrix  

---

### **3. Models Compared**

- **Baseline:** Most Frequent Class  
- **Traditional Models:**  
  - Naive Bayes (MultinomialNB)  
  - Logistic Regression (TF-IDF features)  
- **Neural Model:**  
  - Fine-tuned `bert-base-uncased`

---

## 📁 Project Structure

```text
news-topic-classification/
│
├── data/                           # Dataset (optional, auto-loaded)
│
├── src/                            # Core implementation
│   ├── baseline_models.py          # NB & Logistic Regression
│   ├── bert_model.py               # BERT fine-tuning
│   ├── utils.py                    # Data loading & preprocessing
│   ├── error_analysis.py           # Misclassification analysis
│   └── plot_results.py
|   
│
├── results/
│   ├── csv/                        # Evaluation results
│   └── plots/                      # Visualization outputs
│
├── writeup/                        # Final report
├── presentation/                   # Slides
│
├── requirements.txt
└── README.md
````
---

## ⚙️ Installation

### 1. Clone repository

```bash
git clone https://github.com/your-username/news-topic-classification.git
cd news-topic-classification
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate environment

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running Experiments

### **Train Baseline Models**

```bash
python src/baseline_models.py
```

### **Train BERT Model**

```bash
python src/bert_model.py
```

---

## 📊 Results Summary

| Model               | Accuracy | Precision | Recall   | F1       |
| ------------------- | -------- | --------- | -------- | -------- |
| Baseline            | ~25%     | -         | -        | -        |
| Naive Bayes         | ~90%     | 0.90      | 0.90     | 0.90     |
| Logistic Regression | ~91%     | 0.91      | 0.91     | 0.91     |
| BERT                | **~94%** | **0.94**  | **0.94** | **0.94** |

### 🔑 Key Insights

* Logistic Regression is a strong traditional baseline
* BERT significantly improves performance via contextual understanding
* Major confusion occurs between:

  * Business ↔ Sci/Tech
  * World ↔ Sports

---

## 🔍 Error Analysis

Common error patterns:

* **Business vs Sci/Tech**

  * Overlapping vocabulary (companies, AI, products)

* **World vs Sports**

  * International events with similar entities

* **Short text ambiguity**

  * Insufficient context

### Example

```
"Apple reported strong quarterly revenue driven by iPhone sales."
True Label: Business  
Predicted Label: Sci/Tech
```

---

## 📘 Documentation

* Final report: `writeup/report.pdf`
* Presentation: `presentation/slides.pptx`

---

## 📚 References

Kim, Y. (2014).
*Convolutional Neural Networks for Sentence Classification.* EMNLP.

Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2017).
*Bag of Tricks for Efficient Text Classification.* EACL.

Zhang, X., Zhao, J., & LeCun, Y. (2015).
*Character-Level Convolutional Networks for Text Classification.* NeurIPS.

Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016).
*Hierarchical Attention Networks for Document Classification.* NAACL.

Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019).
*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL.

---

## 📄 License

MIT License

```
```













