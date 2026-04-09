這份新的版本針對你提到的\*\*「分隔感」**與**「大標題風格」\*\*進行了視覺強化。

主要變動如下：

1.  **強制空行**：在每個 `---` 分隔線與標題之間加入了雙倍空行，避免 GitHub 渲染時黏在一起。
2.  **視覺錨點**：標題統一使用 `## 標題` 並搭配特定的 Emoji，增加層次感。
3.  **區塊化 (Blockquotes)**：將核心的「研究問題」與「定義」放進引用塊中，讓它們從背景正文中跳脫出來。
4.  **代碼塊優化**：確保結構圖與代碼範例前後都有足夠的白空間（White Space）。

-----

````markdown
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
Captures word importance using frequency-based weighting.

#### **Contextual Embedding (BERT)**
$$
H = \text{BERT}(X)
$$
Generates deep contextual embeddings for each input sequence.

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
  * **Predicted Label:** `Sci/Tech` (Example of common confusion in the AG News dataset)

-----

## 📘 Documentation & Presentation

  * **Final Report:** [View Report](https://www.google.com/search?q=./writeup/report.pdf)
  * **Slides:** [View Presentation](https://www.google.com/search?q=./presentation/slides.pptx)

-----

## 📚 References

  * Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*.
  * Kim (2014). *Convolutional Neural Networks for Sentence Classification*.
  * Zhang et al. (2015). *Character-level Convolutional Networks for Text Classification*.

-----

## ✨ Author

**Gordon Zou** *New York University*

-----

## 📄 License

MIT License. Developed as part of coursework at **New York University (NYU)**.

```
```














