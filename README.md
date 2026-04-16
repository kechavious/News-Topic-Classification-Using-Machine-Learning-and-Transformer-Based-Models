# 📰 News Topic Classification Using Machine Learning and Transformer-Based Models

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![HuggingFace](https://img.shields.io/badge/Transformers-BERT-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 👥 Team Members

- **George Liu**  
- **Gordon Zou**  
- **Louis Dong**  
- **Zhiqi Zhou**

---

## 🔍 Motivation

Text classification plays a critical role in organizing and understanding massive volumes of news content generated daily across digital platforms.

Traditional NLP approaches often rely on **bag-of-words** or **TF-IDF representations**, which can struggle to capture semantic context and long-range dependencies.

With the emergence of transformer-based models such as **BERT**, contextual embeddings enable significantly stronger language understanding.

This project evaluates the performance gap between traditional statistical classifiers and transformer-based neural models for multi-class news topic classification.

---

## 🧠 Research Question

> **How much improvement does a transformer-based model (BERT) provide over traditional TF-IDF-based classifiers for news topic classification?**

---

## 📂 Dataset

We use the **AG News Dataset**, a standard benchmark for text classification.

| Label | Category |
|------|----------|
| 1 | World |
| 2 | Sports |
| 3 | Business |
| 4 | Sci/Tech |

### Dataset Size

- Training: ~108,000  
- Validation: ~12,000  
- Test: ~7,600  

The original AG News training split (120,000 samples) was partitioned into
108,000 training examples and 12,000 validation examples.

The validation set was used for hyperparameter tuning and model selection,
while the official test set was reserved strictly for final evaluation.

---

## 🧪 Methodology

### 1️⃣ Feature Representation

### 2️⃣ Experimental Protocol

All models were trained on the training split and evaluated on the held-out
validation split during development.

Final performance metrics were reported only once on the untouched test set.

### 🔹 TF-IDF Representation

$$
\text{tfidf}(t,d)=\text{tf}(t,d)\cdot \log\left(\frac{N}{df(t)}\right)
$$

Where:

- \( t \): term  
- \( d \): document  
- \( N \): total number of documents  
- \( df(t) \): number of documents containing term \( t \)

---

### 🔹 Transformer Representation (BERT)

Input sequence:

$$
X=(x_1,x_2,\dots,x_n)
$$

Contextual encoding:

$$
H=\text{BERT}(X)
$$

Classification output:

$$
\hat{y}=\arg\max \text{Softmax}(W h_{[CLS]})
$$

Where:

- \( h_{[CLS]} \) is the sentence-level representation  
- \( W \) is the learned classification matrix

---

## 🤖 Models Compared

### Traditional Machine Learning

- Most Frequent Baseline  
- Multinomial Naive Bayes  
- Logistic Regression (TF-IDF)

### Transformer-Based Deep Learning

- Fine-tuned `bert-base-uncased`

---

## 📊 Final Results

| Model | Accuracy | Precision | Recall | F1 Score |
|------|----------|-----------|--------|----------|
| Most Frequent Baseline | 0.2500 | 0.0625 | 0.2500 | 0.1000 |
| Naive Bayes | 0.9024 | 0.9024 | 0.9024 | 0.9024 |
| Logistic Regression | 0.9180 | 0.9180 | 0.9180 | 0.9180 |
| **BERT** | **0.9487** | **0.9487** | **0.9487** | **0.9487** |

---

## 📈 Key Insights

- Logistic Regression is a strong traditional baseline.
- BERT achieved the best performance across all metrics.
- Contextual embeddings significantly improve classification quality.
- Traditional TF-IDF models remain competitive with lower computational cost.

---

## 🔍 Error Analysis

Common remaining BERT errors occurred in semantically overlapping categories:

For readability, a representative subset of misclassified examples is shown below.

### Business ↔ Sci/Tech

Shared vocabulary such as:

- Apple
- AI
- chips
- products
- earnings

### World ↔ Sports

International entities appearing in both news and sports contexts.

### Short Headline Ambiguity

Some headlines provide insufficient context.

### Example

```text
Apple reported strong quarterly revenue driven by iPhone sales.

True Label: Business
Predicted Label: Sci/Tech
````

---

## 📁 Project Structure

```text
News-Topic-Classification/
│
├── data/
│
├── src/
│   ├── baseline_models.py
│   ├── bert_model.py
│   ├── utils.py
│   ├── error_analysis.py
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
└── README.md
```

---

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/kechavious/News-Topic-Classification-Using-Machine-Learning-and-Transformer-Based-Models.git
cd News-Topic-Classification-Using-Machine-Learning-and-Transformer-Based-Models
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

**Mac / Linux**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

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

### Generate Visualizations

```bash
python src/plot.py
```

---

## ⚙️ BERT Training Configuration

| Parameter     | Value             |
| ------------- | ----------------- |
| Base Model    | bert-base-uncased |
| Epochs        | 5                 |
| Batch Size    | 16                |
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

## 📘 Documentation

* Final report: `writeup/report.pdf`
* Presentation slides: `presentation/slides.pptx`

---

## 🚀 Future Improvements

* RoBERTa / DistilBERT comparison
* Hyperparameter tuning
* Confusion matrix visualization
* Streamlit deployment
* Real-time news classification demo

---

## 📚 References

* Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*
* Kim (2014). *CNN for Sentence Classification.*
* Joulin et al. (2017). *Bag of Tricks for Efficient Text Classification.*
* Yang et al. (2016). *Hierarchical Attention Networks.*

---

## 👤 Author

**Gordon Zou**
GitHub: [https://github.com/kechavious](https://github.com/kechavious)

---

## 📄 License

MIT License

```
```















