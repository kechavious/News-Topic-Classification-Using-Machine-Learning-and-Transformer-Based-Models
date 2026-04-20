````md
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

Text classification plays a critical role in organizing and understanding the massive volume of news content published daily.

Traditional NLP approaches often rely on **bag-of-words** or **TF-IDF representations**, which can be effective but may struggle to capture semantic context and long-range dependencies.

With the emergence of transformer-based models such as **BERT**, contextual embeddings enable substantially stronger language understanding.

This project evaluates the performance gap between traditional statistical classifiers and transformer-based neural models for multi-class news topic classification.

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

### Dataset Size

- Training: ~108,000  
- Validation: ~12,000  
- Test: ~7,600  

The original AG News training split (120,000 samples) was partitioned into training and validation subsets using a fixed random seed.

The validation split was used for model selection and hyperparameter tuning, while the official test set was reserved strictly for final evaluation.

---

## 🧪 Methodology

### 1️⃣ Feature Representation

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
- \( W \) is the learned classification layer

---

### 2️⃣ Experimental Protocol

All models were trained on the training split and evaluated on the held-out validation split during development.

Final performance metrics were reported only once on the untouched test set.

This ensures a fair comparison between traditional machine learning baselines and transformer-based models.

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

The improvement is especially visible in semantically overlapping categories such as:

- **Business ↔ Sci/Tech**
- **World ↔ Sports**

This suggests that contextual embeddings help resolve ambiguity that sparse lexical features cannot fully capture.

Rather than improving uniformly across all examples, BERT appears most beneficial where classification becomes semantically harder.

---

## 🔍 Error Analysis

To strengthen qualitative analysis, we compared model predictions on the same test examples and inspected representative misclassifications.

Rather than simply listing the first 100 errors, we constructed a stratified sample grouped by confusion pair so that the selected examples reflect dominant error patterns.

### 1️⃣ Business ↔ Sci/Tech

These categories often share vocabulary involving:

- Apple
- AI
- chips
- products
- earnings
- launches

Example:

```text
Apple reported strong quarterly revenue driven by iPhone sales.

True Label: Business
Predicted Label: Sci/Tech
````

This type of headline mixes company, product, and financial signals.

---

### 2️⃣ World ↔ Sports

International entities and events may blur the boundary between geopolitical and sports coverage.

Examples include:

* Olympic events
* international teams
* national federations
* cross-border competitions

---

### 3️⃣ Short Headline Ambiguity

Very short headlines often provide insufficient context, making classification difficult even for BERT.

Examples:

```text
Champions advance after upset.
Markets react to shock move.
Leaders meet after crisis.
```

Without surrounding context, multiple interpretations remain plausible.

---

## 💡 Key Insights

* Logistic Regression is a strong traditional baseline.
* BERT achieved the best performance across all metrics.
* The largest gains appear in overlapping categories rather than easy cases.
* Contextual embeddings improve robustness on ambiguous headlines.
* Traditional TF-IDF models remain competitive with much lower computational cost.

---

## ⚠️ Limitations

* AG News is a relatively clean and balanced benchmark dataset.
* Headlines are short and may lack sufficient context.
* Training and testing come from the same benchmark distribution.
* Reported BERT performance is based on a single run and may vary across random seeds.
* Results may overestimate real-world performance on noisier news streams.

---

## 🚀 Future Improvements

* Compare additional transformer models such as RoBERTa and DistilBERT
* Perform broader hyperparameter tuning
* Evaluate robustness under domain shift
* Analyze which BERT layers carry classification signal
* Build an interactive Streamlit demo for real-time classification

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

## 📘 Documentation

* Final report: `writeup/report.pdf`
* Presentation slides: `presentation/slides.pptx`

---

## 📚 References

* Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*
* Joulin et al. (2017). *Bag of Tricks for Efficient Text Classification.*
* Kim (2014). *CNN for Sentence Classification.*
* Yang et al. (2016). *Hierarchical Attention Networks.*

---

## 📄 License

MIT License

```
```















