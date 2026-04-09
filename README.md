# News Topic Classification Using Machine Learning and Transformer-Based Models

## 👥 Team Members
- George Liu  
- Gordon Zou  
- Louis Dong  
- Zhiqi Zhou  

---

## 📌 Motivation

Text classification is a fundamental task in Natural Language Processing (NLP) with applications in:

- Document organization  
- Content recommendation  
- Spam detection  
- Information retrieval  

Traditional models such as Naive Bayes and Logistic Regression rely on sparse lexical features (e.g., TF-IDF), while modern transformer-based models like BERT can capture contextual semantics.

👉 This project aims to **quantitatively compare traditional models and transformer-based models** on a real-world dataset.

---

## ❓ Research Question

> How much improvement does a transformer-based model (BERT) provide over traditional TF-IDF-based classifiers for news topic classification?

---

## 📊 Dataset

We use the **AG News Dataset** from Hugging Face.

- Number of classes: 4  
- Categories:
  - World  
  - Sports  
  - Business  
  - Sci/Tech  

Dataset split:

- Training set: ~108,000  
- Development set: ~12,000  
- Test set: ~7,600  

---

## ⚙️ Methods

We implement and compare the following models:

### 1️⃣ Baseline
- Most Frequent Class (always predicts the most common label)

### 2️⃣ Traditional Machine Learning
- Naive Bayes (MultinomialNB)
- Logistic Regression with TF-IDF features

### 3️⃣ Transformer Model
- BERT (`bert-base-uncased`)
- Fine-tuned for multi-class classification

---

## 🧪 Evaluation Metrics

We evaluate all models using:

- Accuracy  
- Precision (macro)  
- Recall (macro)  
- F1-score (macro)  

We also perform:

- Confusion matrix analysis  
- Error analysis using misclassified examples  

---

## 📈 Results Summary

| Model | Accuracy |
|------|--------|
| Most Frequent Baseline | ~25% |
| Naive Bayes | ~90% |
| Logistic Regression | ~91% |
| BERT | ~92–95% |

### Key Observations

- Logistic Regression outperforms Naive Bayes due to better use of weighted features  
- BERT achieves the highest performance by leveraging contextual representations  
- Major confusion occurs between:
  - Business ↔ Sci/Tech  
  - World ↔ Sports  

---

## 🔍 Error Analysis

We analyzed misclassified examples and found:

- **Business vs Sci/Tech confusion**
  - Articles often contain both financial and technological terms  
- **World vs Sports confusion**
  - International sports events resemble geopolitical news  
- **Short text ambiguity**
  - Lack of context leads to incorrect classification  

Example:


"Apple reported strong quarterly revenue driven by iPhone sales."
True: Business
Predicted: Sci/Tech


---

## 🧱 Project Structure

```text
news-topic-classification/
├── README.md
├── requirements.txt
├── src/
│   ├── baseline_models.py
│   ├── bert_model.py
│   ├── utils.py
│   └── error_analysis.py
├── results/
│   └── csv/
├── analysis/
├── writeup/
└── presentation/

---

## 🚀 Installation
1. Clone the repository
git clone https://github.com/your-repo/news-topic-classification.git
cd news-topic-classification

2. Create virtual environment
python -m venv venv

3. Activate environment
Windows:
venv\Scripts\activate
Mac/Linux:
source venv/bin/activate

4. Install dependencies
pip install -r requirements.txt

---

## ▶️ How to Run
Run Baseline Models
python src/baseline_models.py

Outputs:

baseline_results.csv
error analysis CSV files
Run BERT Model
python src/bert_model.py

Outputs:

bert_results_summary.csv
errors_bert_test.csv

---
## 📂 Output Files

After running, results are saved in:

results/csv/

Including:

Model performance summary
Misclassified examples
Error analysis

---


## 🔮 Future Work
Hyperparameter tuning
Try other transformer models (RoBERTa, DistilBERT)
Improve class imbalance handling
Add visualization (confusion matrix heatmaps)
Expand dataset to multi-domain classification
---


## 📚 References
Kim, Y. (2014).
Convolutional Neural Networks for Sentence Classification. Proceedings of EMNLP.

Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2017).
Bag of Tricks for Efficient Text Classification. Proceedings of EACL.

Zhang, X., Zhao, J., & LeCun, Y. (2015).
Character-Level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems.

Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016).
Hierarchical Attention Networks for Document Classification. Proceedings of NAACL.

Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019).
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of NAACL.

---

## 📜 License
This project is for academic use only.












