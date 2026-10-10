# GitHub About settings

These are the recommended repository metadata fields. This file documents them;
GitHub's sidebar is configured separately through the repository's About editor.

## Description

AG News text classification with TF-IDF baselines and BERT, featuring paired statistical analysis, reproducible training tools, local inference, and a research report.

## Website

Link to the editable-source research report rendered as PDF:

https://github.com/kechavious/News-Topic-Classification-Using-Machine-Learning-and-Transformer-Based-Models/blob/main/writeup/paper.pdf

## Topics

- ag-news
- bert
- text-classification
- natural-language-processing
- machine-learning
- pytorch
- transformers
- scikit-learn
- reproducible-research
- error-analysis

## Applying the settings

On the repository home page, click the gear next to **About**. Paste the description,
website and topics, then save. Alternatively, an authenticated maintainer with
repository administration access can apply the description and add the topics:

```bash
gh repo edit kechavious/News-Topic-Classification-Using-Machine-Learning-and-Transformer-Based-Models \
  --description "AG News text classification with TF-IDF baselines and BERT, featuring paired statistical analysis, reproducible training tools, local inference, and a research report." \
  --add-topic ag-news,bert,text-classification,natural-language-processing,machine-learning,pytorch,transformers,scikit-learn,reproducible-research,error-analysis
```

If the repository already has a project website, retain that URL. Otherwise the
report URL above is a useful website field. These settings do not imply peer review,
production readiness or exact reproduction of the historical training runs.
