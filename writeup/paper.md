---
title: "AG News Classification: TF–IDF Baselines, BERT, and a Paired Evaluation Audit"
author:
  - George Liu
  - Gordon Zau
  - Louis Dong
  - Zhiqi Zhou
date: "Research artifact revision · October 10, 2026"
fontsize: 10pt
geometry: margin=0.85in
colorlinks: true
linkcolor: blue
urlcolor: blue
---

# Abstract

We examine a four-model AG News classification study and audit the saved predictions of its two strongest models. The historical test summaries report 25.00% accuracy for a most-frequent baseline, 90.24% for Multinomial Naive Bayes, 91.80% for TF–IDF Logistic Regression (LR), and 94.25% for a fine-tuned BERT encoder. The 7,600 saved LR/BERT prediction pairs reproduce their reported scores and show 301 BERT-only correct examples versus 115 LR-only correct examples, a net gain of 186. A paired bootstrap estimates a 95% interval of 1.92–2.97 percentage points for the accuracy difference, conditional on these fixed models and test data. Directional confusion reductions show gains across several topic pairs; Business/Sci-Tech accounts for 40 net improvements, rather than a majority of the total. We release analysis and training/inference tools, test coverage and editable report sources. The audit does not independently reproduce historical training, quantify variation across seeds, or establish a causal explanation for BERT's advantage.

# 1. Research question and scope

Short-text topic classification provides a useful setting for comparing sparse lexical representations with pretrained contextual encoders. We ask how much BERT improves over TF–IDF classifiers in this particular AG News experiment, and which saved test predictions change. BERT's contextual architecture motivates hypotheses about vocabulary overlap [@devlin2019], but descriptive errors alone cannot establish which features caused a decision.

This is a revision of a course-project research artifact, not a claim of state-of-the-art performance or a peer-reviewed publication. The reported full-model scores are historical results. The additional paired statistics in this revision use historical saved predictions. A separate current-environment classical rerun is reported below. Hardware and historical run metadata are incomplete.

# 2. Dataset and experimental protocol

AG News [@zhang2015] provides four labels: World (0), Sports (1), Business (2), and Sci/Tech (3). Inputs concatenate a news headline and description; referring to their length as headline length alone is imprecise.

| Split | Examples | Role |
|:--|--:|:--|
| Training | 108,000 | Fit feature extractor and models |
| Development | 12,000 | Evaluate fixed baselines; select BERT checkpoint |
| Official test | 7,600 | Report final evaluation |

The scripts divide the official 120,000-example training split 90/10, stratified by label, with seed 42 by default. Each test class contains 1,900 examples. TF–IDF is fit only on training text. The supplied code implements fixed classical hyperparameters and does not substantiate the original presentation's claim of a classical hyperparameter search. For BERT, the lowest development-loss checkpoint is restored after training. The historical test-access history cannot be independently established from the artifacts alone.

## 2.1 Models

The mode baseline predicts the most frequent training label, with ties resolved by the first encountered label; it does not use TF–IDF. The classical models use scikit-learn [@pedregosa2011] TF–IDF unigrams and bigrams, lowercasing, English stop-word removal and a 20,000-feature cap. Multinomial Naive Bayes uses default smoothing; LR uses default L2 regularization and `max_iter=2000`.

The BERT script uses `bert-base-uncased` with four output classes through Hugging Face Transformers [@wolf2020]. The documented settings are a 128-token limit, batch size 8, two epochs, learning rate $2\times10^{-5}$, AdamW [@loshchilov2019], and weight decay 0.01. Checkpoint selection uses development loss. The current script sets random seeds before constructing the model and saves the final selected model and tokenizer. These current implementation improvements do not reconstruct the original training state.

# 3. Evaluation and statistical methods

We report accuracy and macro-averaged precision, recall and F1 with zero-division values set to zero. Macro recall equals accuracy for this balanced test set; macro precision and macro F1 need not equal their micro counterparts.

Saved LR and BERT files each contain 7,600 nonmissing predictions with labels 0–3. Their texts and gold labels agree in row order. The audit validates this correspondence before any comparison. New outputs contain example IDs; new analyses align by unique IDs and verify text and gold labels. Legacy files must have identical order. A join on text alone is unsafe when repeated texts occur and is not used in the revised code.

For fixed models, let $b$ denote LR-wrong/BERT-correct examples and $c$ BERT-wrong/LR-correct examples. The observed accuracy difference is $(b-c)/N$. Exact two-sided McNemar testing uses the binomial distribution with success probability 0.5 over the $b+c$ discordant correctness pairs. The percentile bootstrap resamples paired examples through equivalent multinomial draws over the four correctness outcomes (10,000 resamples, seed 42).

These analyses are post hoc and conditional on the saved models. Exchangeability of examples is assumed; related news stories may weaken that assumption. The interval does not quantify training-seed variation, external-data uncertainty, multiple exploratory comparisons or domain shift.

# 4. Results

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|:--|--:|--:|--:|--:|
| Most frequent | 0.2500 | 0.0625 | 0.2500 | 0.1000 |
| Multinomial NB | 0.9024 | 0.9019 | 0.9024 | 0.9019 |
| Logistic Regression | 0.9180 | 0.9178 | 0.9180 | 0.9178 |
| BERT | 0.9425 | 0.9424 | 0.9425 | 0.9424 |

The LR and BERT metrics are recomputed from saved predictions. The other rows reproduce supplied historical summary tables. No full training rerun is implied. BERT's accuracy advantage over LR is 2.447 percentage points; LR remains a competitive baseline on this dataset.

| Paired correctness outcome | Count | Share |
|:--|--:|--:|
| Both correct | 6,862 | 90.29% |
| LR wrong / BERT correct | 301 | 3.96% |
| BERT wrong / LR correct | 115 | 1.51% |
| Both wrong | 322 | 4.24% |

The net gain is exactly $301-115=186$ examples, accounting for the entire accuracy difference. The paired bootstrap interval is 1.921–2.974 percentage points. Exact McNemar testing yields $p=2.71\times10^{-20}$. This supports a difference between the fixed saved classifiers on this test sample under the stated assumptions; it does not establish general superiority across training runs or deployments.

## 4.1 Gross corrections and net gains

Business/Sci-Tech comprises 100 of 301 gross BERT corrections (56 Sci/Tech-to-Business and 44 Business-to-Sci/Tech LR errors corrected by BERT). However, BERT also introduces errors, and gross corrections must be distinguished from net changes.

| Gold-to-predicted direction | LR errors | BERT errors | Net reduction |
|:--|--:|--:|--:|
| Business to Sci/Tech | 147 | 139 | 8 |
| Sci/Tech to Business | 118 | 86 | 32 |
| World to Business | 69 | 47 | 22 |
| Business to World | 64 | 33 | 31 |
| World to Sports | 55 | 23 | 32 |
| Sports to World | 17 | 7 | 10 |

Business/Sci-Tech therefore accounts for 40 of 186 net additional correct predictions (21.5%). World/Business accounts for 53, and World/Sports for 42. Improvements are distributed across topic pairs; the original claim that most net gains concentrate on Business/Sci-Tech is unsupported.

| True class | LR correct | BERT correct | Net gain |
|:--|--:|--:|--:|
| World | 1,730 | 1,802 | 72 |
| Sports | 1,864 | 1,883 | 19 |
| Business | 1,674 | 1,718 | 44 |
| Sci/Tech | 1,709 | 1,760 | 51 |

Sports is near ceiling in both models. World contributes the largest absolute per-class gain. Business/Sci-Tech remains a frequent confusion direction for both classifiers.

## 4.2 Input length and interpretive limits

| Partition | Examples | Median words | Mean words |
|:--|--:|--:|--:|
| All test examples | 7,600 | 37 | 37.72 |
| BERT correct | 7,163 | 37 | 37.79 |
| BERT wrong | 437 | 36 | 36.58 |
| LR wrong / BERT correct | 301 | 35 | 36.87 |

Lengths count whitespace-separated words in the combined input. Similar medians are a limited descriptive check. They neither rule out length interactions nor identify contextual reasoning as the cause of improved accuracy. That would require length-stratified performance, truncation analysis and controlled interventions. Likewise, apparent topic ambiguity is not evidence of annotation error without a separate annotation study.

# 5. Reproducibility and software release

Code and artifacts are available in the [project repository](https://github.com/kechavious/News-Topic-Classification-Using-Machine-Learning-and-Transformer-Based-Models). The repository contains historical prediction tables, summaries, confusion matrices and figures. The revised audit records input SHA-256 hashes and writes its statistics, length summaries, class gains and confusion reductions to machine-readable files. Training commands now require fresh output directories, save seed/configuration/package metadata and split fingerprints, and preserve published artifacts. LR/Naive Bayes pipelines and selected BERT models can be saved and used for new-text inference. Inference probabilities have not been calibrated.

Offline tests cover malformed labels, ID alignment, repeated text, exact statistical cases, deterministic sampling, published prediction totals, and small end-to-end classical and BERT training/save/reload/inference workflows. Tiny BERT tests validate execution, not the accuracy of a pretrained encoder.

The original fine-tuned BERT checkpoint, original run configuration JSONs, precise hardware information and immutable dataset/model revisions are absent. Consequently, saved-prediction evaluation is auditable, while historical training replication remains unverified. Current dependency pins and the tested Linux/Python 3.12 CPU environment are documented separately from historical metadata. The code is MIT-licensed; upstream news text and models retain their own reuse terms.

## 5.1 Current-environment replication check

A separate full classical training/evaluation run completed in the tested CPU environment with dataset revision `eb185aade064a813bc0b7f42de02595523103ca4`. Its test accuracy was 25.00% for the mode baseline, 90.16% for Naive Bayes and 91.88% for LR. These are close to, but not identical with, the historical scores. The 7,600 test texts and labels match the historical LR file exactly; 74 LR predictions differ. Original training metadata is insufficient to identify the source of this discrepancy. The rerun's configuration, summaries, predicted labels and text hashes are released separately, preserving historical artifacts.

The actual pretrained BERT tokenizer and safetensors weights (model revision `86b5e0934494bd15c9632b12f734a8a67f723594`) were loaded, and a CPU optimizer step on training examples completed with finite loss. This is an execution check, not a rerun of the full fine-tuning experiment or its reported accuracy.

# 6. Limitations and next experiments

This study contains one saved run per model and no training-seed variance estimate. It does not provide a controlled ablation of contextual representations, a matched hyperparameter-search budget, a new-domain evaluation, calibrated uncertainty or measured training/inference cost. The test examples come from a balanced, short-text benchmark, so deployment behavior is not established.

Useful extensions are to freeze external revisions, evaluate multiple independent seeds, tune baselines on development data under a specified budget, measure resource use on named hardware, evaluate length bins and truncation rates, test a separate news distribution, and use controlled text changes or blinded annotations to assess causal hypotheses. These are proposed experiments, not completed results.

# 7. Conclusion

The saved predictions support a 2.45-percentage-point accuracy advantage for BERT over LR on this AG News test set. Paired analysis quantifies the difference and reveals gains across several topic pairs. Careful separation of observed results, post-hoc uncertainty and untested explanations makes this repository useful as a transparent research artifact without overstating reproducibility or mechanism.

# References
