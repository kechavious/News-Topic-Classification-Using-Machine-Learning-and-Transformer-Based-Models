---
title: "AG News: TF–IDF and BERT"
subtitle: "A paired evaluation audit"
author: "George Liu · Gordon Zau · Louis Dong · Zhiqi Zhou"
date: "Research artifact revision · October 10, 2026"
theme: Boadilla
colortheme: seahorse
aspectratio: 169
fontsize: 10pt
---

# Question and scope

- How much does BERT improve over TF–IDF Logistic Regression in this experiment?
- Which predictions change on the same test examples?
- Single-run academic artifact; no state-of-the-art or peer-review claim.
- Paired statistics audit historical predictions; full BERT fine-tuning was not rerun.

# Experimental protocol

| Split | Examples | Use |
| --- | ---: | --- |
| Train | 108,000 | Fit models/features |
| Development | 12,000 | Evaluate baselines; select BERT checkpoint |
| Test | 7,600 | Final evaluation |

Four balanced labels: World, Sports, Business, Sci/Tech. Stratified 90/10 train/development split, seed 42. Inputs include a headline and description.

# Models

- Mode baseline: training labels only.
- Multinomial NB and LR: TF–IDF unigrams/bigrams, English stop words, 20,000 features; fixed settings.
- BERT: bert-base-uncased, 128 tokens, batch size 8, two epochs, AdamW, learning rate 2e-5, weight decay 0.01.
- BERT checkpoint selected by development loss.

# Historical test scores

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Most frequent | 25.00% | 10.00% |
| Multinomial NB | 90.24% | 90.19% |
| Logistic Regression | 91.80% | 91.78% |
| BERT | 94.25% | 94.24% |

LR and BERT scores recomputed from saved predictions. Other rows are historical summaries. No measured compute-cost comparison is available.

# Paired outcomes

| Outcome | Count |
| --- | ---: |
| Both correct | 6,862 |
| LR wrong / BERT correct | 301 |
| BERT wrong / LR correct | 115 |
| Both wrong | 322 |

Net gain: **186 examples**, or **2.45 percentage points**.

# Conditional statistical evidence

- Paired bootstrap 95% interval: **1.92–2.97 percentage points**.
- 10,000 paired resamples; seed 42.
- Exact two-sided McNemar p: $2.71 \times 10^{-20}$.
- Conditional on fixed saved models and exchangeable test examples.
- Does not measure variation across training seeds or prove a causal mechanism.

# Gross corrections are not net gains

- Business/Sci-Tech: 100 of 301 BERT-only corrections.
- Net Business/Sci-Tech error reduction: **40 of 186** (21.5%).
- Net World/Business reduction: 53.
- Net World/Sports reduction: 42.
- Gains are spread across topic pairs; most do not concentrate in Business/Sci-Tech.

# Class-level gains

| Gold class | LR correct | BERT correct | Gain |
| --- | ---: | ---: | ---: |
| World | 1,730 | 1,802 | 72 |
| Sports | 1,864 | 1,883 | 19 |
| Business | 1,674 | 1,718 | 44 |
| Sci/Tech | 1,709 | 1,760 | 51 |

Each class has 1,900 examples. World has the largest absolute gain; Sports is close to ceiling.

# Length is descriptive, not causal evidence

| Partition | Median words |
| --- | ---: |
| All inputs | 37 |
| BERT correct | 37 |
| BERT wrong | 36 |
| LR wrong / BERT correct | 35 |

Similar medians cannot rule out length effects or establish context understanding. Analyze length bins, truncation and controlled text changes next.

# Research software improvements

- Offline audit: metrics, paired statistics, input hashes and descriptive CSVs.
- Strict pairing; stable example IDs for new runs.
- Fresh output directories preserve historical artifacts.
- Save/reload classical pipelines and BERT; JSON headline predictions.
- Offline unit and training/inference integration tests; CI configuration.
- Editable report/slides and document build script.

# Current replication check

- Full classical rerun: NB **90.16%**, LR **91.88%** test accuracy.
- Historical values: NB 90.24%, LR 91.80%; test texts/labels match.
- LR predictions differ on 74 examples; original metadata is insufficient to explain why.
- Real pretrained BERT loaded and completed one optimizer step on training data.
- These checks validate execution; they do not reproduce the full historical BERT experiment.

# Limitations and next experiments

- Missing historical checkpoint, run configs, hardware and immutable revisions.
- One run per model; no seed-variance estimate.
- No domain shift, calibration, controlled ablation or measured resource benchmark.
- Next: multiple seeds, fixed revisions, documented tuning budgets, new-domain evaluation and causal probes.

# Takeaway

The saved predictions support a **2.45-point accuracy gain** for BERT on this test set. The paired audit makes the evidence inspectable. Stronger claims about generalization, cost and mechanism require new experiments.

Dataset: Zhang et al. (2015). BERT: Devlin et al. (2019). Full references, methods and caveats appear in the revised report.
