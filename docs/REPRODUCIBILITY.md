# Reproducibility and artifact provenance

## What can be reproduced now?

The repository provides historical test summaries, confusion matrices, LR/BERT prediction CSVs and error tables. `src/research_analysis.py` recomputes LR/BERT macro metrics, paired outcomes, exact McNemar testing, a paired bootstrap interval, per-class gains, directional confusion reductions and whitespace length summaries without Internet access. Input SHA-256 hashes are embedded in `results/analysis/paired_statistics.json`.

The historical LR and BERT files each contain 7,600 rows. Their texts and gold labels match in row order, with 1,900 examples per class and no duplicate text/gold keys. New runs save `example_id`, representing the position within the official test split. Pairing verifies IDs, text and gold labels; historical files require identical row order. A text join is deliberately avoided because repeated text can multiply observations in other datasets.

## Historical versus current runs

The historical BERT training was not rerun in full during this revision. A full classical rerun completed, with close but nonidentical results: Naive Bayes 90.16% and LR 91.88% accuracy versus historical 90.24% and 91.80%. Test texts and labels match; LR predictions differ on 74 examples. Its configuration and labels are in `results/replication/`. The reason for the discrepancy is unresolved because original run metadata is missing. Neither the fine-tuned BERT checkpoint nor original experiment-configuration JSONs are present. The original PDF names Python 3.10 and a consumer GPU, but the actual GPU model, training time, model/data revision IDs, full dependency resolution and exact run logs are unavailable. Those details cannot be reconstructed from predictions.

Current training records seeds, split fingerprints, requested external revisions, resolved model revision when available, and installed package versions. Use immutable Hugging Face revision IDs for replication. Record hardware, operating system, commands, wall-clock time and the Git commit alongside each run. Seeds improve repeatability but do not guarantee identical results across hardware, library versions and GPU kernels.

The tested development runtime is Linux/Python 3.12 with PyTorch 2.4.1+cpu. `requirements.lock.txt` is its dependency snapshot. It is not evidence that historical models used those exact transitive dependencies. CPU model tensors and optimizer behavior are checked with local integration tests. The real pretrained BERT weights and tokenizer were also loaded, and one optimizer step on training data passed; full BERT fine-tuning is a separate experiment.

## Statistical analysis

The accuracy difference is `(BERT-only correct - LR-only correct) / N`. Exact two-sided McNemar testing uses a binomial null with probability 0.5 over discordant correctness pairs. When there are no discordances, p is 1. The paired bootstrap resamples the empirical four-case distribution 10,000 times (seed 42); multinomial draws are equivalent to sampling paired test observations for this statistic. The interval is the 2.5th–97.5th percentile of the accuracy differences.

This is a post-hoc audit of fixed saved models. The uncertainty estimate assumes exchangeable test examples; it does not account for related news stories, repeated training runs, multiple exploratory comparisons or model selection. Report repeated seeds and cluster-aware analyses before making stronger generalization claims.

## Model selection and claims

The provided scripts implement a fixed classical configuration and select the BERT checkpoint by development loss. They do not implement the classical hyperparameter tuning claimed in some original prose. The mode baseline does not use TF–IDF features. Similar length medians do not exclude length effects. Error examples do not establish that BERT used context rather than memorized entity associations or other signals. No measured latency, resource use or carbon-emission claims are supported.

## Useful next experiments

1. Freeze dataset/model revisions; run at least five independent model seeds with a fixed split and report mean, standard deviation and all runs.
2. Tune classical regularization and feature settings on development data under a documented budget; use an untouched test set for final comparisons.
3. Measure training time, peak memory and inference throughput on specified hardware.
4. Evaluate length bins, truncation rate, confidence calibration and a separate news distribution.
5. Use counterfactual text changes or blinded annotation to test hypotheses about context and label ambiguity.

These are proposed experiments, not completed results.
