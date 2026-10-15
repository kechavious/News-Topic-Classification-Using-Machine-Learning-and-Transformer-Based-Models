# AG News Preprint — Phase 1 Research Protocol
Date: 2026-10-14
Status: Planning only; not submitted, accepted or peer-reviewed.

## Research objective
Produce a rigorous paired-error analysis of TF-IDF Logistic Regression and BERT on AG News. Evaluate reproducibility and training-seed stability before making claims of generality. No claim of a novel algorithm is made.

## Supported historical evidence
- 7,600 paired AG News test examples.
- Logistic Regression: 91.80% accuracy; BERT: 94.25%.
- Both correct: 6,862; LR wrong/BERT correct: 301; BERT wrong/LR correct: 115; both wrong: 322.
- Net gain: 186 examples, approximately 2.45 percentage points.
- Paired bootstrap 95% CI: approximately 1.92–2.97 percentage points, conditional on fixed predictions.
- Historical BERT checkpoint and original training metadata are not available; a full fine-tuning replication is still outstanding.
- Do not infer contextual reasoning or causal error mechanisms from example descriptions alone.

## Research questions
1. How do the two saved models differ on the same examples?
2. Which directional confusions drive *net* gains instead of gross corrections?
3. Do observations replicate with fresh runs across several seeds?
4. Optionally, can independently annotated disagreement categories support a stronger descriptive claim?

## Immediate run checklist
Run in a clean environment and preserve stdout/stderr as logs:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1
pip install -r requirements.txt
pip check
python -m unittest discover -s tests -v
python src/research_analysis.py --output-dir runs/preprint-audit
python src/overlap_analysis.py --output-dir runs/preprint-overlap
```
Never overwrite original `results/` files. This checklist has **not yet been executed** by this research branch.

## Minimum evidence before a public preprint
- Independently rerun baseline and full BERT training with logged hardware, revisions, seeds and checkpoints.
- Validate exact example alignment; separate development tuning from held-out evaluation.
- Replicate across multiple random seeds and report variability.
- Compare descriptive error results across reruns; acknowledge negative or inconsistent findings.
- Audit every table against saved CSVs and experiment manifests.
- Check relevant literature; state the contribution narrowly and do not imply novelty without evidence.
- Confirm authorship with all contributors to the original four-author course project.
- Resolve inconsistent author name spelling (Gordon Zau vs Gordon Zou).
- Apply repository/preprint disclosure and anonymization rules for any future venue.

## Proposed paper structure
Abstract; Introduction; Related Work; Data and Experimental Design; Paired Error Methodology; Results; Reproduction Audit; Limitations; Ethics and Authorship; References.

## Milestone definitions
M1: historical statistics and tests rerun with archived logs.
M2: new training results reproducibly saved.
M3: literature and new analysis audited.
M4: coauthors approve accurate attribution and manuscript.
M5: preprint is ready for scholarly review (not automatically arXiv-eligible or accepted).
