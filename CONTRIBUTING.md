# Contributing

Use a fresh output directory for each experiment and leave historical `results/csv/` artifacts intact. Include commands, model/data revision IDs, seed, hardware and package versions with new results. Use development data for model selection; do not tune against the test set.

Install `requirements-dev.txt` for the tested dependencies and Ruff. Run `ruff check src tests` and `ruff format --check src tests`. Run `python -m unittest discover -s tests -v` before submitting code changes. Tests run offline and include small classical and BERT training/inference checks. New evaluation code should reject misaligned examples and malformed labels. Statistical or manuscript changes must identify the source artifacts and distinguish observations from hypotheses.

For a bug report, provide the command, Python/dependency versions and traceback. For a research result, provide the full configuration and predictions needed to audit it. Do not commit model checkpoints or caches; use a model registry/release asset with documented provenance if distributing a trained model.
