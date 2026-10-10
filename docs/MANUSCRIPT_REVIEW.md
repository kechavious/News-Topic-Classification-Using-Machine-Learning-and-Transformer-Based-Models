# Manuscript and presentation review

The original PDFs are retained unchanged as historical course-project artifacts. `writeup/paper.md` and `presentation/slides.md` are the editable revised sources; their PDFs are rebuilt with `scripts/build_documents.sh`.

| Original issue | Revision |
| --- | --- |
| Claimed a fully reproducible release with all configurations | Distinguish saved-prediction evaluation from full training replication; original run configs/checkpoint are missing |
| Development data described as used for hyperparameter tuning | State the actual code protocol: fixed classical settings, best BERT development loss |
| Most-frequent model described as operating on TF–IDF | Clarify that it uses training labels only |
| Gross BERT corrections treated as where net gains concentrate | Report both gross corrections and net directional error reductions |
| Similar medians used to rule out length as an explanation | Treat lengths as descriptive, without causal inference |
| Single-run score gaps called lower bounds on variance-aware comparison | Remove: a single-run gap is not a statistical lower bound |
| Context understanding and annotation ambiguity asserted as established | Present as hypotheses; no controlled intervention or annotation study |
| Quantified compute/deployment claims without measurements | Remove unsupported timing and deployment superiority claims |
| README/slides reported 57 both-wrong Sci-Tech/World errors | Original paper's 50 matches a BERT-prediction-based pair count; new artifacts emphasize clearly defined measures |
| Error sample requested 100 but generated 97 | Preserve row indices through selection and test exact sample size |
| PDFs had no editable source in checkout | Add source documents and a reproducible PDF build command |

The original claimed institutional affiliation and detailed author-contribution statement were not independently verified. The revised report retains the four listed authors and avoids adding new affiliations or individual responsibility claims. No venue acceptance, DOI, peer review or state-of-the-art claim is implied.
