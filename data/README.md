# Data

AG News has four classes: World (0), Sports (1), Business (2), Sci/Tech (3). Training scripts retrieve the public `fancyzhx/ag_news` dataset using Hugging Face Datasets. The official training split has 120,000 examples; the test split has 7,600. The training scripts reserve 10% of training for development, stratified by label with seed 42 by default.

Inputs combine a headline and description. Word-length statistics describe this combined text, not isolated headlines.

Dataset reference: Xiang Zhang, Junbo Zhao and Yann LeCun (2015), [Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626). [Original distribution](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv) · [Dataset card](https://huggingface.co/datasets/fancyzhx/ag_news).

The repository's MIT code license does not grant rights to upstream news text or model weights. The Hugging Face dataset card inspected on October 10, 2026 labels its license `unknown`; this repository does not establish permission to redistribute the source news text. Check the original data distribution and current dataset card for their reuse terms. Saved prediction/error CSVs include news text and should be treated as derived dataset artifacts.

New runs can specify `--dataset-revision` to freeze the input version. Historical results do not record that revision; see [provenance](../docs/REPRODUCIBILITY.md).
