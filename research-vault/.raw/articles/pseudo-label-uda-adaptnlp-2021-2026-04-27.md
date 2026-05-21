---
source_url: https://aclanthology.org/2021.adaptnlp-1.2.pdf
fetched: 2026-04-27
---

# Pseudo-Label Guided Unsupervised Domain Adaptation of Contextual Embeddings

**Authors:** Tianyu Chen, Shaohan Huang, Furu Wei, Jianxin Li
**Year:** 2021
**Venue:** AdaptNLP Workshop @ EACL 2021

Leverages pseudo-labels to guide adaptation of pre-trained contextual embeddings (BERT) to a target domain without labeled target data. Iteratively generates pseudo-labels from source-trained model predictions on target data, then fine-tunes the model using these pseudo-labels. Low-resource UDA scenario. Builds on self-training loop but specifically frames it as guiding embedding adaptation, not just classifier adaptation.
