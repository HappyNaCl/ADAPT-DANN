---
type: source
created: 2026-04-27
updated: 2026-04-27
tags: [source, pseudo-labels, self-training, bert, adaptnlp, contextual-embeddings]
url: "https://aclanthology.org/2021.adaptnlp-1.2"
authors: [Tianyu Chen, Shaohan Huang, Furu Wei, Jianxin Li]
year: 2021
venue: AdaptNLP Workshop @ EACL 2021
key_claim: Pseudo-labels from source-trained models can guide iterative BERT adaptation to a target domain without any labeled target data.
methodology: Generate pseudo-labels from source model predictions on target data; iteratively fine-tune BERT using pseudo-labels; repeat until convergence
sources: []
---

# Pseudo-Label Guided UDA of Contextual Embeddings (Chen et al., AdaptNLP 2021)

## Key Claim
Pseudo-labels generated from source-domain classifier predictions on target data can drive iterative adaptation of BERT contextual embeddings, closing the domain gap without requiring labeled target data.

## Methodology
- Source model (BERT fine-tuned on source) generates pseudo-labels for target domain instances
- BERT is fine-tuned on these pseudo-labeled target instances
- Process iterates: updated model generates better pseudo-labels → better fine-tuning
- Thresholding or confidence filtering applied to reduce pseudo-label noise

## Key Findings
- Iterative pseudo-label refinement improves over one-shot self-training
- Confidence filtering is important — low-confidence pseudo-labels degrade performance
- Contextual embeddings (BERT) are more amenable to pseudo-label adaptation than static embeddings
- Complements DAPT-style adaptation: pseudo-labels guide the task head, not just the language model

## Failure Modes
- Noisy pseudo-labels in the first iteration can compound (error accumulation)
- Performance degrades if source and target domains are very dissimilar (poor initial pseudo-labels)
- See [[Self-Training for NLP DA]] for general failure modes

## Connections
- [[Self-Training for NLP DA]] — primary instantiation; extends with contextual embedding focus
- [[Domain-Adaptive Pre-Training (DAPT)]] — complementary mechanism (different objective)
- [[BERT]] — backbone model adapted
