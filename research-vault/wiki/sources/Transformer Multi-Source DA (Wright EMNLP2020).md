---
type: source
created: 2026-04-27
updated: 2026-04-27
tags: [source, multi-source-da, adversarial, transformers, mixture-of-experts, emnlp]
url: "https://aclanthology.org/2020.emnlp-main.639/"
authors: [Dustin Wright, Isabelle Augenstein]
year: 2020
venue: EMNLP 2020
key_claim: Transformer-based domain experts produce homogeneous predictions, making mixture-of-experts gains modest; adversarial training still helps in multi-source settings.
methodology: Mixture-of-experts + domain adversarial training on pretrained transformers; attention-based mixing metric
sources: []
---

# Transformer Based Multi-Source Domain Adaptation (Wright & Augenstein, EMNLP 2020)

## Key Claim
Techniques proven effective for CNN/RNN multi-source DA (mixture of experts, domain adversarial training) transfer to transformers but yield smaller gains, because pretrained transformers are already domain-robust. Adversarial training still provides benefit; combining predictions from domain-specific experts improves over single-source baselines.

## Methodology
- **Multi-source setup**: multiple labeled source domains, one unlabeled target domain
- **Mixture of experts**: per-domain transformer classifiers; predictions mixed at inference via learned metrics (including a novel attention-based metric)
- **Domain adversarial training**: GRL applied to transformer representations across all source domains simultaneously
- **Baseline**: single-source transformer fine-tuned directly on the aggregate source

## Key Findings
- Transformer domain experts produce **highly homogeneous predictions** — the mixing signal is weak compared to CNN/RNN era
- Adversarial training still helps but gains are smaller than with RNNs
- Attention-based mixing metric is proposed but only marginally outperforms simpler alternatives
- Transformers' pretraining provides a prior that subsumes some of what domain-specific expert ensembles used to provide

## Relevance to Vault
- Directly addresses **multi-source DA** — a coverage gap noted in the vault
- Complements [[DANN for NLP Text DA]] (single-source adversarial → multi-source extension)
- Connects to [[NLP UDA Method Combinations]]: multi-source + adversarial is a distinct combination pattern
- Supports the finding in [[Adversarial and Domain-Aware BERT (Guo ACL2020)]] that vanilla DANN on BERT needs modification

## Connections
- [[Multi-Source DA for NLP]] — primary source for this concept
- [[DANN (Domain-Adversarial Neural Networks)]] — technique applied
- [[NLP UDA Method Combinations]] — adds multi-source pattern
- [[Transformer Multi-Source DA (Wright EMNLP2020)]] ← this page
