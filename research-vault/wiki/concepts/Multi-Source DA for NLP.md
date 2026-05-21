---
type: concept
status: developing
created: 2026-04-27
updated: 2026-04-27
tags: [concept, nlp, multi-source, domain-adaptation, mixture-of-experts, adversarial]
sources:
  - "[[Transformer Multi-Source DA (Wright EMNLP2020)]]"
  - "[[Neural UDA in NLP Survey (Ramponi COLING2020)]]"
---

# Multi-Source DA for NLP

Domain adaptation from **multiple labeled source domains** to a single unlabeled target domain. Distinct from standard (single-source) UDA because: (1) the model must aggregate knowledge across heterogeneous sources, and (2) the domain alignment problem is a one-to-many alignment.

## Core Approaches

### Mixture of Experts
Each source domain trains a separate expert model. At inference, expert predictions are mixed using a learned or heuristic weighting:
- **Uniform mixture**: average all expert outputs
- **Distance-weighted**: weight by source-target domain similarity
- **Attention-based**: learn a mixing attention over source expert outputs given target input

### Multi-Domain Adversarial Training
A single domain discriminator is trained to distinguish the target domain from all source domains simultaneously. GRL is applied across all domain pairs.

### Source Selection / Domain Similarity Filtering
Select the single most similar source domain (or a weighted subset) before applying standard single-source UDA methods. Simple but effective in practice.

## Transformer-Era Findings

> [!key-insight] Transformers reduce the multi-source DA gap
> Wright & Augenstein (EMNLP 2020) find that transformer-based domain experts produce **highly homogeneous predictions** — the mixing signal that was useful for CNN/RNN experts is weak for BERT-based models. Pretrained transformers are already partially domain-invariant, so the incremental benefit of multi-source ensemble is smaller than in the BiLSTM era.

Key findings from [[Transformer Multi-Source DA (Wright EMNLP2020)]]:
- Mixture of experts still outperforms single-source baselines, but margins are smaller than with RNNs
- Domain adversarial training across all sources still provides benefit
- Attention-based mixing is only marginally better than simpler alternatives
- Transformers' pretraining provides a strong domain-agnostic prior

## Relationship to Other Methods

**vs. Single-source UDA**: multi-source has access to more labeled data but must handle heterogeneous source distributions. In the PLM era, the additional sources help less because BERT already generalizes.

**vs. Domain generalization**: multi-source DA uses a specific target domain (unlabeled); domain generalization assumes no target domain access at all.

**vs. Multi-task learning**: multi-task learns one model across all domains jointly; multi-source DA specifically optimizes for target domain performance.

## When Multi-Source Matters Most

- Many labeled source domains available with similar task
- Target domain is distant from any individual source but covered by the union
- Pre-BERT / limited pretrained models: ensemble gains are larger

## Open Questions

- Do DAPT + multi-source adversarial combine additively?
- Does adapter-per-source (vs. shared adapter) improve mixture-of-experts for BERT?
- Is there a principled source selection strategy for PLM-based multi-source DA?

## Related

[[DANN (Domain-Adversarial Neural Networks)]], [[DANN for NLP Text Domain Adaptation]], [[NLP UDA Method Combinations]], [[Transformer Multi-Source DA (Wright EMNLP2020)]], [[Adapters (NLP)]]
