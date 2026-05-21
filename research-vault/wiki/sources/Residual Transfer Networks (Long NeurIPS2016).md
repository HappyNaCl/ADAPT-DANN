---
type: source
created: 2026-04-27
updated: 2026-04-27
tags: [source, discrepancy-minimization, mmd, rkhs, distribution-matching, neurips, cv-origin]
url: "https://proceedings.neurips.cc/paper_files/paper/2016/file/ac627ab1ccbdb62ec96e702f07f6425b-Paper.pdf"
authors: [Mingsheng Long, Han Zhu, Jianmin Wang, Michael I. Jordan]
year: 2016
venue: NeurIPS 2016
key_claim: Residual layers that explicitly model classifier differences, combined with RKHS-based multi-layer feature matching, improve domain adaptation over shared-classifier approaches.
methodology: Residual Transfer Networks (RTN); tensor product fusion of multi-layer features in RKHS; trainable end-to-end
sources: []
---

# Unsupervised Domain Adaptation with Residual Transfer Networks (Long et al., NeurIPS 2016)

> [!warning] CV Paper
> This paper is **primarily a computer vision paper**, evaluated on image classification benchmarks (Office-31, Caltech-256). It is ingested here because RTN's RKHS-based distribution matching methodology is domain-agnostic and cited in NLP discrepancy minimization literature. Do not cite its benchmark numbers in NLP contexts.

## Key Claim
Domain adaptation improves when you (1) do not assume classifiers are shared across domains (use residual functions to model classifier gap) and (2) match multi-layer feature distributions using tensor product fusion in RKHS (a generalization of MMD).

## Methodology
- **Residual Transfer Networks (RTN)**: residual layers on top of shared deep network learn domain-specific classifier adjustments
- **RKHS matching**: tensor product of multi-layer features mapped into reproducing kernel Hilbert space; matches joint feature-classifier distributions
- **End-to-end**: trainable via standard backpropagation
- **Comparison**: outperforms DDC (single-layer MMD) and DAN (multi-layer MMD without residuals)

## NLP Relevance
- RKHS/MMD-based distribution matching is the **non-adversarial baseline** in NLP DA
- RTN's residual approach for classifier mismatch is conceptually related to task-specific adapter modules in NLP
- Cited in [[Discrepancy Minimization for NLP DA]] as foundational MMD extension
- Tensor product fusion idea appears in later NLP multi-task and DA work

## Connections
- [[Discrepancy Minimization for NLP DA]] — RKHS/MMD methodology context
- [[Alignment-Discriminability Tradeoff]] — residual classifier modeling addresses this
- [[DANN (Domain-Adversarial Neural Networks)]] — adversarial vs. discrepancy-based DA comparison
