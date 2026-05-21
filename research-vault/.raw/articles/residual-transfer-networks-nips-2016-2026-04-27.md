---
source_url: https://proceedings.neurips.cc/paper_files/paper/2016/file/ac627ab1ccbdb62ec96e702f07f6425b-Paper.pdf
fetched: 2026-04-27
---

# Unsupervised Domain Adaptation with Residual Transfer Networks

**Authors:** Mingsheng Long, Han Zhu, Jianmin Wang, Michael I. Jordan
**Year:** 2016
**Venue:** NeurIPS 2016

NOTE: Primarily a computer vision paper evaluated on image benchmarks (Office-31, etc.). Ingested for its discrepancy minimization methodology, which is domain-agnostic and cited in NLP DA literature.

Introduces Residual Transfer Networks (RTN): residual layers learn domain-specific classifier adaptations on top of shared feature networks. Uses tensor product fusion of multi-layer features in Reproducing Kernel Hilbert Spaces (RKHS) to match feature distributions across domains. Trainable end-to-end via backprop. Shows superior performance vs. DDC and DAN. Key NLP relevance: RKHS-based MMD matching is a non-adversarial distribution alignment alternative used in NLP discrepancy minimization methods.
