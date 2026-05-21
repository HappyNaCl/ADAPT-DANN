---
type: source
created: 2026-04-27
updated: 2026-04-27
tags: [source, self-training, self-adaptation, pre-bert, sentiment, ranlp]
url: "https://aclanthology.org/R19-1025"
authors: [Xia Cui, Danushka Bollegala]
year: 2019
venue: RANLP 2019
key_claim: Models can adapt to target domain by iteratively updating their own representations based on target domain characteristics, without adversarial training or labeled target data.
methodology: Self-adaptation mechanism — iterative representation update using target domain signals; pre-BERT; evaluated on sentiment classification
sources: []
---

# Self-Adaptation for Unsupervised Domain Adaptation (Cui & Bollegala, RANLP 2019)

## Key Claim
Rather than adversarial alignment or explicit distribution matching, models can self-adapt by iteratively adjusting representations toward the target domain distribution. This is a non-adversarial, self-supervised approach to UDA evaluated on sentiment classification.

## Methodology
- **Self-adaptation loop**: model generates target-domain signals (e.g., reconstruction error, domain-specific features) and uses them to update representations
- **Pre-BERT**: uses traditional or simple neural text representations
- **Task**: cross-domain sentiment classification
- **No adversarial component**: pure representation adaptation via iterative self-supervision

## Key Findings
- Self-adaptation (without adversarial training) is competitive with adversarial approaches in pre-BERT setting
- Iterative approach converges to reasonable target-domain representations
- Simpler and more stable than adversarial training at the cost of weaker alignment signal

## Historical Context
- Pre-BERT, pre-DAPT era (2019)
- Represents a self-training adjacent approach that influenced later pseudo-label methods
- Largely superseded by DAPT (which does self-supervised adaptation via MLM) and adversarial BERT methods

## Connections
- [[Self-Training for NLP DA]] — related paradigm; self-adaptation as a variant
- [[Pivot-Based Methods for NLP DA]] — historical peer (also pre-BERT era)
- [[Domain-Adaptive Pre-Training (DAPT)]] — DAPT can be seen as a principled, scalable version of the self-adaptation idea
