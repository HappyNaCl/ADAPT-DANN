---
type: source
created: 2026-04-27
updated: 2026-04-27
tags: [source, adda, knowledge-distillation, bert, sentiment, arxiv]
url: "https://arxiv.org/abs/2010.11478"
authors: [Minho Ryu, Kichun Lee]
year: 2020
venue: arXiv 2020 (cs.CL)
key_claim: Combining ADDA adversarial alignment with knowledge distillation (AAD) stabilizes BERT domain adaptation and achieves SOTA on 30 cross-domain sentiment pairs.
methodology: AAD = ADDA decoupled encoder training + KD soft labels from source-adapted teacher; evaluated across 30 Amazon Review domain pairs
sources: []
---

# Knowledge Distillation for BERT Unsupervised Domain Adaptation (Ryu & Lee, arXiv 2020)

## Key Claim
ADDA adversarial training on BERT is unstable alone. Adding knowledge distillation (KD) — transferring soft probability outputs from a source-trained teacher to the target-domain student — provides a stable supervision signal that complements adversarial alignment. The resulting AAD system achieves SOTA on 30 sentiment domain pairs.

## Methodology
- **ADDA framework**: separate source and target encoders; GAN-style adversarial training (no GRL; discriminator trained separately)
- **Knowledge Distillation**: source-adapted BERT teacher provides soft label distributions; target student minimizes KL divergence from teacher
- **AAD (Adversarial Adaptation with Distillation)**: joint objective combining ADDA alignment loss + KD loss
- **Scale**: evaluated on 30 domain pairs from Amazon Reviews (cross-product of ~6 domains)

## Key Findings
- KD **stabilizes** ADDA training on BERT — without KD, ADDA alone is brittle on PLMs
- Soft teacher labels act as domain-invariant supervision, bridging source knowledge to target
- 30-pair evaluation provides strong statistical reliability (vs. single domain-pair papers)
- ADDA + KD > DANN + KD (decoupled encoder training helps with BERT)

## Relevance to Vault
- Fleshes out [[ADDA]] stub with concrete NLP application details
- Adds knowledge distillation as a stability mechanism for adversarial BERT DA
- Connects to [[Alignment-Discriminability Tradeoff]]: KD preserves class discriminability while ADDA aligns domains

## Connections
- [[ADDA]] — primary adversarial mechanism used
- [[DANN (Domain-Adversarial Neural Networks)]] — ADDA variant comparison
- [[BERT]] — backbone model
- [[Alignment-Discriminability Tradeoff]] — KD addresses discriminability side
- [[NLP UDA Method Combinations]] — ADDA + KD as a validated combination
