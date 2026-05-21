---
type: source
created: 2026-04-27
updated: 2026-04-27
tags: [source, pivot-based, neural-uda, pre-bert, adversarial, naacl]
url: "https://aclanthology.org/N19-1039"
authors: [Timothy Miller]
year: 2019
venue: NAACL 2019
key_claim: Removing manually-selected pivot features and using joint end-to-end training achieves competitive neural UDA without SCL's brittle heuristics.
methodology: BiLSTM + adversarial domain discriminator; joint optimization of representation + task learners; no pivot feature selection
sources: []
---

# Simplified Neural Unsupervised Domain Adaptation (Miller, NAACL 2019)

## Key Claim
Prior neural UDA methods (e.g., DANN variants) required manually-selected pivot features (from SCL/SFA lineage). This work shows that joint end-to-end training subsumes pivot selection — achieving competitive results without the brittle, task-specific heuristic.

## Methodology
- **Architecture**: BiLSTM encoder + task classifier + domain discriminator (GRL-based adversarial)
- **No pivot features**: unlike SCL and its neural successors, no manual feature selection step
- **Joint training**: representation learning and task learning co-optimized end-to-end
- **Pre-BERT** era: static or simple contextual embeddings as input

## Key Findings
- Joint training eliminates the need for pivot features while matching SOTA (2019) performance
- Simpler architecture is more reproducible and generalizable
- Confirms DANN-style adversarial training works even without pivot feature scaffolding
- Historical significance: bridges SCL-era methods to modern end-to-end neural DA

## Historical Context
- Published 2019, immediately before BERT-era DA methods took over
- Represents the peak of pre-BERT neural UDA simplification
- [[Don't Stop Pretraining (Gururangan ACL2020)]] and [[Adversarial and Domain-Aware BERT (Guo ACL2020)]] effectively supersede this with PLM backbones

## Connections
- [[Pivot-Based Methods for NLP DA]] — directly extends/simplifies this lineage
- [[DANN (Domain-Adversarial Neural Networks)]] — underlying adversarial mechanism
- [[DANN for NLP Text DA]] — pre-BERT era context
