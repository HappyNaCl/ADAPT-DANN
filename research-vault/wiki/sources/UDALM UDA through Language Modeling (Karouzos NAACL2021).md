---
type: source
created: 2026-04-27
updated: 2026-04-27
tags: [source, dapt, adversarial, language-modeling, sentiment, naacl]
url: "https://aclanthology.org/2021.naacl-main.203"
authors: [Constantinos Karouzos, Georgios Paraskevopoulos, Alexandros Potamianos]
year: 2021
venue: NAACL 2021
key_claim: Combining domain-adaptive language modeling (DAPT-style) with adversarial training is complementary — each addresses a distinct type of domain shift.
methodology: Continue MLM on target domain text; add adversarial domain discriminator on top; evaluate on cross-domain sentiment classification
sources: []
---

# UDALM: Unsupervised Domain Adaptation through Language Modeling (Karouzos et al., NAACL 2021)

## Key Claim
Domain-adaptive language modeling (continuing MLM on target domain) and adversarial feature alignment are complementary strategies. Stacking them — DAPT first, then adversarial fine-tuning — outperforms either alone on cross-domain sentiment classification.

## Methodology
- **LM adaptation**: continue pre-trained BERT MLM on unlabeled target domain text (identical to DAPT in [[Don't Stop Pretraining (Gururangan ACL2020)]])
- **Adversarial alignment**: domain discriminator with GRL applied after LM adaptation step
- **Task**: cross-domain sentiment classification (Amazon Reviews-style domains)
- **Evaluation**: standard multi-domain sentiment benchmarks

## Key Findings
- DAPT + adversarial > DAPT alone > adversarial alone
- Confirms **DAPT should precede adversarial fine-tuning** (DAPT → DANN order matters)
- Provides direct empirical evidence for the stacking hypothesis in [[Research - DAPT vs DANN and Combining Them]]
- Language modeling adapts token representations; adversarial training aligns higher-level features — different layers of the shift

## Relevance to Vault
- Strong empirical support for DAPT → DANN combination stack
- Fills the evidence gap in [[NLP UDA Method Combinations]] for explicit DAPT + adversarial combination
- Directly validates the proposed recipe in [[Research - DAPT Adapter DANN Full Stack]]

## Connections
- [[Domain-Adaptive Pre-Training (DAPT)]] — primary mechanism
- [[DANN (Domain-Adversarial Neural Networks)]] — adversarial component
- [[DANN for NLP Text DA]] — practical implementation context
- [[NLP UDA Method Combinations]] — DAPT+DANN combination evidence
- [[Research - DAPT vs DANN and Combining Them]] — supports key thesis
