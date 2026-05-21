---
type: source
created: 2026-04-27
updated: 2026-04-27
tags: [source, dapt, sequence-labeling, bert, mlm, emnlp, early-dapt]
url: "https://aclanthology.org/D19-1433"
authors: [Xiaochuang Han, Jacob Eisenstein]
year: 2019
venue: EMNLP-IJCNLP 2019
key_claim: Domain-adaptive MLM fine-tuning of BERT on unlabeled target domain text substantially improves sequence labeling in linguistically distant domains, especially for OOV words.
methodology: Continue BERT MLM on target domain text (Early Modern English, Twitter); evaluate on NER/POS sequence labeling; compare BERT, ELMo, and domain-adapted variants
sources: []
---

# UDA of Contextualized Embeddings for Sequence Labeling (Han & Eisenstein, EMNLP 2019)

## Key Claim
Continuing masked language modeling on unlabeled target domain text (domain-adaptive fine-tuning) significantly improves BERT performance on sequence labeling tasks in linguistically distant target domains. Gains are especially large for out-of-vocabulary words not seen during BERT pretraining.

## Methodology
- **Domain-adaptive fine-tuning**: continue BERT MLM on unlabeled target domain text (same mechanism as [[Domain-Adaptive Pre-Training (DAPT)]])
- **Tasks**: Named Entity Recognition and POS tagging (sequence labeling)
- **Target domains**: Early Modern English (historical), Twitter (social media) — both highly OOV-heavy
- **Comparison**: base BERT, ELMo, domain-adapted BERT, domain-adapted ELMo

## Key Findings
- Domain-adaptive MLM fine-tuning yields **substantial improvements** over base BERT on both domains
- Especially effective for OOV-heavy domains where token-level representations are most stale
- Confirms that the contextualized nature of BERT representations makes them amenable to targeted domain adaptation via continued MLM
- Early Modern English benefits more than Twitter (larger vocabulary gap)

## Historical Significance
- Published October 2019, 5 months before [[Don't Stop Pretraining (Gururangan ACL2020)]]
- **First systematic application of DAPT to sequence labeling tasks**
- Gururangan et al. generalize this across 8 tasks; Han & Eisenstein establish the principle for seq labeling specifically

## Connections
- [[Domain-Adaptive Pre-Training (DAPT)]] — this paper is a key early source; sequence labeling focus
- [[BERT]] — backbone adapted
- [[DANN for NLP Text DA]] — complementary: DAPT addresses token representations; adversarial addresses feature distributions
- [[Adversarial MLM for NLP DA (Vu EMNLP2020)]] — extends this by making MLM adversarially domain-aware
