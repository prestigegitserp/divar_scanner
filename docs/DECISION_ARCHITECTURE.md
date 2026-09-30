# Decision architecture — Laya-first System-One v0.4

The project treats decision as a different computational object from text generation.

## Primary runtime

`laya-multilingual` is the default engine. It uses a multilingual encoder and typed decision head. Candidate options are supplied by the caller; the model does not autoregressively write an answer string.

This project preserves the observable Jev contract — bounded `choice`, `score`, and `noul` — without claiming to reproduce TypeSafe's private weights or proprietary training recipe.

## Primitive compiler

For robustness, all logical primitives are transported through opaque closed choices:

    logical choice / score / noul
               ↓
      opaque A/B/C/... options
               ↓
        Laya decision head
               ↓
       bounded probabilities
               ↓
      map back to logical type

For score, the returned logical value is the expected ordered level. For noul, the returned logical value is P(yes). No free-form output is generated.

## Option-order robustness

Each question schema is evaluated under deterministic option permutations. By default pass 0 uses canonical order and pass 1 reverses it. Every result is mapped back to canonical logical keys and pooled geometrically.

Order instability is measured with normalized Jensen-Shannon divergence. `decision_order_stability = 1 - normalized_JSD`.

## Evidence firewall

Four state views reduce shortcut leakage:

- content: ad text + structured fields + consistency evidence; no market anomaly.
- bait: content + duplicate graph; no single-listing market anomaly.
- market: structured facts + statistical market evidence; no persuasive ad copy.
- full: all evidence, only for final disposition and manual-review need.

Thus a cheap or expensive outlier is not automatically interpreted as deception.

## Cross-question coherence

Independent typed questions must agree. For example, data-error probability should align with integrity classification and consistency; misleading disposition should align with bait/duplicate decisions; market-outlier disposition should align with market status.

The resulting `decision_coherence_score` is part of selective prediction.

## Selective prediction / abstention

Model confidence, option-order stability, and cross-question coherence produce `decision_effective_confidence`. Below configured thresholds the decision layer abstains.

Abstention means 'send to human review', not 'this is fraud'.

## Three independent risk channels

- `data_problem_score`: record corruption, extraction problems, structured/text inconsistencies.
- `market_outlier_score`: unusual market position relative to peers and price models.
- `misleading_risk_score`: bait/misrepresentation evidence from bounded decisions and contradictory duplicate clusters.

Market-outlier evidence is intentionally not injected directly into misleading risk.

## Review priority

`review_priority_score` combines the strongest independent channels with uncertainty, explicit manual-review probability, and a small abstention uplift. It is a triage score, not a probability of criminal fraud.

## Calibration and fine-tuning

Question-specific temperature scaling is fit from held-out human labels. The exporter `divar-scanner export-laya-training` creates Laya state/questions/gold JSONL with the same opaque-marker transport used at inference.

## Baselines

mDeBERTa NLI, ParsBERT ParsiNLU, mBERT ParsiNLU and the Persian NLI ensemble remain research baselines only. They are non-generative but are not native System-One typed-decision heads.