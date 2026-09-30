# System-One model research for the Fatemi housing scanner

## Requirement

The project requires a bounded decision model, not a generative LLM: caller-supplied options, probabilities, no free-text generation, Colab practicality, multilingual/Persian support, and open weights.

## Selected primary model: Laya Multilingual

Model: convaiinnovations/laya, subfolder multilingual. License: Apache-2.0.

Laya is the closest open implementation found to the public Jev/System-One shape while also satisfying the stricter requirement of no autoregressive decoder. It uses a multilingual BERT-style encoder (mmBERT) plus a typed decision head. It is not a chat/generative LLM, although scientifically it is still a pretrained language encoder.

state + caller options -> multilingual encoder -> typed decision head -> closed option probabilities

## Upstream limitations and v0.4 mitigations

Laya documents position bias for multilingual score questions, sensitivity of noul to label wording, overconfident probabilities, and recommends not using the action-head probability as the primary gate.

v0.4 therefore compiles logical choice, score and noul to the same opaque closed-choice transport. It runs deterministic option permutations, maps every pass back to canonical keys, logarithmically pools probabilities, records Jensen-Shannon disagreement as order instability, applies target-domain temperature calibration, and abstains when confidence/coherence/stability are weak.

## Why OpenJev is not primary

OpenJev is an excellent Jev-style architectural reference, but current public checkpoints use a Qwen-derived backbone and are much larger. That violates this project's stricter no-autoregressive-LLM-backbone requirement.

## Why Lev is not primary

Lev is System-One / zero-output-token at inference, but its public implementation is a LoRA on a Qwen decoder model.

## Why OpenDecider is not primary

OpenDecider is compact and interesting, but its public evaluations are currently much more English-centric than this Persian housing use case.

## Scientific baselines

The repo retains mDeBERTa NLI, ParsBERT ParsiNLU, mBERT ParsiNLU and a Persian NLI ensemble only as non-generative baselines. They are not the default runtime because they were not trained as native typed System-One decision heads.

## Domain-adaptation path

crawl -> Laya zero-shot + independent statistical evidence -> stratified human annotation -> temperature calibration -> Laya-format supervised/RLCD-compatible training export -> fine-tuned housing decision head -> held-out evaluation

The training exporter uses the exact same opaque option mapping as inference.

## References

- TypeSafe Jev: https://www.typesafeai.org/guides/choice-score-noul
- Laya: https://huggingface.co/convaiinnovations/laya
- OpenJev: https://huggingface.co/AlexWortega/openjev
- Lev: https://github.com/interfaze-ai/lev