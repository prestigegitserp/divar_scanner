# Divar Scanner v0.3 — Jev-style bounded decision system for Tehran rentals

A research-grade pipeline for collecting public apartment-rental listings around **Fatemi, Tehran** and prioritizing listings that deserve human review.

The central design is **not an LLM agent**. The default decision layer is a non-generative NLI sequence classifier that implements a Jev-style closed decision contract:

```text
state + caller-supplied options
            │
            ▼
  entail / neutral / contradict
            │
            ▼
 bounded option evidence
            │
            ▼
 choice / score / noul probabilities
```

No answer token is generated. A label that was not supplied by the caller cannot be invented.

> The project does not claim to reproduce TypeSafe Jev's unpublished internal weights/training. It preserves the public typed-decision shape and uses open sequence-classification models as the decision motor.

---

## Architecture

```text
Divar public listing/search payloads
                │
                ▼
       conservative crawler + cache
                │
                ▼
    Persian normalization / extraction
                │
        ┌───────┴──────────────────────────────────────┐
        │                                              │
        ▼                                              ▼
field consistency                              market evidence
area/room/amenity                         contract-style aware peers
cross-checks                              25x / 30x / 35x sensitivity
        │                                  Isolation Forest
        │                                  OOF price model
        │                                  Local Outlier Factor
        │                                              │
        └────────────────┬─────────────────────────────┘
                         │
                         ▼
                duplicate similarity graph
        copied text / neighborhood / price inconsistency
                         │
                         ▼
                   cheap prefilter
                         │
                  top candidates only
                         ▼
            bounded decision layer
          ┌──────────────┼──────────────┐
          │              │              │
       choice           score          noul
          │              │              │
          └──────────────┼──────────────┘
                         ▼
             confidence / disagreement
                         │
                         ▼
           evidence-aware risk fusion
                         │
             normal / review / high
                         │
          CSV / Parquet / HTML report
```

---

## Decision model: no generation

### Default backend

```text
mdeberta-nli
MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
```

It is loaded with:

```python
AutoModelForSequenceClassification
```

not `AutoModelForCausalLM` and not a text-generation pipeline.

The model is multilingual NLI and its published training-language list includes Persian (`fa`). The default checkpoint is MIT licensed.

### Persian specialist

```text
parsbert-parsinlu
persiannlp/parsbert-base-parsinlu-entailment
```

This is a Persian textual-entailment classifier trained on ParsiNLU. Its model card exposes:

```text
entails / contradicts / neutral
```

License: `CC-BY-NC-SA-4.0`. Review the license before any commercial deployment.

### Persian ensemble

```text
persian-ensemble
```

Default members:

```text
0.55 × mdeberta-nli
0.45 × parsbert-parsinlu
```

The ensemble fuses **pre-softmax decision evidence**, not final probabilities:

```text
e_(m,i) = log P_m(entail_i) - log P_m(contradict_i)

e_i = Σ_m w_m e_(m,i)

P(option_i) = softmax(e_i / T)
```

If the models disagree strongly on a candidate, that disagreement lowers the reported decision confidence.

See:

- `docs/DECISION_ARCHITECTURE.md`
- `docs/DECISION_MODELS.md`

---

## Jev-style primitives

### `choice`

Caller supplies a closed set of options and descriptions.

Example:

```text
plausible
data_error
market_outlier
misleading_or_bait
ambiguous_mixed
```

Each option becomes an NLI hypothesis. The decision engine returns a probability distribution only over these options.

### `noul`

Binary bounded decision.

The engine explicitly evaluates:

```text
no
yes
```

and returns:

```text
P(yes)
```

The housing pipeline uses independent noul questions for:

- bait/misleading evidence;
- data-error evidence;
- manual-review need.

### `score`

Ordered levels become bounded candidates.

For consistency:

```text
0 major contradictions
1 important inconsistencies
2 ambiguous/minor mismatch
3 mostly consistent
4 strongly consistent
```

After the option distribution is obtained:

```text
score = Σ_k k × P(level_k)
```

---

## Evidence firewall

Not every decision question sees every signal.

This is deliberate.

### Bait/misleading decision

It sees:

- title and description;
- structured fields;
- cross-field contradictions;
- duplicate graph;
- duplicate neighborhood/price inconsistencies.

It **does not see the single-listing market anomaly / OOF price residual**.

This prevents the shortcut:

```text
cheap listing → fraud
```

### Data-error / consistency decision

It sees the listing content and structured-field mismatch evidence, not unrelated market evidence.

### Primary disposition / manual review

These can see the full evidence state.

This separation makes the decision semantics more auditable and reduces circular reasoning.

---

## Housing-market logic

Iranian rental listings cannot be treated as one homogeneous price variable.

The pipeline distinguishes:

```text
full_deposit
rent_only
mixed
unknown
```

and creates `contract_rent_share`.

Instead of trusting one fixed deposit/rent conversion assumption, it evaluates multiple assumptions:

```text
25× monthly rent
30× monthly rent
35× monthly rent
```

The anomaly score uses the median behavior across these assumptions and records:

```text
equivalence_sensitivity
```

If a listing looks anomalous only under one arbitrary conversion factor, confidence in that market anomaly is reduced.

Raw rent/deposit peer comparisons are also segmented by contract style.

---

## Other anomaly views

### Robust local peers

Hierarchical peer groups:

1. neighborhood + area bucket + rooms
2. neighborhood + area bucket
3. neighborhood
4. global fallback

Median/MAD is used instead of mean/std.

### Isolation Forest

Checks unusual multivariate combinations.

### OOF property-price expectation

A cross-validated `HistGradientBoostingRegressor` predicts equivalent deposit from property characteristics and neighborhood.

Each row is predicted by a fold that did not train on that row.

Importantly, raw rent/deposit are **not** used as predictors for a target algebraically derived from rent/deposit; that target leakage was explicitly removed.

### Local Outlier Factor

LOF uses property characteristics, equivalent price and contract-rent share—not raw rent/deposit as independent coordinates—so full-deposit and rent-heavy contracts are less likely to be falsely flagged merely because of contract structure.

### Duplicate graph

Near-duplicate title/body text creates a graph rather than only pairwise matches.

Per component:

```text
duplicate_cluster_size
duplicate_cluster_neighborhoods
duplicate_cluster_price_span
duplicate_bait_score
```

Repeated text alone is not automatically suspicious. Repeated text combined with conflicting neighborhoods/prices is stronger review evidence.

---

## Probability calibration

Raw NLI option probabilities are **not automatically real-world calibrated probabilities**.

The review CSV includes blank human-label columns:

```text
human_disposition
human_bait
human_data_error
human_manual_review
human_consistency_level
human_notes
```

After manually annotating enough rows:

```bash
divar-scanner calibrate \
  --input outputs/fatemi/review_queue_LABELLED.csv \
  --output config/fatemi_calibration.json
```

Then set:

```yaml
decision:
  calibration_file: "config/fatemi_calibration.json"
```

The project fits **question-specific temperature scaling** on bounded candidate evidence.

Temperature scaling changes probability sharpness but does not change the winning argmax option.

Keep a separate held-out evaluation set. Do not fit calibration and report final performance on the same rows.

See `docs/ANNOTATION_GUIDE.md`.

---

## Colab

Open:

https://colab.research.google.com/github/prestigegitserp/divar_scanner/blob/main/colab/Divar_Fatemi_Anomaly_Scanner.ipynb

The notebook:

1. clones into `/content/divar_scanner_repo` to avoid the old namespace collision;
2. installs `.[decision]`;
3. asserts the exact package import origin;
4. prints dependency/GPU diagnostics;
5. optionally runs a small Persian bounded-decision diagnostic;
6. crawls and scores listings;
7. shows the review table and HTML report.

Decision backend options:

```text
mdeberta-nli
parsbert-parsinlu
mbert-parsinlu
persian-ensemble
```

For a first run:

```text
MAX_LISTINGS = 100
DECISION_BACKEND = mdeberta-nli
DECISION_TOP_K = 30
```

A GPU is recommended but the base encoder can run on CPU.

---

## The old Colab import bug

Never clone the repository into:

```text
/content/divar_scanner
```

because the Python package has the same name.

The notebook uses:

```text
/content/divar_scanner_repo
```

and explicitly puts:

```text
/content/divar_scanner_repo/src
```

at the front of `sys.path`.

CI also compiles every notebook Python cell to catch malformed escapes/syntax before delivery.

---

## Local install

```bash
git clone https://github.com/prestigegitserp/divar_scanner.git divar_scanner_repo
cd divar_scanner_repo

python -m venv .venv
source .venv/bin/activate

python -m pip install -e '.[dev,decision]'
pytest
python -m divar_scanner.doctor --network
```

Run:

```bash
divar-scanner run \
  --config config/fatemi.yaml \
  --max-listings 200 \
  --decision-backend mdeberta-nli
```

Run without the decision model:

```bash
divar-scanner run --config config/fatemi.yaml --max-listings 200 --no-decision
```

Analyze an existing normalized dataset:

```bash
divar-scanner analyze \
  --config config/fatemi.yaml \
  --input listings.parquet
```

---

## Main output columns

| Column | Meaning |
|---|---|
| `data_quality_score` | impossible/missing/cross-field mismatch evidence |
| `contract_style` | full_deposit / rent_only / mixed / unknown |
| `equivalence_sensitivity` | sensitivity to rent↔deposit conversion assumptions |
| `market_anomaly_score` | fused market outlier evidence |
| `price_model_ratio` | observed / OOF expected equivalent price |
| `price_model_anomaly_score` | OOF price residual anomaly |
| `lof_anomaly_score` | local-density anomaly |
| `duplicate_bait_score` | duplicate-graph contradiction signal |
| `decision_disposition` | bounded primary disposition |
| `decision_disposition_probs_json` | closed-set option distribution |
| `decision_bait_probability` | noul P(yes) |
| `decision_data_error_probability` | noul P(yes) |
| `decision_manual_review_probability` | noul P(yes) |
| `decision_consistency_score` | ordered score normalized to 0–1 |
| `decision_disposition_confidence` | entropy/neutral/disagreement diagnostic |
| `uncertainty_score` | disagreement across evidence families |
| `suspicion_score` | evidence-fused review risk |
| `review_priority_score` | final triage priority |
| `flag_reasons` | human-readable reasons |

---

## Why not use OpenJev directly?

OpenJev is a very useful architectural reference: it demonstrates a typed, non-generative cross-encoder decision pattern with entailment / contradiction / neutral and probabilities over closed options.

Its current public checkpoints use **Qwen3.5-derived** sequence-classification backbones.

This repository's current design requirement is stricter:

```text
no free-text generation
AND
non-LLM encoder backbone
```

Therefore OpenJev is treated as a benchmark/reference, while mDeBERTa / ParsBERT NLI are the executable decision backends.

---

## What the system does not claim

It does not claim:

- that an outlier is fraud;
- that a high review score proves deceptive intent;
- that NLI zero-shot probabilities are calibrated without target-domain labels;
- that this reimplements TypeSafe Jev's proprietary RLCD training;
- that one neighborhood snapshot is sufficient for production fraud detection.

The strongest next improvements are longitudinal snapshots, human labels, calibration, threshold validation, and independent evaluation.

---

## Responsible crawling

The crawler:

- uses public listing/search responses;
- caches responses;
- rate-limits requests;
- stops on 401/403/429;
- does not bypass CAPTCHA/access controls;
- does not use OTP/login/contact-info endpoints;
- redacts phone-like strings from listing descriptions by default.

Divar's web schema is not a stable public contract; schema changes should be handled by the adapter rather than more aggressive crawling.

---

## Tests / CI

CI currently checks:

- Python 3.10 / 3.11 / 3.12;
- packaging/import path;
- unit tests;
- bounded `choice / score / noul` math;
- Persian model registry/label mapping;
- ensemble disagreement behavior;
- temperature calibration;
- synthetic end-to-end pipeline;
- YAML parsing;
- every Colab Python cell via `compile()`;
- simulated Colab clone layout.

The heavy model weights are not downloaded in ordinary CI; the notebook's Persian diagnostic is the runtime model check.

---

## References

- TypeSafe Jev guides/API: https://www.typesafeai.org/guides/choice-score-noul
- OpenJev: https://huggingface.co/AlexWortega/openjev
- mDeBERTa multilingual NLI: https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
- ParsBERT ParsiNLU entailment: https://huggingface.co/persiannlp/parsbert-base-parsinlu-entailment
- ParsiNLU: https://github.com/persiannlp/parsinlu
- Temperature scaling: Guo et al., 2017, *On Calibration of Modern Neural Networks*

## License

Project code: MIT.

Model licenses are separate. In particular, the ParsiNLU ParsBERT/mBERT checkpoints are marked `CC-BY-NC-SA-4.0`; verify compatibility with your use case before deployment.
