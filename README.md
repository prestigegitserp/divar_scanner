# Divar Scanner v0.4.4 — automatic browser crawler + Laya System-One

A research-grade pipeline for public apartment-rental listings around Fatemi, Tehran.

The core decision layer is **not a generative LLM**. The default engine is **Laya Multilingual**, an open, non-autoregressive System-One model with a multilingual encoder and typed decision head. It receives caller-defined bounded options and returns probabilities; it does not write free-form answers.

> The scanner prioritizes rows for human review. It does not prove fraud, deception, or criminal intent.

## Why Laya

`laya-multilingual` is the primary backend because it best matches the project requirement:

- Jev/System-One style bounded decisions;
- no autoregressive text generation;
- dynamic caller-supplied option descriptions;
- open weights / Apache-2.0;
- multilingual encoder suitable for Persian experiments;
- practical on Google Colab.

Scientifically, Laya still contains a pretrained language **encoder** (mmBERT-like). It is not an autoregressive/chat LLM. If 'non-language-model at all' were required, dynamic semantic labels would have to be replaced with a fixed supervised head.

## Architecture

```text
Divar public listings
       ↓
normalization + Persian field extraction
       ↓
deterministic consistency checks
       ↓
robust market peers + conversion sensitivity
       ↓
Isolation Forest + OOF price model + LOF
       ↓
duplicate similarity graph
       ↓
priority candidates + stratified exploration
       ↓
Laya System-One decision packs
  ├─ integrity/data-error
  ├─ duplicate/bait
  ├─ market status
  └─ final disposition/review
       ↓
option-permutation stability + cross-question coherence
       ↓
selective prediction / abstention
       ↓
three independent risk channels
  ├─ data_problem_score
  ├─ market_outlier_score
  └─ misleading_risk_score
       ↓
review_priority_score
       ↓
CSV / Parquet / HTML / annotation export
```

## Jev-style primitives without generation

The project keeps the logical primitives `choice`, `score`, and `noul`, but v0.4 transports all of them through one robust closed-choice interface.

```text
logical option meaning
       ↓
opaque A/B/C/... marker
       ↓
Laya bounded decision head
       ↓
probability distribution
       ↓
map back to logical keys
```

- `choice`: argmax and full option distribution.
- `score`: expected ordered level from the bounded distribution.
- `noul`: P(yes) from a closed two-option distribution.

No label can be returned unless the caller supplied it.

## Position-bias mitigation

Laya's public notes document position bias for multilingual score questions and sensitivity in noul. v0.4 therefore:

1. uses opaque option markers;
2. runs deterministic option permutations (default: canonical + reversed);
3. maps every pass back to canonical keys;
4. pools probabilities geometrically;
5. records Jensen-Shannon disagreement as `decision_order_stability`;
6. abstains when order stability is too low.

The native action-head probability is not used as a trust gate.

## Evidence firewall

Different decisions see different evidence.

### Content / integrity pack

Sees ad text, structured fields and deterministic consistency evidence. It does not see market anomaly.

### Bait / duplicate pack

Sees content plus duplicate-graph evidence. It does **not** see single-listing market anomaly or OOF price residual. This blocks the shortcut `cheap => fraud`.

### Market pack

Sees structured property facts and market statistics, not persuasive ad copy.

### Full pack

Sees all evidence and is used only for final disposition and whether a human should review the row.

## Trust stages

Laya Multilingual is intentionally not given full influence zero-shot. The default config is:

    decision:
      trust_stage: bootstrap
    scoring:
      bootstrap_decision_multiplier: 0.25
      calibrated_decision_multiplier: 0.45
      adapted_decision_multiplier: 1.0

`bootstrap` is for collecting/triaging labels with the base multilingual checkpoint. `calibrated` is for a checkpoint whose probabilities were fitted on held-out Fatemi labels. `adapted` should only be used after domain fine-tuning plus held-out evaluation. Model confidence cannot bypass this cap.
## Selective prediction

Laya is allowed to abstain. v0.4 computes:

- `decision_effective_confidence`;
- `decision_order_stability`;
- `decision_coherence_score`.

If thresholds fail, `decision_abstain=True`. Abstention increases review priority; it does not increase fraud suspicion directly.

## Three independent output concepts

### `data_problem_score`

Likelihood/evidence that the row is internally inconsistent, corrupted, or incorrectly extracted.

### `market_outlier_score`

How unusual the property is relative to local peers and price models.

### `misleading_risk_score`

Evidence compatible with misleading/bait representation, using duplicate contradictions and bounded decision outputs. Market anomaly is intentionally not directly injected here.

`suspicion_score` remains as a backward-compatible alias of `misleading_risk_score`.

### `review_priority_score`

A triage score combining the strongest independent channels, uncertainty, explicit manual-review probability, and a small abstention uplift.

## Decision questions

v0.4 asks bounded questions for:

- `disposition`: plausible / data_error / market_outlier / misleading_or_bait / ambiguous_mixed;
- `integrity_class`: consistent / extraction_error / listing_claim_conflict / insufficient_evidence;
- `consistency`: five ordered levels;
- `duplicate_pattern`: no evidence / normal template reuse / same-property repost / cross-property conflict;
- `market_status`: typical / moderate outlier / extreme outlier / insufficient context;
- `bait_evidence`: bounded no/yes;
- `data_error_evidence`: bounded no/yes;
- `manual_review`: bounded no/yes.

## Housing-market logic

Iranian rental contracts are not treated as one homogeneous price variable. The pipeline distinguishes full-deposit, rent-only, mixed and unknown contract styles and evaluates several rent↔deposit equivalence assumptions (25×/30×/35× monthly rent). If anomaly depends strongly on one arbitrary conversion factor, `equivalence_sensitivity` lowers confidence.

## Calibration and domain adaptation

Raw System-One probabilities are not assumed to be calibrated for Tehran housing.

The review/annotation CSV includes human-label columns for every major decision pack. After annotation:

```bash
divar-scanner calibrate \
  --input outputs/fatemi/review_queue_LABELLED.csv \
  --output config/fatemi_calibration.json
```

Question-specific temperature scaling changes probability sharpness, not the winning option.

Evaluate the labelled bounded decisions on an untouched split:

```bash
divar-scanner evaluate-decisions \
  --input outputs/fatemi/heldout_LABELLED.csv \
  --output outputs/decision_evaluation.json
```

The evaluator reports accuracy together with NLL, multiclass Brier score, 10-bin ECE, abstention coverage, and selective accuracy on non-abstained rows.

For deeper adaptation:

```bash
divar-scanner export-laya-training \
  --input outputs/fatemi/decision_annotation_sample_LABELLED.csv \
  --output data/laya/fatemi_train.jsonl
```

The exporter uses the same opaque option transport as inference.

## Automatic Divar crawling

The repository default is now a **real Chromium browser crawler** driven by Playwright.
You do not need to browse or save pages manually.

The crawler:

1. opens the public Fatemi rental search page;
2. waits for listing links;
3. scrolls automatically until it reaches the configured limit or the list stops growing;
4. deduplicates listing tokens;
5. optionally opens each public listing page for richer text/JSON-LD evidence;
6. writes an immutable raw JSONL snapshot;
7. can then feed that snapshot directly into the statistical + Laya pipeline.

It does not log in, open contact/phone endpoints, solve CAPTCHAs, rotate proxies, spoof
fingerprints, or continue after explicit 401/403/429/access-control signals.

### One-time browser setup

```bash
git clone https://github.com/prestigegitserp/divar_scanner.git divar_scanner_repo
cd divar_scanner_repo

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

python -m pip install -e '.[browser]'
python -m playwright install chromium
```

### Fully automatic acquisition

```bash
divar-scanner acquire \
  --config config/fatemi.yaml \
  --transport browser \
  --max-listings 200
```

No manual search is required. The command prints metadata including the generated path,
for example:

```text
data/raw/browser_raw_20261001T....jsonl
```

By default the browser runs headless and enriches public listing details. For debugging you
can watch Chromium:

```bash
divar-scanner acquire --transport browser --max-listings 100 --headed
```

To make acquisition lighter and only collect search cards:

```bash
divar-scanner acquire --transport browser --max-listings 200 --no-detail-enrichment
```

### Crawl + analyze in one command

If the machine also has enough resources for the decision model:

```bash
python -m pip install -e '.[browser,decision]'
python -m playwright install chromium

divar-scanner run \
  --config config/fatemi.yaml \
  --crawl-transport browser \
  --max-listings 200 \
  --decision-backend laya-multilingual
```

### Colab handoff

Your Colab runtime cannot currently establish TCP connections to Divar, so the browser
crawler should run on a machine/network that can open the public site. Colab remains the
GPU analysis environment:

```text
local Playwright crawler
        ↓
browser_raw_*.jsonl
        ↓
Colab snapshot upload
        ↓
statistics + Laya System-One
```

The Colab notebook therefore keeps `ACQUISITION_MODE='snapshot'`; that is a network
constraint of Colab, not a manual-browsing requirement. The snapshot itself is produced
automatically by the browser crawler.

The older `web` and `api` transports remain compatibility paths. Kenar code remains
optional but is not required by the default workflow.

## Google Colab

Open:

https://colab.research.google.com/github/prestigegitserp/divar_scanner/blob/main/colab/Divar_Fatemi_Anomaly_Scanner.ipynb

Recommended first run:

```text
MAX_LISTINGS = 100
RUN_DECISION_MODEL = True
DECISION_BACKEND = laya-multilingual
DECISION_TOP_K = 30
EXPLORATION_SAMPLE = 20
PERMUTATION_PASSES = 2
```

The notebook installs `.[decision]`, verifies package origin, prints GPU diagnostics, can run a small Persian wiring diagnostic, then executes the full pipeline.

## Local install

```bash
git clone https://github.com/prestigegitserp/divar_scanner.git divar_scanner_repo
cd divar_scanner_repo
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,browser,decision]'
python -m playwright install chromium
pytest
```

Run:

```bash
divar-scanner run --config config/fatemi.yaml --max-listings 200 --decision-backend laya-multilingual
```

## Research baselines

The repo retains `mdeberta-nli`, `parsbert-parsinlu`, `mbert-parsinlu`, and `persian-ensemble` only for controlled comparisons. They are non-generative encoder classifiers, but unlike Laya they are not native typed System-One heads.

OpenJev and Lev are highly relevant references, but their current public implementations use Qwen-derived autoregressive backbones, so they do not satisfy this project's stricter no-autoregressive-LLM-backbone requirement.

See:

- `docs/SYSTEM_ONE_RESEARCH.md`
- `docs/DECISION_ARCHITECTURE.md`
- `docs/DECISION_MODELS.md`

## Responsible crawling

The crawler uses public listing/search responses, caches responses, rate-limits requests, stops on 401/403/429, does not bypass CAPTCHA/access controls, does not use OTP/login/contact-info endpoints, and redacts phone-like strings by default.

## CI

CI covers Python 3.10/3.11/3.12, packaging/import checks, bounded decision math, Laya opaque transport and permutation stability, calibration, fine-tune export, synthetic end-to-end pipeline, YAML parsing, Colab code compilation, and simulated Colab path layout.

Heavy model weights are not downloaded in ordinary CI.

## License

Project code: MIT. Model licenses are separate. Laya is Apache-2.0; ParsiNLU baseline checkpoints have more restrictive non-commercial/share-alike terms.