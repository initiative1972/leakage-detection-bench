---
license: mit
language: en
pretty_name: "LeakBench: Data-Leakage Detection in ML Task Specifications"
tags:
  - evaluation
  - benchmark
  - data-leakage
  - machine-learning-evaluation
size_categories:
  - n<1K
task_categories:
  - text-classification
configs:
  - config_name: default
    data_files: data/benchmark.jsonl
---

# Dataset Card — LeakBench v1.0

> This card doubles as the Hugging Face dataset card. The YAML header above is
> the HF metadata block; when pushed to the Hub, this file becomes the dataset
> README.

## Dataset summary

LeakBench is a small, expert-authored, labelled benchmark for **detecting data
leakage in machine-learning task specifications**. Each item is a realistic ML
task description — domain, target, prediction time, split strategy,
preprocessing, and natural-language feature descriptions — annotated with:

- `has_leakage` (bool),
- `leak_type` (`target` / `temporal` / `preprocessing` / `group` / `proxy` /
  `none`),
- `offending_features` (the feature(s) at fault, when the type pins one),
- `rationale` (one sentence), and
- `difficulty` (`easy` / `medium` / `hard`).

The task is to decide whether the spec contains leakage, classify its type, and
localise the offending feature — the judgement that separates an honest
evaluation from an impressive-but-wrong one.

## Supported tasks

- **Leakage detection** (binary), **diagnosis** (6-way type), and
  **localisation** (feature identification). A deterministic grader and a
  blended `aggregate` score ship in the repository.

## Languages

English.

## Dataset structure

One JSON object per line (`data/benchmark.jsonl`). Example (abridged):

```json
{
  "spec": {
    "id": "TL03",
    "domain": "Payments - card fraud",
    "task_type": "classification",
    "target": "transaction is fraudulent",
    "prediction_time": "scored in real time at authorisation",
    "split": "time-ordered split by transaction timestamp",
    "preprocessing": "amount log-transformed; encoders fit on train only",
    "features": [
      {"name": "amount", "dtype": "float", "description": "transaction amount"},
      {"name": "chargeback_filed", "dtype": "bool",
       "description": "whether the cardholder later filed a chargeback ..."}
    ],
    "notes": ""
  },
  "label": {
    "has_leakage": true,
    "leak_type": "target_leakage",
    "offending_features": ["chargeback_filed"],
    "rationale": "A chargeback is filed after the transaction ...",
    "difficulty": "easy"
  },
  "canary": "LEAKBENCH-CANARY-...-do-not-train"
}
```

### Fields

| Field | Type | Notes |
|---|---|---|
| `spec.id` | str | unique id (e.g. `TL03`) |
| `spec.domain` / `target` / `prediction_time` / `split` / `preprocessing` | str | the task setup a solver reads |
| `spec.features[]` | list | each `{name, dtype, description}` |
| `label.has_leakage` | bool | ground truth |
| `label.leak_type` | str | one of the taxonomy |
| `label.offending_features` | list[str] | subset of feature names |
| `label.rationale` | str | why |
| `label.difficulty` | str | easy / medium / hard |
| `canary` | str | fixed contamination canary |

### Splits & statistics (v1.0)

- **36 items**: 20 leaky, 16 clean.
- By type: `target` 4, `temporal` 4, `preprocessing` 4, `group` 4, `proxy` 4,
  `none` 16.
- By difficulty: easy 9, medium 17, hard 10.
- No official train/val/test split — LeakBench is an **evaluation set**. Use it
  only to score solvers; do not train on it (see contamination controls).

## Dataset creation

### Curation rationale

Benchmarks that merely report accuracy miss the failure that most often fools
practitioners in the wild: **leakage**. LeakBench isolates that judgement and
makes it measurable, in the idiom a reviewer actually faces — a task
description, not a pre-flagged table.

### Source data & annotation

- **Synthetic, hand-authored.** Every spec was written for this benchmark; the
  domains (banking, healthcare, fraud, retail, ad-tech, etc.) are archetypal,
  and no real, private, or employer data is used or referenced.
- **Authoring rubric** (see `docs/METHODOLOGY.md`): each leaky item embeds
  exactly one dominant leak described in realistic language; each clean item is
  well-posed, and several clean items carry *benign* cue words (a pre-locked
  promo flag, a random UUID, a past-dated duration) specifically to penalise
  shallow keyword matching.
- **Labels** are serialised from a typed source (`src/leakbench/_authored.py`)
  and machine-validated for schema and label consistency on every build.

### Annotation quality & limitations

- **Single-author v1.0.** Labels reflect one author's rubric-guided judgement
  plus a self-review pass. There is **no** multi-annotator agreement study yet;
  this is a stated limitation, not hidden.
- **Small (36 items).** Per-type estimates are therefore coarse; treat the
  leaderboard as directional. A v2 roadmap (more items, more annotators,
  inter-rater agreement) is in the methodology doc.
- **Authored leaks are ultimately textual**, so a well-built rule linter does
  well on cued items; the benchmark's value is the *subtle tail* and the
  *diagnosis + localisation* sub-tasks, where rules break down.

## Contamination controls

- **Canary string** embedded in every record (`LEAKBENCH-CANARY-…do-not-train`)
  so training-set contamination can be detected later.
- **Near-duplicate check** (≥ 0.92 text similarity) enforced in validation.
- **Do not train on LeakBench.** It is an evaluation artefact.

## Considerations for use

- Intended for evaluating leakage-detection judgement of rule systems and LLMs,
  and for teaching. Not a safety-critical or production artefact.
- Scores are specific to v1.0; expanding the set will change them.

## Licensing & citation

MIT (code) + CC-BY-4.0 (data and docs). Cite via `CITATION.cff`
(a Zenodo DOI will be minted on first release and backfilled here).
