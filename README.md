# leakage-detection-bench
**A labelled benchmark for detecting data leakage in ML task specifications —
with a deterministic grader, reproducible baselines, a leaderboard, and
contamination controls.** This is the AfterQuery-shaped companion to
[`decision-grade-ml-eval`](https://github.com/initiative1972/decision-grade-ml-eval):
there the finding was *"the leak is the finding"*; here that judgement is turned
into a **benchmark** that measures whether a solver — a rule linter or an LLM —
can find the leak, name its type, and point at the offending feature.

Each item is a realistic ML **task specification** (domain, target, prediction
time, split, preprocessing, and natural-language feature descriptions). A solver
must answer three things: **is there leakage?** (detection), **what kind?**
(diagnosis), and **which feature?** (localisation). The leak must be *reasoned
about* — it is never a single machine-readable flag.

[![ci](https://github.com/initiative1972/leakage-detection-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/initiative1972/leakage-detection-bench/actions/workflows/ci.yml)

---

## Leaderboard (v1.0, 36 items, reproduced in CI)

| Solver | Detection F1 | Precision | Recall | Type acc. | Localisation | **Aggregate** |
|---|---:|---:|---:|---:|---:|---:|
| **heuristic** (curated linter) | 0.947 | **1.000** | 0.900 | 0.900 | 0.917 | **0.927** |
| keyword_naive | 0.889 | 0.800 | 1.000 | 0.800 | 1.000 | 0.884 |
| always_leak | 0.714 | 0.556 | 1.000 | 0.200 | 0.000 | 0.417 |
| always_clean | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

`aggregate = 0.5·F1 + 0.3·type_accuracy + 0.2·localisation` — so shouting
"leak!" on everything (always_leak) scores a deceptively high F1 (0.714) but
collapses on the aggregate (0.417) because it cannot **diagnose** or **localise**.

> **The three LLM slots are intentionally empty.** Running a live model is a
> *you-run-it* step (see [`llm_adapter`](src/leakbench/llm_adapter.py)); no
> network calls live in this repo, so CI stays offline, deterministic and
> secret-free. The headroom below is what a reasoning model has to earn.

### What the baselines teach (the point of the benchmark)

- **Shallow pattern-matching over-flags.** `keyword_naive` has perfect recall
  but precision 0.80: it trips on label-ish words in the domain/target text
  (e.g. a *"churn"* task whose target says *"cancels"*), treats **any** random
  split as group leakage, and flags **any** id column as a proxy — five clean
  specs wrongly flagged.
- **…and misdiagnoses.** `keyword_naive` labels all four genuine
  *group-leakage* specs as *proxy* leakage — it blames the visible id column
  instead of reasoning about the split.
- **Curated rules are precise but hit a ceiling.** The `heuristic` linter is
  perfectly precise and strong (F1 0.947), yet it still **misses the subtle
  tail**: group leakage described without an explicit multiplicity cue, and a
  positional/ordering proxy. Recall 0.90 and type-accuracy 0.90 are the
  headroom that judgement (or an LLM) must close.

![leaderboard](docs/figures/leaderboard.svg)
![recall by type](docs/figures/recall_by_type.svg)

---

## The leakage taxonomy

| Type | One-line definition |
|---|---|
| `target_leakage` | a feature is a consequence/proxy of the label (known only after the outcome) |
| `temporal_leakage` | a feature uses information time-stamped after the prediction moment |
| `preprocessing_leakage` | a transform (scale/impute/select/resample) is fit on all data before the split |
| `group_leakage` | the same entity appears in both train and test (row-level split) |
| `proxy_leakage` | an id / ordering / source code encodes the label |
| `none` | the spec is well-posed |

Dataset v1.0: **36 items** — 20 leaky (4 of each positive type) + 16 clean
controls (several deliberately contain *benign* cue words — a legitimately
pre-locked promo flag, a random UUID, a past-dated duration — to punish shallow
matching). Difficulty: 9 easy / 17 medium / 10 hard.

---

## Use it

```bash
make install          # dev deps only (flake8, pytest); the benchmark itself
                      # has ZERO runtime dependencies (pure standard library)
make reproduce        # build + validate dataset, run baselines, render, gate
```

Score your own solver (any `TaskSpec -> Prediction` callable):

```python
from leakbench.dataset import load
from leakbench.evaluate import evaluate

def my_solver(spec):
    ...  # return leakbench.schema.Prediction(has_leakage, leak_type, offending)

print(evaluate(load(), my_solver, "my_solver"))
```

Plug in a real LLM (you provide the API call; nothing leaves your machine
unless you make it):

```python
from leakbench.llm_adapter import LLMJudge
from leakbench.dataset import load
from leakbench.evaluate import evaluate

judge = LLMJudge(complete=my_model_call)   # my_model_call(prompt:str)->str
print(evaluate(load(), judge, "gpt-x"))
```

Load the dataset with Hugging Face `datasets` (optional):

```python
from datasets import load_dataset
ds = load_dataset("json", data_files="data/benchmark.jsonl", split="train")
```

See the **[dataset card](docs/DATASET_CARD.md)** and **[methodology](docs/METHODOLOGY.md)**.

---

## Contamination controls

- **Canary.** Every record embeds a fixed canary string
  (`LEAKBENCH-CANARY-…do-not-train`). If a future model emits it, this dataset
  leaked into its training data. **Do not train on LeakBench.**
- **Near-duplicate check.** `validate_dataset.py` fails if any two specs are
  ≥ 0.92 similar, so a solver cannot pattern-match duplicates or double-count.
- **Held-out guidance.** The repo ships labels (it is an eval set); the dataset
  card documents how to treat it as held-out.

---

## "Runs" vs "design"

| Capability | Runs in this repo | Design / you-run-it |
|---|---|---|
| Labelled benchmark + typed schema + validation | ✅ 36 items, schema-checked | growing to v2 (more items/types) |
| Deterministic grader (detection + diagnosis + localisation) | ✅ pure stdlib, unit-tested | — |
| Non-LLM baselines + leaderboard | ✅ 4 solvers, reproduced in CI | — |
| Contamination controls (canary + near-dup) | ✅ enforced in validation | — |
| LLM-judge evaluation | adapter + prompt + parser + test double ✅ | live model call (bring your own key) |
| Inter-annotator agreement | — | single-author v1.0 (documented honestly) |

---

## Honesty note

- **All content is synthetic and illustrative.** No private or employer data,
  systems, or projects are used or referenced. No network, GPU, or secrets.
- Every leaderboard number is produced by `scripts/reproduce_all.py` and
  committed under `results/`; the run is deterministic (pure standard library).
- **Single-author v1.0.** Labels follow the rubric in the methodology doc and a
  self-review pass; there is **no** multi-annotator agreement study yet, and
  that is stated as a limitation rather than dressed up.
- This is a **portfolio / teaching** benchmark: small, honest, and reproducible,
  built to demonstrate benchmark-construction craft — not a published SOTA
  leaderboard.

## License

MIT for code; the benchmark data and docs are additionally CC-BY-4.0.
© 2026 Henry Yan. See [`CITATION.cff`](CITATION.cff).
