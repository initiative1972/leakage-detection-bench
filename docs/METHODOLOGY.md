# Methodology

## 1. What LeakBench measures

A solver reads a **task specification** (never the label) and returns a
`Prediction(has_leakage, leak_type, offending_features)`. It is scored on three
sub-tasks of increasing difficulty:

1. **Detection** — binary: is there leakage? (precision / recall / F1 / accuracy)
2. **Diagnosis** — for truly-leaky items, is the `leak_type` correct? Credit is
   given only if the item was also *detected*.
3. **Localisation** — for leaky items whose type pins a feature
   (`target` / `temporal` / `proxy`), is the offending feature identified?
   (Jaccard overlap; credit only if detected.)

### The aggregate

```
aggregate = 0.5 · detection_F1 + 0.3 · type_accuracy + 0.2 · localisation
```

`type_accuracy` and `localisation` are computed over **truly-leaky** items
(missing a leak scores zero for that item), so the aggregate cannot be gamed by
predicting "leak" everywhere — that wins detection recall but forfeits diagnosis
and localisation. The weights are a deliberate editorial choice (detection is
necessary but not sufficient); they are documented here so results are
interpretable, and the grader exposes every component separately.

## 2. The leakage taxonomy

| Type | Definition | Localisable? |
|---|---|---|
| `target_leakage` | feature is a consequence/proxy of the label | yes |
| `temporal_leakage` | feature uses post-prediction-time information | yes |
| `preprocessing_leakage` | transform fit on all data before the split | no (process-level) |
| `group_leakage` | same entity in train and test (row-level split) | no (split-level) |
| `proxy_leakage` | an id/ordering/source encodes the label | yes |
| `none` | well-posed | — |

`target` vs `temporal` is the subtle pair: **target** leakage is a *consequence
of the label itself* (a chargeback, a recovery, a cancellation reason);
**temporal** leakage is any *post-prediction-time* value that is not the label's
direct consequence (a six-months-later bureau snapshot, a quantity that accrues
over the stay). Both are annotated with rationales that make the call defensible.

## 3. Authoring rubric

- **One dominant leak per leaky item**, described in realistic language — the
  leak must be *inferred*, not read off a machine-readable flag.
- **Clean items are genuinely well-posed**, and several carry *benign cue words*
  on purpose: a promotion flag that is *locked before* the forecast (vs TE01
  where it is finalised *after*), a *random UUID independent of the outcome*, a
  *past-dated* duration, a *published forecast* available before prediction.
  These exist to penalise shallow keyword matching.
- **Balance**: 4 items per positive type; 16 clean controls; a spread of easy /
  medium / hard and of domains.
- Items are written in a typed Python source and **machine-validated** (schema +
  label consistency) on every build, so a malformed or self-contradictory label
  cannot ship.

## 4. Baselines

All baselines are deterministic, dependency-free, and read only the spec.

| Baseline | Idea | Expected failure |
|---|---|---|
| `always_clean` | predict "no leak" | base-rate floor (F1 0) |
| `always_leak` | predict "leak", guess the modal type | high F1, but diagnosis/localisation collapse |
| `keyword_naive` | flag on a broad token list; treat any id column as a proxy and any random split as group leakage | **over-flags** (false positives on benign cue words) and **mis-blames ids** |
| `heuristic` | curated linter: post-outcome / future-info cues per feature, negation-aware preprocessing check, structural group rule | precise, but **misses the subtle tail** |

### v1.0 results (reproduced in CI)

| Solver | F1 | Prec | Rec | Type | Loc | Aggregate |
|---|---:|---:|---:|---:|---:|---:|
| heuristic | 0.947 | 1.000 | 0.900 | 0.900 | 0.917 | **0.927** |
| keyword_naive | 0.889 | 0.800 | 1.000 | 0.800 | 1.000 | 0.884 |
| always_leak | 0.714 | 0.556 | 1.000 | 0.200 | 0.000 | 0.417 |
| always_clean | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### Error analysis (why the numbers look the way they do)

- **`keyword_naive` false positives (5):** `CL02` (a churn target says
  *"cancels"*), `CL05` (the word *"readmission"* in the domain label), `CL06`
  and `CL15` (a random split over independent entities — perfectly fine),
  `CL08` (a random-UUID id column). Precision 0.80.
- **`keyword_naive` misdiagnosis:** all four `group_leakage` items (`GR01–GR04`)
  are labelled `proxy_leakage` — it blames the visible id column instead of
  reasoning about the split. Type accuracy 0.80.
- **`heuristic` false negatives (2):** `GR02` (group leakage stated only as *"a
  minority of reviewers account for most reviews"* — no explicit per-entity
  multiplicity cue) and `PX03` (a positional/ordering proxy from an unshuffled
  split). Recall 0.90, type accuracy 0.90.

These residual errors — benign-cue false positives, group-vs-proxy confusion,
and the subtle group/ordering tail — are the **headroom** that a reasoning model
(via the LLM adapter) must close to beat the linter.

## 5. Running an LLM

`src/leakbench/llm_adapter.py` provides `build_prompt`, a tolerant
`parse_response`, and `LLMJudge(complete)` where `complete(prompt)->str` is your
own model call. No network call lives in the repo, so CI is offline and
secret-free. A deterministic `ScriptedJudge` test double exercises the plumbing.

## 6. Reproducibility

- **Zero runtime dependencies** (pure Python standard library); dev tooling is
  just flake8 + pytest.
- `scripts/reproduce_all.py` rebuilds and validates the dataset, runs the
  baselines, renders the SVG figures, writes `results/leaderboard.json`, and
  **asserts** the headline findings (ordering, precision gap, residual
  headroom) so CI fails loudly if a change breaks the story.

## 7. Limitations & v2 roadmap

- Small (36 items) and **single-author**; no inter-annotator agreement yet.
- Authored leaks are textual, so strong rule linters do well on cued items.
- **v2**: grow to ~150 items, add a second annotator and report Cohen's κ, add
  harder "implicit" items (no telltale vocabulary), and publish baseline LLM
  results with a pinned prompt and seeds.
