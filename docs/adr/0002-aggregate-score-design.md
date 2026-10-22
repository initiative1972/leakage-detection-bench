# ADR 0002 — Aggregate score that cannot be gamed by "always leak"

## Status
Accepted.

## Context
Detection F1 alone is a poor headline. On a benchmark where leaky items
outnumber clean ones, a solver that predicts "leak" on everything earns a high
F1 (0.714 here) while demonstrating zero understanding — it cannot say *what*
kind of leak or *where* it is.

## Decision
Report a blended aggregate:

```
aggregate = 0.5 · detection_F1 + 0.3 · type_accuracy + 0.2 · localisation
```

with `type_accuracy` and `localisation` computed over **truly-leaky** items
(an undetected leak scores zero for that item). Every component is also exposed
separately so the blend is never a black box.

## Consequences
- `always_leak` scores F1 0.714 but aggregate 0.417 — the metric rewards
  *diagnosis* and *localisation*, not just raised alarms.
- The weights are an editorial choice (detection necessary, not sufficient) and
  are documented; changing them is a conscious, visible decision.
- Clean items still matter: false positives lower precision and therefore F1,
  so over-flagging is penalised even though clean items have no diagnosis
  component.
