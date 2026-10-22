"""LeakBench -- a labelled benchmark for data-leakage detection.

Each benchmark item is a realistic ML *task specification* (domain, target,
prediction time, split, preprocessing, and natural-language feature
descriptions). The task for a solver is to decide whether the spec contains a
data-leakage flaw, classify the *type* of leak, and point at the offending
feature(s) -- exactly the judgement call that separates an honest evaluation
from an impressive-but-wrong one.

The package ships:
  * a typed schema with validation (``schema``);
  * a deterministic grader (``grader``);
  * reproducible, dependency-free baselines (``baselines``);
  * an LLM-judge adapter *interface* for plugging in a real model yourself
    (``llm_adapter``) -- no network calls live in this repo;
  * a leaderboard assembler and SVG figures.

Public/synthetic content only. No private or employer data.
"""

__version__ = "1.0.0"
