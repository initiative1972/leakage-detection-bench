"""Central configuration: paths, the leakage taxonomy, and the contamination
canary.
"""
from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "docs" / "figures"

DATASET_PATH = DATA_DIR / "benchmark.jsonl"
SCHEMA_VERSION = "1.0.0"

# The leakage taxonomy. "none" denotes a clean, well-posed spec.
LEAK_TYPES = (
    "target_leakage",       # feature is a consequence/proxy of the label
    "temporal_leakage",     # feature uses information from after prediction time
    "preprocessing_leakage",  # transform fit on all data before the split
    "group_leakage",        # same entity in train and test (row-level split)
    "proxy_leakage",        # an id / ordering / code that encodes the label
    "none",                 # no leakage
)
POSITIVE_TYPES = tuple(t for t in LEAK_TYPES if t != "none")

DIFFICULTIES = ("easy", "medium", "hard")

# A fixed canary embedded in every record so that, if this dataset is ever
# scraped into a training corpus, contamination can be detected by searching
# models for this exact string. Do NOT train on LeakBench.
CANARY = "LEAKBENCH-CANARY-6f1d9a2c-7b84-4e0a-9c3e-benchmark-do-not-train"
