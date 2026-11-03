"""Tests for stratified val sampling used by eval/review scripts.

Run:  python -m pytest tests/test_sample_balanced.py -q
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.pair_dataset import sample_balanced_indices  # noqa: E402


def _df(n=100, gens=("flux", "qwen")):
    return pd.DataFrame({
        "generator": [gens[i % len(gens)] for i in range(n)],
        "prompt": [f"p{i}" for i in range(n)],
    })


def test_balanced_two_strata():
    df = _df(100)
    index = list(range(100))
    idxs = sample_balanced_indices(df, index, 64, seed=0, strata_col="generator")
    assert len(idxs) == 64
    assert len(set(idxs)) == 64
    counts = {g: sum(1 for i in idxs if df["generator"].iloc[index[i]] == g)
              for g in ("flux", "qwen")}
    assert counts == {"flux": 32, "qwen": 32}


def test_reproducible_and_disjoint_from_seed():
    df = _df(50)
    index = list(range(50))
    a = sample_balanced_indices(df, index, 20, seed=1, strata_col="generator")
    b = sample_balanced_indices(df, index, 20, seed=1, strata_col="generator")
    c = sample_balanced_indices(df, index, 20, seed=2, strata_col="generator")
    assert a == b
    assert a != c


def test_n_ge_len_returns_all_positions():
    df = _df(10)
    index = list(range(10))
    idxs = sample_balanced_indices(df, index, 99, seed=0, strata_col="generator")
    assert sorted(idxs) == list(range(10))


def test_missing_strata_falls_back_to_random():
    df = pd.DataFrame({"prompt": [f"p{i}" for i in range(40)]})
    index = list(range(40))
    idxs = sample_balanced_indices(df, index, 8, seed=0, strata_col="generator")
    assert len(idxs) == 8
    assert len(set(idxs)) == 8


def test_single_stratum():
    df = pd.DataFrame({"generator": ["flux"] * 30})
    index = list(range(30))
    idxs = sample_balanced_indices(df, index, 10, seed=0, strata_col="generator")
    assert len(idxs) == 10
    assert len(set(idxs)) == 10


def test_uneven_strata_proportional():
    df = pd.DataFrame({"generator": ["flux"] * 75 + ["qwen"] * 25})
    index = list(range(100))
    idxs = sample_balanced_indices(df, index, 40, seed=0, strata_col="generator")
    counts = {g: sum(1 for i in idxs if df["generator"].iloc[index[i]] == g)
              for g in ("flux", "qwen")}
    assert counts["flux"] >= counts["qwen"]
    assert sum(counts.values()) == 40
