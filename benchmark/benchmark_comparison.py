#!/usr/bin/env python3
"""
benchmark/benchmark_comparison.py — Python (implicit) benchmark counterpart.

Run:
    python benchmark/benchmark_comparison.py

Generates benchmark/results_python.csv for comparison with Julia results.
"""
import os
import time
import csv
import numpy as np
from scipy.sparse import csr_matrix, random as sp_random

import implicit

print("=" * 70)
print("Python (implicit) Performance Benchmark")
print("=" * 70)
print(f"  implicit version: {implicit.__version__}")
print(f"  CPU count: {os.cpu_count()}")
print()

# ─────────────────────────────────────────────
# Generate synthetic sparse matrices at 3 scales
# ─────────────────────────────────────────────
def generate_matrix(n_users: int, n_items: int, density: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    nnz_target = int(n_users * n_items * density)
    rows = rng.integers(0, n_users, size=nnz_target)
    cols = rng.integers(0, n_items, size=nnz_target)
    vals = np.ones(nnz_target, dtype=np.float32)
    X = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    X.sum_duplicates()
    return X


MATRIX_CONFIGS = [
    {"name": "hundreds",   "n_users": 500,     "n_items": 300,     "density": 0.10},   # ~15K nnz
    {"name": "thousands",  "n_users": 5_000,   "n_items": 3_000,   "density": 0.02},   # ~300K nnz
    {"name": "millions",   "n_users": 100_000, "n_items": 50_000,  "density": 0.001},  # ~5M nnz
]


def get_benchmarks():
    """Return algorithm configurations matching the Julia benchmark."""
    return [
        {
            "name": "ALS",
            "model": lambda: implicit.als.AlternatingLeastSquares(
                factors=64, regularization=0.1, alpha=40.0,
                iterations=10, random_state=42, use_gpu=False,
            ),
        },
        {
            "name": "ALS-CG",
            "model": lambda: implicit.als.AlternatingLeastSquares(
                factors=64, regularization=0.1, alpha=40.0,
                iterations=10, random_state=42, use_gpu=False,
            ),
        },
        {
            "name": "BPR",
            "model": lambda: _make_bpr(),
        },
        {
            "name": "LogisticMF",
            "model": lambda: _make_lmf(),
        },
    ]


def _make_bpr():
    bpr_cls = None
    if hasattr(implicit.bpr, "BPR"):
        bpr_cls = implicit.bpr.BPR
    elif hasattr(implicit.bpr, "BayesianPersonalizedRanking"):
        bpr_cls = implicit.bpr.BayesianPersonalizedRanking
    else:
        raise RuntimeError("No compatible BPR class found")
    return bpr_cls(
        factors=64, regularization=0.01, learning_rate=0.05,
        iterations=10, random_state=42, use_gpu=False,
    )


def _make_lmf():
    lmf_mod = getattr(implicit, "lmf", None)
    if lmf_mod is None or not hasattr(lmf_mod, "LogisticMatrixFactorization"):
        raise RuntimeError("LogisticMatrixFactorization not available")
    return lmf_mod.LogisticMatrixFactorization(
        factors=64, learning_rate=1.0, regularization=0.6,
        iterations=10, random_state=42,
    )


# ─────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────
def run_benchmarks():
    results = []
    benchmarks = get_benchmarks()

    for cfg in MATRIX_CONFIGS:
        print("-" * 50)
        print(f"Matrix: {cfg['name']} ({cfg['n_users']} × {cfg['n_items']}, "
              f"density={cfg['density']:.3f})")
        print("-" * 50)

        X = generate_matrix(cfg["n_users"], cfg["n_items"], cfg["density"])
        print(f"  Generated: {X.shape[0]} users × {X.shape[1]} items, nnz={X.nnz}")

        for b in benchmarks:
            try:
                model = b["model"]()
            except Exception as e:
                print(f"  {b['name']:<15}  SKIPPED ({e})")
                continue

            # Benchmark
            t0 = time.perf_counter()
            model.fit(X)
            elapsed = time.perf_counter() - t0

            print(f"  {b['name']:<15}  {elapsed:8.3f} s")

            results.append({
                "matrix": cfg["name"],
                "n_users": cfg["n_users"],
                "n_items": cfg["n_items"],
                "nnz": X.nnz,
                "algorithm": b["name"],
                "time_seconds": round(elapsed, 4),
                "n_iters": 10,
            })

        print()

    return results


# ─────────────────────────────────────────────
# Run and save results
# ─────────────────────────────────────────────
if __name__ == "__main__":
    results = run_benchmarks()

    outpath = os.path.join(os.path.dirname(__file__), "results_python.csv")
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "matrix", "n_users", "n_items", "nnz", "algorithm", "time_seconds", "n_iters"
        ])
        writer.writeheader()
        writer.writerows(results)

    print("=" * 70)
    print(f"Results saved to: {outpath}")
    print("=" * 70)

    # Summary table
    print("\nSummary:")
    print("-" * 60)
    print(f"{'Matrix':<8} {'Algorithm':<15} {'Time (s)':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['matrix']:<8} {r['algorithm']:<15} {r['time_seconds']:>10.3f}")
    print("-" * 60)
