#!/usr/bin/env python3
"""
Generate Python reference fixtures for Gideon validation.

Usage:
    python3 validation/fixtures_py.py

Outputs:
    /tmp/gideon_fixtures/python/
    - X_small.csv
    - X_small_dims.csv
    - py_als_scores.csv
    - py_bpr_scores.csv
    - meta.json
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
from scipy.sparse import csr_matrix


def write_triplet(x: csr_matrix, path: pathlib.Path) -> None:
    coo = x.tocoo()
    with path.open("w", encoding="utf-8") as f:
        f.write("row,col,val\n")
        for r, c, v in zip(coo.row, coo.col, coo.data):
            f.write(f"{r + 1},{c + 1},{float(v)}\n")


def main() -> int:
    try:
        import implicit
    except Exception as exc:  # pragma: no cover
        print("[ERROR] Missing Python dependency: implicit")
        print("Install with: pip install implicit")
        print(f"Details: {exc}")
        return 2

    out_dir = pathlib.Path(
        os.environ.get("GIDEON_PY_FIXTURE_DIR", "/tmp/gideon_fixtures/python")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    rng = np.random.default_rng(seed)

    n_users = 120
    n_items = 100
    density = 0.06

    nnz = int(n_users * n_items * density)
    rows = rng.integers(0, n_users, size=nnz)
    cols = rng.integers(0, n_items, size=nnz)
    vals = rng.integers(1, 6, size=nnz).astype(np.float32)
    x = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
    x.sum_duplicates()

    rank = 16
    als_iters = 20
    bpr_iters = 40

    als = implicit.als.AlternatingLeastSquares(
        factors=rank,
        regularization=0.1,
        alpha=40.0,
        iterations=als_iters,
        random_state=seed,
        use_gpu=False,
    )
    als.fit(x)

    bpr_cls = None
    if hasattr(implicit.bpr, "BPR"):
        bpr_cls = implicit.bpr.BPR
    elif hasattr(implicit.bpr, "BayesianPersonalizedRanking"):
        bpr_cls = implicit.bpr.BayesianPersonalizedRanking
    else:
        raise RuntimeError("No compatible BPR class found in implicit.bpr")

    bpr = bpr_cls(
        factors=rank,
        regularization=0.01,
        learning_rate=0.05,
        iterations=bpr_iters,
        random_state=seed,
        use_gpu=False,
    )
    bpr.fit(x)

    als_scores = als.user_factors @ als.item_factors.T
    bpr_scores = bpr.user_factors @ bpr.item_factors.T

    write_triplet(x, out_dir / "X_small.csv")
    with (out_dir / "X_small_dims.csv").open("w", encoding="utf-8") as f:
        f.write("nr,nc\n")
        f.write(f"{n_users},{n_items}\n")

    np.savetxt(out_dir / "py_als_scores.csv", als_scores, delimiter=",")
    np.savetxt(out_dir / "py_bpr_scores.csv", bpr_scores, delimiter=",")

    meta = {
        "seed": seed,
        "n_users": n_users,
        "n_items": n_items,
        "density": density,
        "rank": rank,
        "als_iters": als_iters,
        "bpr_iters": bpr_iters,
        "bpr_class": bpr_cls.__name__,
        "implicit_version": getattr(implicit, "__version__", "unknown"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[OK] Python fixtures generated:")
    print(f"  {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
