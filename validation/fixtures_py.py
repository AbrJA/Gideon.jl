#!/usr/bin/env python3
"""
Generate Python reference fixtures for Gideon validation.

Usage:
    python3 validation/fixtures_py.py

Outputs:
    /tmp/gideon_fixtures/python/
    - X_small.csv
    - X_small_dims.csv
    - X_train.csv
    - X_train_dims.csv
    - X_test.csv
    - X_test_dims.csv
    - py_als_scores.csv
    - py_bpr_scores.csv
        - py_lmf_scores.csv (optional)
        - py_ease_B.csv
    - py_bpr_metrics.json
    - py_lmf_metrics.json (optional)
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


def ndcg_at_k(preds: np.ndarray, test: csr_matrix, k: int = 10) -> float:
    vals = []
    for u in range(preds.shape[0]):
        relevant = set(test[u].indices.tolist())
        if not relevant:
            continue
        dcg = 0.0
        for r, item in enumerate(preds[u, :k]):
            if int(item) in relevant:
                dcg += 1.0 / np.log2(r + 2.0)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2.0) for i in range(ideal_hits))
        vals.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def recall_at_k(preds: np.ndarray, test: csr_matrix, k: int = 10) -> float:
    vals = []
    for u in range(preds.shape[0]):
        relevant = set(test[u].indices.tolist())
        if not relevant:
            continue
        hits = sum(1 for item in preds[u, :k] if int(item) in relevant)
        vals.append(hits / len(relevant))
    return float(np.mean(vals)) if vals else 0.0


def split_train_test(x: csr_matrix, rng: np.random.Generator, test_frac: float = 0.2):
    x = x.tocsr()
    n_users, n_items = x.shape
    train_rows: list[int] = []
    train_cols: list[int] = []
    train_vals: list[float] = []
    test_rows: list[int] = []
    test_cols: list[int] = []
    test_vals: list[float] = []

    for u in range(n_users):
        start, end = x.indptr[u], x.indptr[u + 1]
        cols_u = x.indices[start:end]
        vals_u = x.data[start:end]
        nnz_u = len(cols_u)
        if nnz_u == 0:
            continue

        n_test = max(1, int(np.floor(nnz_u * test_frac))) if nnz_u > 1 else 0
        perm = rng.permutation(nnz_u)
        test_idx = set(perm[:n_test].tolist())

        for j in range(nnz_u):
            c = int(cols_u[j])
            v = float(vals_u[j])
            if j in test_idx:
                test_rows.append(u)
                test_cols.append(c)
                test_vals.append(v)
            else:
                train_rows.append(u)
                train_cols.append(c)
                train_vals.append(v)

    train = csr_matrix((np.array(train_vals, dtype=np.float32),
                        (np.array(train_rows, dtype=np.int32), np.array(train_cols, dtype=np.int32))),
                       shape=(n_users, n_items))
    test = csr_matrix((np.array(test_vals, dtype=np.float32),
                       (np.array(test_rows, dtype=np.int32), np.array(test_cols, dtype=np.int32))),
                      shape=(n_users, n_items))
    train.sum_duplicates()
    test.sum_duplicates()
    return train, test


def topk_from_scores(scores: np.ndarray, train: csr_matrix | None = None, k: int = 10) -> np.ndarray:
    s = scores.copy()
    if train is not None:
        tr = train.tocsr()
        for u in range(tr.shape[0]):
            start, end = tr.indptr[u], tr.indptr[u + 1]
            if start < end:
                s[u, tr.indices[start:end]] = -np.inf
    part = np.argpartition(-s, kth=k - 1, axis=1)[:, :k]
    row = np.arange(s.shape[0])[:, None]
    order = np.argsort(-s[row, part], axis=1)
    return part[row, order]


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

    train, test = split_train_test(x, rng, test_frac=0.2)

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

    # Optional Logistic Matrix Factorization parity fixture.
    lmf_scores = None
    lmf_class = None
    try:
        lmf_mod = getattr(implicit, "lmf", None)
        if lmf_mod is not None and hasattr(lmf_mod, "LogisticMatrixFactorization"):
            lmf_class = lmf_mod.LogisticMatrixFactorization
            lmf = lmf_class(
                factors=rank,
                learning_rate=0.01,
                regularization=0.01,
                iterations=30,
                random_state=seed,
            )
            lmf.fit(x)
            lmf_scores = lmf.user_factors @ lmf.item_factors.T
    except Exception:
        # Keep fixture generation resilient; validate script will skip LogisticMF if absent.
        lmf_scores = None

    # Deterministic Python EASE reference implementation.
    ease_lambda = 100.0
    gram = (x.T @ x).toarray().astype(np.float64)
    n_items = gram.shape[0]
    gram[np.diag_indices(n_items)] += ease_lambda
    p = np.linalg.inv(gram)
    b = -p / np.diag(p)
    b[np.diag_indices(n_items)] = 0.0

    write_triplet(x, out_dir / "X_small.csv")
    with (out_dir / "X_small_dims.csv").open("w", encoding="utf-8") as f:
        f.write("nr,nc\n")
        f.write(f"{n_users},{n_items}\n")

    np.savetxt(out_dir / "py_als_scores.csv", als_scores, delimiter=",")
    np.savetxt(out_dir / "py_bpr_scores.csv", bpr_scores, delimiter=",")
    np.savetxt(out_dir / "py_ease_B.csv", b, delimiter=",")
    if lmf_scores is not None:
        np.savetxt(out_dir / "py_lmf_scores.csv", lmf_scores, delimiter=",")

    write_triplet(train, out_dir / "X_train.csv")
    with (out_dir / "X_train_dims.csv").open("w", encoding="utf-8") as f:
        f.write("nr,nc\n")
        f.write(f"{n_users},{n_items}\n")
    write_triplet(test, out_dir / "X_test.csv")
    with (out_dir / "X_test_dims.csv").open("w", encoding="utf-8") as f:
        f.write("nr,nc\n")
        f.write(f"{n_users},{n_items}\n")

    bpr_preds = topk_from_scores(bpr_scores, train=train, k=10)
    bpr_metrics = {
        "k": 10,
        "ndcg": ndcg_at_k(bpr_preds, test, k=10),
        "recall": recall_at_k(bpr_preds, test, k=10),
    }
    (out_dir / "py_bpr_metrics.json").write_text(json.dumps(bpr_metrics, indent=2), encoding="utf-8")

    if lmf_scores is not None:
        lmf_preds = topk_from_scores(lmf_scores, train=train, k=10)
        lmf_metrics = {
            "k": 10,
            "ndcg": ndcg_at_k(lmf_preds, test, k=10),
            "recall": recall_at_k(lmf_preds, test, k=10),
        }
        (out_dir / "py_lmf_metrics.json").write_text(json.dumps(lmf_metrics, indent=2), encoding="utf-8")

    meta = {
        "seed": seed,
        "n_users": n_users,
        "n_items": n_items,
        "density": density,
        "rank": rank,
        "test_fraction": 0.2,
        "als_iters": als_iters,
        "bpr_iters": bpr_iters,
        "bpr_class": bpr_cls.__name__,
        "lmf_available": lmf_scores is not None,
        "lmf_class": None if lmf_class is None else lmf_class.__name__,
        "ease_lambda": ease_lambda,
        "implicit_version": getattr(implicit, "__version__", "unknown"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[OK] Python fixtures generated:")
    print(f"  {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
