"""
benchmark/bench_ml1m_python.py — Python implicit benchmark on ML-1M
Run: python benchmark/bench_ml1m_python.py
"""
import time
import os
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from collections import defaultdict

print("=" * 70)
print("Python (implicit) Benchmark — MovieLens 1M")
print("=" * 70)

import implicit
print(f"  implicit version: {implicit.__version__}")
print(f"  CPU count: {os.cpu_count()}")
print()

# Load ML-1M
print("[Loading ML-1M...]")
t0 = time.time()
users, items = [], []
with open("usage/ml-1m/ratings.dat") as f:
    for line in f:
        parts = line.strip().split("::")
        users.append(int(parts[0]))
        items.append(int(parts[1]))

# Re-index
u_map, i_map = {}, {}
for u in users:
    if u not in u_map:
        u_map[u] = len(u_map)
for i in items:
    if i not in i_map:
        i_map[i] = len(i_map)

rows = np.array([u_map[u] for u in users], dtype=np.int32)
cols = np.array([i_map[i] for i in items], dtype=np.int32)
vals = np.ones(len(rows), dtype=np.float32)
n_users, n_items = len(u_map), len(i_map)
X = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
print(f"  Loaded in {time.time()-t0:.2f} s")
print(f"  Matrix: {n_users} users × {n_items} items, nnz={X.nnz}")

# Train/test split
print("[Splitting train/test...]")
rng = np.random.RandomState(42)
train = lil_matrix((n_users, n_items), dtype=np.float32)
test = lil_matrix((n_users, n_items), dtype=np.float32)

for u in range(n_users):
    items_u = X[u].indices.tolist()
    n_test = max(1, int(len(items_u) * 0.2))
    perm = rng.permutation(items_u)
    test_items = set(perm[:n_test])
    for i in items_u:
        if i in test_items:
            test[u, i] = 1.0
        else:
            train[u, i] = 1.0

X_train = csr_matrix(train)
X_test = csr_matrix(test)
print(f"  Train: nnz={X_train.nnz}, Test: nnz={X_test.nnz}")
print()


def ndcg_at_k(predictions, test_csr, k=10):
    n_users = predictions.shape[0]
    ndcgs = np.zeros(n_users)
    for u in range(n_users):
        relevant = set(test_csr[u].indices)
        if not relevant:
            continue
        dcg = 0.0
        for rank, item in enumerate(predictions[u, :k]):
            if item in relevant:
                dcg += 1.0 / np.log2(rank + 2)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        if idcg > 0:
            ndcgs[u] = dcg / idcg
    return float(np.mean(ndcgs))


def recall_at_k(predictions, test_csr, k=10):
    n_users = predictions.shape[0]
    recalls = np.zeros(n_users)
    for u in range(n_users):
        relevant = set(test_csr[u].indices)
        if not relevant:
            continue
        hits = sum(1 for item in predictions[u, :k] if item in relevant)
        recalls[u] = hits / len(relevant)
    return float(np.mean(recalls))


def predict_top_k(model, user_items, k=10):
    n_users = user_items.shape[0]
    ids, _ = model.recommend(
        np.arange(n_users), user_items, N=k, filter_already_liked_items=True
    )
    return ids


def benchmark_bpr(X_train, X_test, rank=64, n_iter=50, lr=0.05, reg=0.01):
    print("-" * 70)
    print(f"BPR — rank={rank}, iters={n_iter}, lr={lr}")
    print("-" * 70)

    model = implicit.bpr.BPR(
        factors=rank, regularization=reg, learning_rate=lr,
        iterations=n_iter, use_gpu=False, random_state=42
    )

    t_train = time.time()
    model.fit(X_train)
    t_train = time.time() - t_train
    print(f"  Train time: {t_train:.2f} s ({t_train/n_iter:.3f} s/iter)")

    t_pred = time.time()
    preds = predict_top_k(model, X_train, k=10)
    t_pred = time.time() - t_pred
    print(f"  Predict time: {t_pred:.2f} s")

    ndcg = ndcg_at_k(preds, X_test, 10)
    rec = recall_at_k(preds, X_test, 10)
    print(f"  NDCG@10:   {ndcg:.4f}")
    print(f"  Recall@10: {rec:.4f}")
    print()
    return {"train": t_train, "predict": t_pred, "ndcg": ndcg, "recall": rec}


def benchmark_als(X_train, X_test, rank=64, n_iter=15, alpha=40.0, reg=0.1):
    print("-" * 70)
    print(f"ALS — rank={rank}, iters={n_iter}, α={alpha}")
    print("-" * 70)

    model = implicit.als.AlternatingLeastSquares(
        factors=rank, regularization=reg, alpha=alpha,
        iterations=n_iter, use_gpu=False, random_state=42
    )

    t_train = time.time()
    model.fit(X_train)
    t_train = time.time() - t_train
    print(f"  Train time: {t_train:.2f} s ({t_train/n_iter:.3f} s/iter)")

    t_pred = time.time()
    preds = predict_top_k(model, X_train, k=10)
    t_pred = time.time() - t_pred
    print(f"  Predict time: {t_pred:.2f} s")

    ndcg = ndcg_at_k(preds, X_test, 10)
    rec = recall_at_k(preds, X_test, 10)
    print(f"  NDCG@10:   {ndcg:.4f}")
    print(f"  Recall@10: {rec:.4f}")
    print()
    return {"train": t_train, "predict": t_pred, "ndcg": ndcg, "recall": rec}


# Run benchmarks
print("=" * 70)
print("BENCHMARKS")
print("=" * 70)

results = {}
for rank in [32, 64]:
    results[f"BPR_r{rank}"] = benchmark_bpr(X_train, X_test, rank=rank, n_iter=50)

for rank in [32, 64]:
    results[f"ALS_r{rank}"] = benchmark_als(X_train, X_test, rank=rank)

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Algorithm':<20} {'Train(s)':>8} {'Pred(s)':>8} {'NDCG@10':>8} {'Recall@10':>10}")
print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
for name in sorted(results.keys()):
    r = results[name]
    print(f"{name:<20} {r['train']:8.2f} {r['predict']:8.2f} {r['ndcg']:8.4f} {r['recall']:10.4f}")
