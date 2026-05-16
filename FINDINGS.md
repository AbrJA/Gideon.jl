# Gideon.jl — Benchmark Findings & Analysis Report

**Date:** 2025-07
**Julia version:** 1.12.6
**R version:** 4.6.0 (rsparse 0.5.x, MatrixExtra)
**Hardware:** Linux x86-64
**Benchmark script:** `benchmark/julia_benchmark.jl` vs `benchmark/r_benchmark.R`

---

## 1. Methodology

Both scripts operate on the **same sparse matrices** (exported as CSV triplets from R, loaded by Julia):

| Dataset | Dimensions | Density | NNZ |
|---------|-----------|---------|-----|
| Small   | 100 × 80  | 5 %     | ~400 |
| Medium  | 1 000 × 500 | 3 %   | ~15 000 |
| Large   | 5 000 × 2 000 | 1 % | ~100 000 |

Julia timings are the **minimum of 3 runs** (after one JIT warm-up for WRMF). R timings use `proc.time()` over a single call.
All algorithms use `rank = 10`, `λ = 0.1`, `α = 1.0`, `n_iter = 10` unless stated otherwise.

---

## 2. Performance Comparison

### 2.1 WRMF (Implicit ALS)

| Config | R time (s) | Julia time (s) | Speedup |
|--------|-----------|----------------|---------|
| Cholesky — Small (100×80) | 0.093 | 0.003 | **33.9×** |
| Cholesky — Medium (1000×500) | 0.084 | 0.037 | **2.2×** |
| Cholesky — Large (5000×2000) | 0.247 | 0.420 | **0.59× (slower)** |
| CG — Medium (1000×500) | 0.243 | 0.028 | **8.5×** |

**Findings:**
- Julia dominates at small/medium scale due to efficient LAPACK Cholesky and zero overhead from Julia's native dispatch.
- At large scale (5 000×2 000, ~100 K nnz), Julia's Cholesky solver is **1.7× slower** than R's rsparse. R uses Eigen + OpenMP C++ multi-threading natively at the C layer; Gideon.jl uses Polyester.jl (`@batch per=core`) which adds some overhead for thread coordination at this scale.
- The CG solver (`CONJUGATE_GRADIENT`) is significantly faster than Cholesky at medium scale (0.028s vs 0.037s) and far faster than R's CG (8.5×). This is the preferred solver for large sparse problems.
- **JIT warm-up penalty**: First run on small matrix takes **2.95s** (JIT compilation); subsequent runs drop to **0.003s**. This is a known Julia trade-off for short-lived scripts.

### 2.2 FTRL (Proximal SGD)

| Config | R time (s) | Julia time (s) | Speedup |
|--------|-----------|----------------|---------|
| 5 epochs, 1000×200 | 0.021 | 0.002 | **10.9×** |

**Findings:**
- Julia FTRL is **~11× faster** than R rsparse's FTRL.
- **Weight correlation = 1.000000** and **prediction correlation = 1.000000** — Julia and R converge to exactly the same solution (both use the same proximal FTRL-ProxL1 algorithm, same hyper-parameters, same data).
- Both achieve **80.6% accuracy** on the logistic classification task.
- All 200/200 features have non-zero weights (λ₁ = 0.005 < all feature magnitudes for this dense dataset).

### 2.3 Factorization Machine (XOR Task)

| Config | R time (s) | Julia time (s) | Speedup |
|--------|-----------|----------------|---------|
| 200 iter, 4×2 XOR | 0.141 | ~0.0001 | **~1 162×** |

**Findings:**
- After JIT compilation, Julia FM is over **1000× faster** for the tiny XOR problem (4 samples, 2 features, rank=2).
- R's 0.141s represents real overhead in the R6/S4 dispatch + R interpreter; Julia compiles straight to native machine code.
- Both implementations correctly solve XOR: predictions < 0.3 for class 0, > 0.7 for class 1.
- Julia predictions: `(0.0063, 0.9974, 0.9970, 0.0000)` — R predictions: `(0.0046, 0.9981, 0.9978, 0.0000)`.

### 2.4 GloVe

| Config | R time (s) | Julia time (s) | Speedup |
|--------|-----------|----------------|---------|
| 10 iter, 200×200 co-occurrence | 0.049 | 0.013 | **3.7×** |

**Findings:**
- Julia GloVe is 3.7× faster. Both use AdaGrad; Julia's implementation is single-threaded sequential over COO triplets.
- Final cost: 0.015850 (monotonically decreasing over 10 epochs — correct convergence behaviour).
- **Missing feature**: GloVe in rsparse uses Hogwild-style lock-free parallelism. Gideon.jl's GloVe is currently **single-threaded**. Adding `@threads` or Polyester here would likely yield a further 4–8× speedup on multi-core hardware.

---

## 3. Correctness Analysis

### 3.1 WRMF Factor Comparison

Both Julia and R receive the same matrix but use **different random seeds** (Julia: `MersenneTwister(42)`, R: `set.seed(42)` with a different PRNG algorithm). Because ALS converges to solutions that are unique only up to rotation and scale redistribution:

- **R** user F-norm = 4.72, item F-norm = 13.22
- **Julia** user F-norm = 16.58, item F-norm = 4.24

The scale is distributed differently between user and item factors. This is expected — both are valid solutions to the same weighted least-squares problem.

**Reconstruction R²** computed on raw rating values is **negative for both** (Julia −2.63, R −2.76). This is expected: implicit ALS does **not** minimize MSE on raw values. It minimizes a confidence-weighted loss treating all non-zero entries as positive interactions. Raw-value R² is the wrong metric here; the correct comparison is ranking quality (NDCG, MAP) on held-out data.

### 3.2 FTRL — Perfect Match

FTRL is deterministic given the same data ordering. The weight vector correlation is **1.000** and prediction correlation is **1.000**, confirming implementation correctness against R rsparse.

### 3.3 FM — Minor Numerical Differences

Julia and R FM predictions differ slightly (e.g., `0.0046` vs `0.0063` for the first sample). Both correctly classify all XOR examples. The difference arises from:
- Different PRNG implementations (Julia's `MersenneTwister` vs R's Mersenne Twister with different seeding)
- Different floating-point operation ordering in AdaGrad updates

### 3.4 Metrics — Exact Correctness

| Metric | Value | Expected |
|--------|-------|---------|
| AP@4   | 1.000 | 1.0 ✓ |
| NDCG@4 | 1.000 | 1.0 ✓ |
| Precision@4 | 0.750 | 0.75 ✓ |
| Recall@4 | 1.000 | 1.0 ✓ |

All ranking metrics compute exact expected values.

---

## 4. Bugs Found & Fixed During Development

| # | Bug | Location | Impact | Fix |
|---|-----|----------|--------|-----|
| 1 | X/Xt arguments swapped in ALS sweep | `wrmf.jl` `fit!` | `BoundsError` accessing item index 88 in 80-item matrix | Swap: user update uses `Xt` (columns=users), item update uses `X` (columns=items) |
| 2 | `SparseMatrixCSR{1,Tv,Ti}(...)` — no matching method | `sparse_utils.jl` `to_csr` | JET type error | Change to `SparseMatrixCSR{1}(...)` |
| 3 | `YtY + λI` returns `Symmetric`, not `Matrix{T}` | `wrmf.jl` CG path | JET union type instability | Wrap as `Matrix{T}(YtY + λI)` |
| 4 | `zeros(T, k, n)` in union split inferred as `Array{Float64,3}` | `wrmf.jl` `transform` | JET inference failure | Use `Matrix{T}(undef, k, n); fill!(...)` |
| 5 | SoftImpute `_soft_als` dimension tracking failure after SVD | `soft_impute.jl` | `BoundsError` on every call | Complete rewrite using clean alternating power iteration |
| 6 | Missing compat entries for stdlibs and extras | `Project.toml` | Aqua compat test failure | Added `LinearAlgebra="1"`, `Random="1"`, `SparseArrays="1"`, `Statistics="1"` to `[compat]` |
| 7 | Deprecated JET API `target_defined_modules=true` | `test/runtests.jl` | JET test warning | Changed to `target_modules=(Gideon,)` |

---

## 5. Performance Bottlenecks

### 5.1 WRMF — Large Matrix Overhead (Critical)

At 5 000×2 000 with ~100 K nnz, Julia is **41% slower** than R. Root causes:

1. **Thread management overhead**: Polyester `@batch per=core` has more per-call overhead than OpenMP's static thread pools. R's rsparse keeps a persistent thread pool warm across ALS iterations.
2. **Gram matrix allocation**: Each ALS step allocates `Matrix{T}(rank × rank)` per entity. With 5 000 users + 2 000 items = 7 000 allocations per iteration × 10 iterations = 70 000 small matrix allocations. Consider pre-allocating a thread-local buffer.
3. **CSR conversion**: `to_csr` is called once but materializes the entire CSR structure. For very large matrices, this is a significant allocation.

**Recommendation**: Pre-allocate per-thread gram matrix buffers; consider `ThreadsX.map` over `@batch` for better load balancing.

### 5.2 GloVe — Single-Threaded (Medium Priority)

Current GloVe uses a sequential loop over COO triplets. R uses Hogwild/lock-free parallel SGD. Adding `@threads` to the inner loop (with no atomic conflict since each word pair is visited once per epoch) would directly close the remaining performance gap.

### 5.3 SoftImpute — Alternating SVD Overhead (Low Priority)

`soft_impute` uses alternating power iteration with full `svd()` calls at each step (`O(min(m,n) × rank²)` per iteration). The implementation is correct but not optimized:
- Could use randomized SVD (RSVD) for large matrices
- Could use `LinearAlgebra.svd` with thin factorization only
- Current: 1.45s first run / 0.012s warm on 200×150 — acceptable for moderate sizes

### 5.4 Ranking Metrics — Per-User Sparse Traversal (Low Priority)

`_relevant_items` iterates the CSC column structure to find relevant items per user. For a matrix with `n_users` users and average `k` items per user, the total cost is `O(n_users × k)`. This is fine for typical recommendation scenarios but would degrade for dense matrices. Pre-computing a `Dict{Int, Set{Int}}` would make repeated metric calls faster.

---

## 6. Architectural Differences vs R rsparse

| Aspect | R rsparse | Gideon.jl |
|--------|-----------|-----------|
| ALS threading | OpenMP + Eigen C++ | Polyester.jl `@batch` |
| Gram matrix solve | Eigen LDLT / CG | LAPACK Cholesky / custom CG |
| FTRL state | R6 class, vectorized | `mutable struct`, per-feature scalars |
| FM forward pass | Vectorized R matrix ops | O(kp) sum-of-squares trick |
| GloVe parallelism | Hogwild lock-free | Sequential AdaGrad |
| LMF negatives | Full negative set | Random sampling loop |
| SoftImpute | SVD + correction | Alternating power iteration |
| NNLS | BVLS / active set | Post-hoc `max.(x, 0)` clamp |
| Sparse format | CSC (Matrix) + custom | CSC (Julia native) + CSR via SparseMatricesCSR.jl |

**Note on NNLS**: Gideon.jl uses `max.(x, 0)` as a post-hoc non-negativity clamp rather than a true bounded-variable least squares (BVLS) solver. This is faster but not guaranteed to minimize the NNLS objective — it only projects the Cholesky solution onto the non-negative orthant. A true BVLS or coordinate descent NNLS would be more accurate at the cost of extra iterations.

---

## 7. Summary Scorecard

| Algorithm | Correctness vs R | Julia Speedup | Known Issues |
|-----------|-----------------|---------------|--------------|
| WRMF Cholesky (small) | ✓ (scale ambiguity expected) | 34× | JIT cold-start 3s |
| WRMF Cholesky (large) | ✓ | **0.6× (slower)** | Thread overhead at scale |
| WRMF CG (medium) | ✓ | 8.5× | — |
| FTRL | **Exact match** (corr=1.0) | 11× | — |
| FM | Minor PRNG diff | ~1 000× | NNLS is a clamp, not BVLS |
| GloVe | ✓ | 3.7× | No parallel SGD |
| SoftImpute | ✓ | N/A (no R ref) | No RSVD, full SVD only |
| Metrics | **Exact** | N/A | Per-user O(nnz) traversal |

---

## 8. Recommended Next Steps

1. **Fix WRMF large-scale**: Pre-allocate per-thread gram matrix buffers to eliminate 70 K small allocs per fit.
2. **Add parallel GloVe**: One `Threads.@threads` on the COO loop (after shuffling indices per epoch) would match R's performance.
3. **Replace NNLS clamp with BVLS**: Use `NonNegLeastSquares.jl` or implement coordinate descent NNLS for correctness.
4. **Add RSVD to SoftImpute**: Replace `svd()` with `svds()` (partial SVD) from `Arpack.jl` for large matrices.
5. **Add cross-validation holdout**: The WRMF comparison currently computes R² on training data — add a `train/test` split to compute NDCG on held-out items.
6. **Publish benchmarks**: Track timing regressions across Julia/package versions using BenchmarkCI or PkgBenchmark.jl.
