<div align="center">

# Gideon.jl

**High-performance statistical learning on sparse matrices in pure Julia.**

[![Build Status](https://github.com/AbrJA/Gideon.jl/workflows/CI/badge.svg)](https://github.com/AbrJA/Gideon.jl/actions)
[![codecov](https://codecov.io/gh/AbrJA/Gideon.jl/graph/badge.svg)](https://codecov.io/gh/AbrJA/Gideon.jl)
[![Julia 1.9+](https://img.shields.io/badge/Julia-1.9%2B-blue?logo=julia)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

Gideon.jl is a pure-Julia port and enhancement of the R package [rsparse](https://github.com/dselivanov/rsparse), providing a unified, extensible interface for matrix factorization, sparse regression, and recommender-system evaluation. All algorithms are validated against R reference outputs and optimized for production scale via multithreading and SIMD vectorization.

## Features

- **Unified API** — `fit!` / `predict` / `transform` for every model; no framework lock-in.
- **Production-grade performance** — zero-allocation inner loops, `@inbounds @simd` vectorization, BLAS-2 gram updates, per-thread pre-allocated buffers.
- **R-validated correctness** — the full test suite includes a Tier-2 fixture layer that compares numerically against pre-computed R / rsparse outputs.
- **Sparse-native** — all algorithms operate directly on `SparseMatrixCSC`; no dense conversion needed.

---

## Algorithms

| Model | Type | Reference |
|-------|------|-----------|
| `WRMF` | Implicit / Explicit ALS | Hu, Koren & Volinsky (2008) |
| `LMF` | Logistic Matrix Factorization | Johnson (2014) |
| `GloVe` | Co-occurrence embedding | Pennington, Socher & Manning (2014) |
| `FTRL` | Follow The Regularized Leader (online logistic) | McMahan et al. (2013) |
| `FactorizationMachine` | 2nd-order FM | Rendle (2010) |
| `soft_impute` / `soft_svd` | Low-rank matrix completion | Hastie et al. (2014) |

---

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/AbrJA/Gideon.jl")
```

Requires Julia ≥ 1.9.

---

## Quick Start

### WRMF — Implicit Collaborative Filtering

```julia
using Gideon, SparseArrays, Random

# Build a user–item interaction matrix (n_users × n_items)
rng = MersenneTwister(42)
X = sprand(rng, 1000, 500, 0.02)   # 1 K users, 500 items, 2% density

# Train with Conjugate-Gradient ALS (default, fastest at scale)
model = WRMF(rank=20, λ=0.1, α=1.0, max_iter=15)
fit!(model, X; rng)

# User and item embeddings: rank × n matrix
size(model.user_factors)   # (20, 1000)
size(model.item_factors)   # (20, 500)

# Embed new users from their interaction history
X_new = sprand(rng, 50, 500, 0.03)
U_new = transform(model, X_new)    # (20, 50)
```

Switch to Cholesky for maximum numerical stability, or NNLS for non-negative factors:

```julia
model_chol = WRMF(rank=20, λ=0.1, solver=CHOLESKY)
model_nnls = WRMF(rank=20, λ=0.1, solver=NNLS)
```

---

### GloVe — Co-occurrence Embeddings

```julia
using Gideon, SparseArrays, Random

# Co-occurrence matrix must be square and positive (e.g. from a tokenizer)
C = sprand(MersenneTwister(1), 5000, 5000, 0.005)
C = C + C'   # symmetrize

glove = GloVe(rank=100, learning_rate=0.05, x_max=100.0)
fit!(glove, C; n_iter=20, rng=MersenneTwister(2))

# Final embeddings: average main + context vectors (standard GloVe convention)
E = get_embeddings(glove)   # 100 × 5000
```

---

### Logistic Matrix Factorization (LMF)

```julia
using Gideon, SparseArrays, Random

X = sprand(MersenneTwister(3), 800, 300, 0.03)

lmf = LMF(rank=15, α=1.0, λ=0.1, learning_rate=0.01, max_iter=20, n_negative=5)
fit!(lmf, X; rng=MersenneTwister(3))

size(lmf.user_factors)   # (15, 800)
size(lmf.item_factors)   # (15, 300)
```

---

### FTRL — Online Logistic Regression

FTRL supports Elastic-Net regularization and streaming/online updates via `partial_fit!`.

```julia
using Gideon, SparseArrays, Random

rng = MersenneTwister(7)
n, p = 10_000, 50_000
X_train = sprand(rng, n, p, 0.001)
y_train = rand(rng, Bool, n) .|> Float64

model = FTRL(
    learning_rate       = 0.1,
    learning_rate_decay = 0.5,
    λ                   = 1e-4,
    l1_ratio            = 0.9,   # mostly L1 (Lasso-like)
)

# Single pass — call multiple times for multiple epochs
partial_fit!(model, X_train, y_train; rng)

# Predict probabilities
ŷ = predict(model, X_train)   # Vector{Float64} ∈ (0, 1)

# Online update with a new mini-batch
X_new = sprand(rng, 200, p, 0.001)
y_new = rand(rng, Bool, 200) .|> Float64
partial_fit!(model, X_new, y_new; rng)
```

---

### Factorization Machines

```julia
using Gideon, SparseArrays, Random

rng = MersenneTwister(9)
X = sprand(rng, 5_000, 1_000, 0.01)
y = rand(rng, Bool, 5_000) .|> Float64

fm = FactorizationMachine(
    rank           = 8,
    learning_rate_w = 0.1,
    learning_rate_v = 0.05,
    λ_w            = 1e-5,
    λ_v            = 1e-5,
    family         = :binomial,
)

partial_fit!(fm, X, y; rng)
ŷ = predict(fm, X)
```

---

### SoftImpute — Low-rank Matrix Completion

```julia
using Gideon, SparseArrays, LinearAlgebra, Random

rng = MersenneTwister(11)
X_observed = sprand(rng, 200, 150, 0.3)   # only ~30% of entries observed

# Complete the matrix up to rank 10, nuclear-norm penalty λ=0.5
result = soft_impute(X_observed; rank=10, λ=0.5, n_iter=100)

# Low-rank approximation: result.U * Diagonal(result.d) * result.V'
recon = result.U * Diagonal(result.d) * result.V'
size(recon)   # (200, 150)

# Use soft_svd for a cleaner low-rank SVD (no imputation correction term)
svd_result = soft_svd(X_observed; rank=5, n_iter=50)
```

---

### Ranking Metrics

All metric functions accept a predictions matrix of shape `(n_users, K)` (item indices,
1-based) and a sparse relevance matrix.

```julia
using Gideon, SparseArrays, Random

rng = MersenneTwister(13)
n_users, n_items, K = 500, 2000, 20

# Ground-truth relevance (non-zero = relevant)
actual = sprand(rng, n_users, n_items, 0.02)

# Simulated top-K predictions (replace with your model's output)
preds = hcat([randperm(rng, n_items)[1:K] for _ in 1:n_users]...)'

ap   = ap_at_k(preds, actual; k=K)          # Vector{Float64}, length n_users
ndcg = ndcg_at_k(preds, actual; k=K)
prec = precision_at_k(preds, actual; k=K)
rec  = recall_at_k(preds, actual; k=K)

println("MAP@$K     = ", round(map_at_k(preds, actual; k=K), digits=4))
println("Mean NDCG@$K = ", round(mean(ndcg), digits=4))
```

---

## Architecture

```
Gideon.jl
├── src/
│   ├── Gideon.jl          # Module entry, exports
│   ├── types.jl           # Abstract hierarchy, ALSSolver / FeedbackType enums
│   ├── utils.jl           # init_factors, sigmoid, …
│   ├── sparse_utils.jl    # to_csr, dual_representation, row/col nnz
│   ├── algorithms/
│   │   ├── wrmf.jl        # Implicit/Explicit ALS (Cholesky · CG · NNLS)
│   │   ├── lmf.jl         # Logistic MF with negative sampling
│   │   ├── glove.jl       # GloVe Hogwild AdaGrad
│   │   ├── ftrl.jl        # Follow The Regularized Leader (online)
│   │   ├── fm.jl          # Factorization Machines
│   │   └── soft_impute.jl # SoftImpute / SoftSVD
│   └── metrics/
│       └── ranking.jl     # AP@K, MAP@K, NDCG@K, Precision@K, Recall@K
└── test/
    ├── runtests.jl
    └── r_correctness.jl   # Numerical validation against R / rsparse
```

### Type Hierarchy

```julia
AbstractSparseModel
├── AbstractMatrixFactorization   →  WRMF, LMF, GloVe, SoftImputeResult
└── AbstractSparseRegression      →  FTRL, FactorizationMachine
```

Every model implements the same generic interface:

| Function | Description |
|----------|-------------|
| `fit!(model, X)` | Train in-place on sparse matrix `X` |
| `partial_fit!(model, X, y)` | Online update (FTRL, FM) |
| `predict(model, X)` | Return predictions / scores |
| `transform(model, X)` | Return latent embeddings for new users |
| `coef(model)` | Return learned weight vector (FTRL) |

---

## Performance Design

| Technique | Where used |
|-----------|-----------|
| Pre-allocated per-thread Gram / RHS / Cholesky buffers | WRMF ALS sweep |
| `BLAS.syr!` rank-1 Gram accumulation | WRMF Cholesky solver |
| `BLAS.syrk!` item Gram `YᵀY` | WRMF both solvers |
| Fast-path manual SIMD dot (`@inbounds @simd`) for sparse users with < 32 nnz | WRMF CG `_implicit_matvec!` |
| `@inbounds @simd` vectorized dot / gradient loops | WRMF loss, LMF SGD, GloVe AdaGrad |
| CSR transpose `SparseMatrixCSC(actual')` — O(nnz_u) per-user row access | All ranking metrics |
| `Threads.@threads :static` outer loops | WRMF user / item sweeps |
| `LoopVectorization.jl` as SIMD infrastructure | All hot paths |

WRMF with Conjugate-Gradient at the MEGA scale (1 M users × 100 K items, ~10 nnz/user) benefits most from the fast sparse-dot path, which eliminates BLAS call overhead entirely for sparse rows.

---

## Testing

```bash
julia --project=. --threads=4,2 -e 'using Pkg; Pkg.test()'
```

The suite runs **164 tests** covering:

- Unit correctness (dimensions, NaN / Inf guards, convergence monotonicity)
- R / rsparse numerical fixture comparisons (weights, predictions, loss values)
- Static analysis via [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl) and [JET.jl](https://github.com/aviatesk/JET.jl)

---

## Dependencies

| Package | Role |
|---------|------|
| `SparseArrays` (stdlib) | Core sparse matrix type |
| `LinearAlgebra` (stdlib) | BLAS / LAPACK, SVD, Cholesky |
| `SparseMatricesCSR.jl` | CSR representation for row-oriented access |
| `StaticArrays.jl` | Stack-allocated small arrays |
| `LoopVectorization.jl` | SIMD vectorization infrastructure |

---

## License

MIT — see [LICENSE](LICENSE).
