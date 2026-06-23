<div align="center">

# Gideon.jl

**High-performance statistical learning on sparse matrices in pure Julia.**

[![Build Status](https://github.com/AbrJA/Gideon.jl/workflows/CI/badge.svg)](https://github.com/AbrJA/Gideon.jl/actions)
[![codecov](https://codecov.io/gh/AbrJA/Gideon.jl/graph/badge.svg)](https://codecov.io/gh/AbrJA/Gideon.jl)
[![Julia 1.9+](https://img.shields.io/badge/Julia-1.9%2B-blue?logo=julia)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

Gideon.jl is a pure-Julia port and enhancement of the R package [rsparse](https://github.com/dselivanov/rsparse), providing a unified, extensible interface for matrix factorization, sparse regression, and recommender-system evaluation. All algorithms are validated against R reference outputs and optimized for production scale via multithreading, SIMD vectorization, and optional GPU acceleration.

## Features

- **Unified API** — `fit!` / `predict` / `predict_scores` / `transform` for every model; no framework lock-in.
- **Production-grade performance** — zero-allocation inner loops, `@inbounds @simd` vectorization, BLAS-2 gram updates, per-thread pre-allocated buffers.
- **GPU acceleration** — optional CUDA.jl extension for EASE, iALS, WRMF (via package extensions).
- **R-validated correctness** — the full test suite includes a Tier-2 fixture layer that compares numerically against pre-computed R / rsparse outputs.
- **Sparse-native** — all algorithms operate directly on `SparseMatrixCSC`; no dense conversion needed.
- **Precompilation** — `PrecompileTools.jl` workloads reduce time-to-first-execution.
- **Tables.jl integration** — accept interaction data as `(user, item, value)` triplets from any Tables.jl-compatible source.
- **Cross-validation & search** — built-in temporal split, k-fold CV, grid search, and random search with warm-starting.
- **Callback system** — extensible training hooks for early stopping, checkpointing, learning rate scheduling, and custom logging.

---

## Algorithms

| Model | Type | Reference |
|-------|------|-----------|
| `WRMF` | Implicit / Explicit ALS (Cholesky, CG, NNLS) | Hu, Koren & Volinsky (2008) |
| `IALS` | Implicit ALS with Gramian caching | Rendle et al. (2021) |
| `EALS` | Element-wise ALS with popularity weighting | He et al. (2016) |
| `BPR` | Bayesian Personalized Ranking (pairwise SGD) | Rendle et al. (2009) |
| `LMF` | Logistic Matrix Factorization | Johnson (2014) |
| `GloVe` | Co-occurrence embedding (Hogwild AdaGrad) | Pennington, Socher & Manning (2014) |
| `EASE` | Embarrassingly Shallow Autoencoders | Steck (2019) |
| `SLIM` | Sparse Linear Methods (elastic net) | Ning & Karypis (2011) |
| `FTRL` | Follow The Regularized Leader (online GLM) | McMahan et al. (2013) |
| `FactorizationMachine` | 2nd-order FM (AdaGrad SGD) | Rendle (2010) |
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
    family         = BINOMIAL,
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
│   ├── utils.jl           # init_factors, sigmoid, _inplace_shuffle!, …
│   ├── sparse_utils.jl    # to_csr, dual_representation, row/col nnz
│   ├── callbacks.jl       # EarlyStopping, Checkpoint, LRScheduler, custom hooks
│   ├── crossval.jl        # temporal_split, kfold_cv, grid_search, random_search
│   ├── serialization.jl   # save_model / load_model (versioned binary format)
│   ├── tables.jl          # interactions_to_sparse / sparse_to_interactions
│   ├── progress.jl        # ConvergenceMonitor, logging utilities
│   ├── precompile.jl      # PrecompileTools workloads for TTFX
│   ├── algorithms/
│   │   ├── wrmf.jl        # Implicit/Explicit ALS (Cholesky · CG · NNLS)
│   │   ├── ials.jl        # iALS with Gramian caching
│   │   ├── eals.jl        # Element-wise ALS (popularity-weighted)
│   │   ├── bpr.jl         # Bayesian Personalized Ranking (pairwise SGD)
│   │   ├── lmf.jl         # Logistic MF with negative sampling
│   │   ├── glove.jl       # GloVe Hogwild AdaGrad
│   │   ├── ease.jl        # EASE (closed-form autoencoder)
│   │   ├── slim.jl        # SLIM (elastic-net item-item)
│   │   ├── ftrl.jl        # Follow The Regularized Leader (online)
│   │   ├── fm.jl          # Factorization Machines
│   │   └── soft_impute.jl # SoftImpute / SoftSVD
│   └── metrics/
│       └── ranking.jl     # AP@K, MAP@K, NDCG@K, Precision@K, Recall@K
├── ext/
│   └── GideonCUDAExt.jl   # GPU acceleration (EASE, iALS, WRMF, predict)
└── test/
    ├── runtests.jl
    └── r_correctness.jl   # Numerical validation against R / rsparse
```

### Type Hierarchy

```julia
AbstractSparseModel
├── AbstractMatrixFactorization   →  WRMF, IALS, EALS, LMF, BPR, GloVe, SoftImputeResult
└── AbstractSparseRegression      →  FTRL, FactorizationMachine
# Item-item models (no abstract parent):    EASE, SLIM
```

Every model implements the same generic interface:

| Function | Description |
|----------|-------------|
| `fit!(model, X)` | Train in-place on sparse matrix `X` |
| `partial_fit!(model, X, y)` | Online/incremental update (FTRL, FM, eALS) |
| `predict(model, X; k)` | Return top-k item indices per user |
| `predict_scores(model, X)` | Return full user×item score matrix |
| `transform(model, X)` | Return latent embeddings for new users |
| `coef(model)` | Return learned weight vector (FTRL) |

---

## GPU Acceleration

With [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) installed, Gideon loads a package extension providing:

```julia
using Gideon, CUDA

# GPU-accelerated EASE (fully on GPU)
fit_gpu!(model::EASE, X)

# GPU-accelerated iALS/WRMF (Gramian on GPU, solve on CPU)
fit_gpu!(model::IALS, X)
fit_gpu!(model::WRMF, X)

# Score computation on GPU for any matrix factorization model
predict_scores_gpu(model, X)
predict_gpu(model, X; k=10)
```

---

## Tables.jl Integration

Accept interaction data from any Tables.jl-compatible source (DataFrames, CSV rows, etc.):

```julia
using Gideon

# From a NamedTuple of vectors (column table)
data = (user=[1,1,2,3,3], item=[2,5,3,1,4], value=[1.0,2.0,1.0,3.0,1.0])
X = interactions_to_sparse(data)

# From a Vector of NamedTuples (row table)
rows = [(user=1, item=3, value=1.0), (user=2, item=1, value=2.0)]
X = interactions_to_sparse(rows)

# Convert back to triplets
triplets = sparse_to_interactions(X)
```

---

## Cross-Validation & Hyperparameter Search

```julia
using Gideon, SparseArrays

X = sprand(1000, 500, 0.02)

# Temporal train/test split
X_train, X_test = temporal_split(X; test_fraction=0.2)

# Grid search over hyperparameters
best_params, best_score, results = grid_search(
    p -> WRMF(rank=p.rank, λ=p.λ, α=40.0, max_iter=10, verbose=false),
    X,
    Dict(:rank => [16, 32, 64], :λ => [0.01, 0.1, 1.0]);
    k=10, metric=ndcg_at_k
)

# Random search with budget
best_params, best_score, _ = random_search(
    p -> WRMF(rank=p.rank, λ=p.λ, α=40.0, max_iter=10, verbose=false),
    X,
    Dict(:rank => rng -> rand(rng, [16, 32, 64, 128]),
         :λ   => rng -> 10.0^(rand(rng)*3 - 2));
    n_trials=20, k=10, metric=ndcg_at_k
)
```

---

## Performance Design

| Technique | Where used |
|-----------|-----------|
| Pre-allocated per-thread Gram / RHS / Cholesky buffers | WRMF ALS sweep |
| `BLAS.syr!` rank-1 Gram accumulation | WRMF Cholesky solver |
| `BLAS.syrk!` item Gram `YᵀY` | WRMF, iALS |
| Fast-path manual SIMD dot (`@inbounds @simd`) for sparse users with < 32 nnz | WRMF CG `_implicit_matvec!` |
| `@inbounds @simd` vectorized dot / gradient loops | WRMF, LMF, GloVe, BPR, eALS |
| CSR dual storage for O(nnz_u) per-user row access | All algorithms, metrics |
| `Threads.@threads :static` outer loops | WRMF, iALS, eALS, BPR user/item sweeps |
| Element-wise coordinate descent O(d) per update | eALS |
| Gramian caching (avoids per-user recomputation) | iALS, eALS |
| Zero-allocation Fisher-Yates shuffle | GloVe epoch shuffling |
| Numerical stability (epsilon floors in AdaGrad) | GloVe, FM |
| PrecompileTools workloads | All algorithms (reduces TTFX) |
| Optional GPU offloading via CUDA.jl extension | EASE, iALS, WRMF, predict |

---

## Testing

```bash
julia --project=. --threads=4 -e 'using Pkg; Pkg.test()'
```

The suite runs **420+ tests** covering:

- Unit correctness (dimensions, NaN / Inf guards, convergence monotonicity)
- R / rsparse numerical fixture comparisons (weights, predictions, loss values)
- Static analysis via [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl) and [JET.jl](https://github.com/aviatesk/JET.jl)
- All algorithms: WRMF, iALS, eALS, BPR, LMF, GloVe, EASE, SLIM, FTRL, FM, SoftImpute
- Infrastructure: serialization, cross-validation, callbacks, Tables.jl integration
- GPU stubs (full GPU tests when CUDA available)

---

## Dependencies

| Package | Role |
|---------|------|
| `SparseArrays` (stdlib) | Core sparse matrix type |
| `LinearAlgebra` (stdlib) | BLAS / LAPACK, SVD, Cholesky |
| `SparseMatricesCSR.jl` | CSR representation for row-oriented access |
| `PrecompileTools.jl` | Precompilation workloads for faster TTFX |

### Optional (Extensions)

| Package | Role |
|---------|------|
| `CUDA.jl` | GPU acceleration via package extension |

---

## License

MIT — see [LICENSE](LICENSE).
