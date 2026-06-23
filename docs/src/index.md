# Gideon.jl

A high-performance Julia package for sparse matrix factorization, collaborative filtering, and recommendation systems. Julia port and enhancement of R's [rsparse](https://github.com/rexyai/rsparse).

## Features

- **WRMF** — Weighted Regularized Matrix Factorization (Cholesky, CG & NNLS solvers)
- **iALS** — Implicit ALS with Gramian caching (Rendle et al. 2021)
- **eALS** — Element-wise ALS with popularity-based weighting (He et al. 2016)
- **BPR** — Bayesian Personalized Ranking (pairwise learning)
- **LMF** — Logistic Matrix Factorization with negative sampling
- **GloVe** — Global Vectors for word/item embeddings
- **EASE** — Embarrassingly Shallow Autoencoders (closed-form)
- **SLIM** — Sparse Linear Methods (elastic-net item-item)
- **FTRL** — Follow The Regularized Leader (supports Binomial, Gaussian, Poisson families)
- **Factorization Machines** — Second-order feature interactions with SGD
- **SoftImpute / SoftSVD** — Nuclear-norm regularized matrix completion
- **Ranking Metrics** — MAP@k, NDCG@k, Precision@k, Recall@k
- **Cross-validation** — temporal split, k-fold, grid search, random search
- **GPU acceleration** — CUDA.jl extension for EASE, iALS, WRMF
- **Tables.jl integration** — accept interaction data as (user, item, value) triplets
- **Serialization** — versioned save/load for all models

## Quick Start

```julia
using Gideon, SparseArrays, Random

# Create a sparse user-item interaction matrix
X = sprand(MersenneTwister(42), 1000, 500, 0.02)

# Fit WRMF
model = WRMF(rank=10, λ=0.1, α=40.0, max_iter=15)
fit!(model, X)

# Get top-10 recommendations (seen items automatically masked)
recommendations = recommend(model, X; k=10)

# Full score matrix
scores = score(model, X)

# Scores for specific (user, item) pairs
pair_scores = score(model, [1, 2, 3], [10, 20, 30])

# Find similar items/users by cosine similarity
ids, sims = similar_items(model, 42; k=5)

# Evaluate
hold_out = sprand(MersenneTwister(99), 1000, 500, 0.01)
map_score = map_at_k(recommendations, hold_out; k=10)
println("MAP@10: $map_score")
```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/AbrJA/Gideon.jl")
```

## API Design

Gideon separates **recommender models** from **regression models** with domain-appropriate verbs:

| Model type | Top-k predictions | Raw scores | Regression |
|---|---|---|---|
| Recommenders (WRMF, IALS, EASE, ...) | `recommend(model, X; k)` | `score(model, X)` | — |
| Regression (FTRL, FM) | — | — | `predict(model, X)` |

All models share `fit!(model, X)` for training. Matrix factorization models additionally support:

- `transform(model, X)` — embed new users into the latent space
- `similar_items(model, id; k)` / `similar_users(model, id; k)` — cosine-based neighbors

The type hierarchy uses Julia's dispatch to provide default implementations:

```julia
AbstractSparseModel
├── AbstractRecommender
│   ├── AbstractMatrixFactorization  # WRMF, IALS, EALS, LMF, BPR, GloVe
│   └── AbstractItemSimilarity       # EASE, SLIM
└── AbstractSparseRegression         # FTRL, FactorizationMachine
```

New models inheriting from `AbstractMatrixFactorization` automatically get `recommend`,
`score`, `similar_items`, and `similar_users` with no boilerplate required.
