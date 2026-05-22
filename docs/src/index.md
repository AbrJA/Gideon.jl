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

# Get top-10 recommendations
predictions = predict(model, X; k=10)

# Full score matrix
scores = predict_scores(model, X)

# Evaluate
hold_out = sprand(MersenneTwister(99), 1000, 500, 0.01)
map_score = map_at_k(predictions, hold_out; k=10)
println("MAP@10: $map_score")
```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/AbrJA/Gideon.jl")
```
