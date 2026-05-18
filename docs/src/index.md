# Gideon.jl

A high-performance Julia package for sparse matrix factorization and recommendation systems. Julia port of R's [rsparse](https://github.com/rexyai/rsparse).

## Features

- **WRMF** — Weighted Regularized Matrix Factorization (Cholesky & Conjugate Gradient solvers)
- **FTRL** — Follow The Regularized Leader (supports Binomial, Gaussian, Poisson families)
- **Factorization Machines** — Second-order feature interactions with SGD
- **GloVe** — Global Vectors for word embeddings
- **LMF** — Logistic Matrix Factorization with negative sampling
- **SoftImpute / SoftSVD** — Nuclear-norm regularized matrix completion
- **Ranking Metrics** — MAP@k, NDCG@k, Precision@k, Recall@k

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

# Evaluate
hold_out = sprand(MersenneTwister(99), 1000, 500, 0.01)
map_score = map_at_k(predictions, hold_out; k=10)
println("MAP@10: $map_score")
```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/ajaimes/Gideon.jl")
```
