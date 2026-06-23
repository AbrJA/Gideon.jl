# Algorithms

## WRMF — Weighted Regularized Matrix Factorization

```@docs
WRMF
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 500, 300, 0.03)
model = WRMF(rank=10, λ=0.1, α=40.0, max_iter=20, solver=CHOLESKY)
fit!(model, X; rng=MersenneTwister(42))
preds = recommend(model, X; k=5)
scores = score(model, X)
```

## IALS — Implicit ALS with Gramian Caching

```@docs
IALS
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 1000, 500, 0.02)
model = IALS(rank=32, λ=0.01, α=1.0, max_iter=15)
fit!(model, X; rng=MersenneTwister(42))
preds = recommend(model, X; k=10)
```

## EALS — Element-wise ALS

```@docs
EALS
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 1000, 500, 0.02)
model = EALS(rank=64, λ=0.01, w0=10.0, max_iter=20)
fit!(model, X; rng=MersenneTwister(42))
preds = recommend(model, X; k=10)

# Incremental update with new data
X_new = sprand(MersenneTwister(2), 1000, 500, 0.01)
partial_fit!(model, X_new; n_iter=3)
```

## BPR — Bayesian Personalized Ranking

```@docs
BPR
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 500, 300, 0.03)
model = BPR(rank=32, λ=0.01, learning_rate=0.05, max_iter=50)
fit!(model, X; rng=MersenneTwister(42))
preds = recommend(model, X; k=10)
```

## LMF — Logistic Matrix Factorization

```@docs
LMF
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 800, 300, 0.03)
model = LMF(rank=15, α=1.0, λ=0.1, learning_rate=0.01, max_iter=20, n_negative=5)
fit!(model, X; rng=MersenneTwister(42))
preds = recommend(model, X; k=10)
scores = score(model, X)
```

## GloVe — Global Vectors

```@docs
GloVe
```

### Example

```julia
using Gideon, SparseArrays, Random
# Co-occurrence matrix (symmetric)
X = sprand(MersenneTwister(1), 100, 100, 0.1)
X = X + X'
model = GloVe(rank=50, x_max=100.0, learning_rate=0.05)
fit!(model, X; n_iter=25, rng=MersenneTwister(42))
E = get_embeddings(model)  # rank × n_words
```

## EASE — Embarrassingly Shallow Autoencoders

```@docs
EASE
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 500, 200, 0.05)
model = EASE(λ=500.0)
fit!(model, X)
preds = recommend(model, X; k=10)
```

## SLIM — Sparse Linear Methods

```@docs
SLIM
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 500, 100, 0.05)
model = SLIM(λ=0.1, α=0.5, max_iter=100)
fit!(model, X)
preds = recommend(model, X; k=10)
```

## FTRL — Follow The Regularized Leader

```@docs
FTRL
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 1000, 50, 0.1)
y = rand([0.0, 1.0], 1000)
model = FTRL(learning_rate=0.1, λ=0.01, family=BINOMIAL)
partial_fit!(model, X, y)
p = predict(model, X)
```

## Factorization Machines

```@docs
FactorizationMachine
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sparse([0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0])
y = [0.0, 1.0, 1.0, 0.0]  # XOR
model = FactorizationMachine(rank=4, family=BINOMIAL)
fit!(model, X, y; n_iter=100, rng=MersenneTwister(42))
predict(model, X)
```

## SoftImpute / SoftSVD

```@docs
soft_impute
soft_svd
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 200, 150, 0.3)
result = soft_impute(X; rank=10, λ=0.5, n_iter=100)
# Low-rank: result.U * Diagonal(result.d) * result.V'
```
