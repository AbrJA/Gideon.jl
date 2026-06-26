# Algorithms

## WMF — Weighted Regularized Matrix Factorization

```@docs
WMF
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 500, 300, 0.03)
model = WMF(rank=10, λ=0.1, α=40.0, max_iter=20, solver=CHOLESKY)
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
update!(model, X_new; n_iter=3)
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

## LogisticMF — Logistic Matrix Factorization

```@docs
LogisticMF
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 800, 300, 0.03)
model = LogisticMF(rank=15, α=1.0, λ=0.1, learning_rate=0.01, max_iter=20, n_negative=5)
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
model = GloVe(rank=50, x_max=100.0, learning_rate=0.05, max_iter=25)
fit!(model, X; rng=MersenneTwister(42))
E = embeddings(model)  # rank × n_words
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
coef
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 1000, 50, 0.1)
y = rand([0.0, 1.0], 1000)
model = FTRL(learning_rate=0.1, λ=0.01, family=BINOMIAL)
update!(model, X, y)
p = predict(model, X)
```

## Factorization Machines

```@docs
FM
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sparse([0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0])
y = [0.0, 1.0, 1.0, 0.0]  # XOR
model = FM(rank=4, family=BINOMIAL, max_iter=100)
fit!(model, X, y; rng=MersenneTwister(42))
predict(model, X)
```

## SoftImpute / SoftSVD — Low-rank Matrix Completion

```@docs
SoftImpute
SoftSVD
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 200, 150, 0.3)

# SoftImpute: full imputation correction (default, better for missing data)
model = SoftImpute(rank=10, λ=0.5, max_iter=100)
fit!(model, X; rng=MersenneTwister(1))

# SoftSVD: power-iteration style (faster per iteration)
model_svd = SoftSVD(rank=10, λ=0.5, max_iter=100)
fit!(model_svd, X; rng=MersenneTwister(1))

# Low-rank: model.U * Diagonal(model.d) * model.V'
```
