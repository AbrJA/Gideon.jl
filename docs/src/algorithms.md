# Algorithms

## WeightedMatrixFactorization — Weighted Regularized Matrix Factorization

```@docs
WeightedMatrixFactorization
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 500, 300, 0.03)
model = WeightedMatrixFactorization(rank=10, λ=0.1, α=40.0, max_iter=20, solver=CHOLESKY)
fit!(model, X; rng=MersenneTwister(42))
preds = recommend(model, X; k=5)
scores = score(model, X)
```

## ImplicitALS — Implicit ALS with Gramian Caching

```@docs
ImplicitALS
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 1000, 500, 0.02)
model = ImplicitALS(rank=32, λ=0.01, α=1.0, max_iter=15)
fit!(model, X; rng=MersenneTwister(42))
preds = recommend(model, X; k=10)
```

## ElementwiseALS — Element-wise ALS

```@docs
ElementwiseALS
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 1000, 500, 0.02)
model = ElementwiseALS(rank=64, λ=0.01, w0=10.0, max_iter=20)
fit!(model, X; rng=MersenneTwister(42))
preds = recommend(model, X; k=10)

# Incremental update with new data
X_new = sprand(MersenneTwister(2), 1000, 500, 0.01)
update!(model, X_new; n_iter=3)
```

## BayesianPersonalizedRanking — Bayesian Personalized Ranking

```@docs
BayesianPersonalizedRanking
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 500, 300, 0.03)
model = BayesianPersonalizedRanking(rank=32, λ=0.01, learning_rate=0.05, max_iter=50)
fit!(model, X; rng=MersenneTwister(42))
preds = recommend(model, X; k=10)
```

## LogisticMatrixFactorization — Logistic Matrix Factorization

```@docs
LogisticMatrixFactorization
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 800, 300, 0.03)
model = LogisticMatrixFactorization(rank=15, α=1.0, λ=0.1, learning_rate=0.01, max_iter=20, n_negative=5)
fit!(model, X; rng=MersenneTwister(42))
preds = recommend(model, X; k=10)
scores = score(model, X)
```

## GlobalVectors — Global Vectors

```@docs
GlobalVectors
```

### Example

```julia
using Gideon, SparseArrays, Random
# Co-occurrence matrix (symmetric)
X = sprand(MersenneTwister(1), 100, 100, 0.1)
X = X + X'
model = GlobalVectors(rank=50, x_max=100.0, learning_rate=0.05, max_iter=25)
fit!(model, X; rng=MersenneTwister(42))
E = embeddings(model)  # rank × n_words
```

## ShallowAutoencoder — Embarrassingly Shallow Autoencoders

```@docs
ShallowAutoencoder
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 500, 200, 0.05)
model = ShallowAutoencoder(λ=500.0)
fit!(model, X)
preds = recommend(model, X; k=10)
```

## SparseLinearModel — Sparse Linear Methods

```@docs
SparseLinearModel
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 500, 100, 0.05)
model = SparseLinearModel(λ=0.1, α=0.5, max_iter=100)
fit!(model, X)
preds = recommend(model, X; k=10)
```

## OnlineRegressor — Follow The Regularized Leader

```@docs
OnlineRegressor
coef
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 1000, 50, 0.1)
y = rand([0.0, 1.0], 1000)
model = OnlineRegressor(learning_rate=0.1, λ=0.01, family=BINOMIAL)
update!(model, X, y)
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
model = FactorizationMachine(rank=4, family=BINOMIAL, max_iter=100)
fit!(model, X, y; rng=MersenneTwister(42))
predict(model, X)
```

## SoftImpute — Low-rank Matrix Completion

```@docs
SoftImpute
```

### Example

```julia
using Gideon, SparseArrays, Random
X = sprand(MersenneTwister(1), 200, 150, 0.3)
model = SoftImpute(rank=10, λ=0.5, max_iter=100)
fit!(model, X; rng=MersenneTwister(1))
# Low-rank: model.U * Diagonal(model.d) * model.V'
```
