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
preds = predict(model, X; k=5)
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

## GloVe — Global Vectors

```@docs
GloVe
```

### Example

```julia
using Gideon, SparseArrays, Random
# Co-occurrence matrix
X = sprand(MersenneTwister(1), 100, 100, 0.1)
model = GloVe(rank=50, x_max=100.0, learning_rate=0.05)
fit!(model, X; n_iter=25, rng=MersenneTwister(42))
```

## LMF — Logistic Matrix Factorization

```@docs
LMF
```

## SoftImpute / SoftSVD

```@docs
soft_impute
soft_svd
```
