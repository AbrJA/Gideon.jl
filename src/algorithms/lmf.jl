# ──────────────────────────────────────────────────────────────────────────────
# Logistic Matrix Factorization (LMF)
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: Johnson (2014)
#   "Logistic Matrix Factorization for Implicit Feedback Data"
#
# Loss:
#   L = Σ_{u,i} [r_{ui} · xᵤᵀ yᵢ - (1 + α·r_{ui}) · log(1 + exp(xᵤᵀ yᵢ))]
#       - λ/2 (||X||² + ||Y||²)
# ──────────────────────────────────────────────────────────────────────────────

"""
    LMF{T} <: AbstractMatrixFactorization

Logistic Matrix Factorization for implicit feedback via SGD with negative sampling.

# Constructor
```julia
LMF(; rank=10, λ=0.1, α=1.0, learning_rate=0.01, max_iter=10,
      n_negative=4, convergence_tol=-1.0, verbose=true)
```

# Example
```julia
using SparseArrays, Gideon

X = sprand(1000, 500, 0.01)
model = LMF(rank=32, max_iter=20, learning_rate=0.01)
fit!(model, X)
top_items = predict(model, X; k=10)
```
"""
mutable struct LMF{T<:AbstractFloat} <: AbstractMatrixFactorization
    rank::Int
    λ::T
    α::T
    learning_rate::T
    max_iter::Int
    n_negative::Int
    convergence_tol::T
    verbose::Bool
    user_factors::Matrix{T}
    item_factors::Matrix{T}
    is_fitted::Bool
end

function LMF(;
    rank::Int = 10,
    λ::Float64 = 0.1,
    α::Float64 = 1.0,
    learning_rate::Float64 = 0.01,
    max_iter::Int = 10,
    n_negative::Int = 4,
    convergence_tol::Float64 = -1.0,
    verbose::Bool = true,
)
    @assert rank >= 1 "rank must be ≥ 1"
    @assert λ >= 0.0 "λ must be non-negative"
    @assert learning_rate > 0.0 "learning_rate must be positive"
    @assert n_negative >= 1 "n_negative must be ≥ 1"
    LMF{Float64}(rank, λ, α, learning_rate, max_iter, n_negative, convergence_tol,
                 verbose, Matrix{Float64}(undef,0,0), Matrix{Float64}(undef,0,0), false)
end

"""
    fit!(model::LMF, X; rng) -> model

Fit the LMF model on user-item interaction matrix `X` (n_users × n_items).
"""
function fit!(model::LMF{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG = Random.default_rng()) where {T,Tv,Ti}
    n_users, n_items = size(X)
    k = model.rank

    model.user_factors = randn(rng, T, k, n_users) .* T(0.01)
    model.item_factors = randn(rng, T, k, n_items) .* T(0.01)

    rv = rowvals(X)
    nz = nonzeros(X)

    monitor = ConvergenceMonitor{T}(tol=T(model.convergence_tol), min_iter=2)

    for iter in 1:model.max_iter
        iter_start = time_ns()
        total_loss = zero(T)

        for j in axes(X, 2)
            for idx in nzrange(X, j)
                u   = rv[idx]
                r   = T(nz[idx])
                c   = one(T) + model.α * r

                s  = zero(T)
                @inbounds @simd for f in 1:k
                    s += model.user_factors[f, u] * model.item_factors[f, j]
                end

                σ_s = sigmoid(s)
                grad_mult = r - c * σ_s

                lr = model.learning_rate
                λ  = model.λ
                U = model.user_factors
                V = model.item_factors

                @inbounds @simd for f in 1:k
                    gu = grad_mult * V[f, j] - λ * U[f, u]
                    gi = grad_mult * U[f, u] - λ * V[f, j]
                    U[f, u] += lr * gu
                    V[f, j] += lr * gi
                end

                total_loss += r * s - c * log1pexp(s)
            end

            # Negative sampling
            for _ in 1:model.n_negative
                u_neg = rand(rng, 1:n_users)
                s  = zero(T)
                @inbounds @simd for f in 1:k
                    s += model.user_factors[f, u_neg] * model.item_factors[f, j]
                end
                σ_s = sigmoid(s)

                lr = model.learning_rate
                λ  = model.λ
                U = model.user_factors
                V = model.item_factors

                @inbounds @simd for f in 1:k
                    gu = -σ_s * V[f, j] - λ * U[f, u_neg]
                    gi = -σ_s * U[f, u_neg] - λ * V[f, j]
                    U[f, u_neg] += lr * gu
                    V[f, j]     += lr * gi
                end
                total_loss -= log1pexp(s)
            end
        end

        total_loss -= model.λ / 2 * (sum(abs2, model.user_factors) + sum(abs2, model.item_factors))

        iter_seconds = (time_ns() - iter_start) / 1e9
        total_seconds = elapsed_seconds(monitor)
        if model.verbose
            log_iteration("LMF", iter, model.max_iter, Float64(total_loss),
                         iter_seconds, total_seconds)
        end

        if record!(monitor, total_loss)
            model.verbose && @info "[LMF] converged at iteration $iter"
            break
        end
    end
    model.is_fitted = true
    model
end

"""
    predict(model::LMF, X; k=10) -> Matrix{Int}

Return top-k item indices for each user. Returns `n_users × k` matrix.
"""
function predict(model::LMF{T}, X::SparseMatrixCSC; k::Int = 10) where {T}
    model.is_fitted || error("Model not fitted")
    scores = model.user_factors' * model.item_factors
    n_users = size(scores, 1)
    n_items = size(scores, 2)
    k_actual = min(k, n_items)
    predictions = Matrix{Int}(undef, n_users, k_actual)

    X_csr = to_csr(X)
    Threads.@threads for u in 1:n_users
        s = @view scores[u, :]
        # Mask seen items
        @inbounds for idx in nzrange(X_csr, u)
            j = Int(X_csr.colval[idx])
            scores[u, j] = T(-Inf)
        end
        @inbounds predictions[u, :] .= partialsortperm(Vector(s), 1:k_actual; rev=true)
    end
    predictions
end

"""
    predict_scores(model::LMF, X) -> Matrix

Return the full score matrix (n_users × n_items) = U' * V.
"""
function predict_scores(model::LMF{T}, X::SparseMatrixCSC) where {T}
    model.is_fitted || error("Model not fitted")
    model.user_factors' * model.item_factors
end
