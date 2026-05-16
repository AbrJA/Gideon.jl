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

using SparseArrays, LinearAlgebra, Random, LoopVectorization

"""
    LMF{T} <: AbstractMatrixFactorization

Logistic Matrix Factorization for implicit feedback.
"""
mutable struct LMF{T<:AbstractFloat} <: AbstractMatrixFactorization
    rank::Int
    λ::T
    α::T
    learning_rate::T
    max_iter::Int
    n_negative::Int  # negative samples per positive
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
)
    LMF{Float64}(rank, λ, α, learning_rate, max_iter, n_negative,
                 Matrix{Float64}(undef,0,0), Matrix{Float64}(undef,0,0), false)
end

function fit!(model::LMF{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG = Random.default_rng()) where {T,Tv,Ti}
    n_users, n_items = size(X)
    k = model.rank

    model.user_factors = randn(rng, T, k, n_users) .* T(0.01)
    model.item_factors = randn(rng, T, k, n_items) .* T(0.01)

    rv = rowvals(X)
    nz = nonzeros(X)

    for iter in 1:model.max_iter
        total_loss = zero(T)

        for j in axes(X, 2)  # iterate items (columns)
            for idx in nzrange(X, j)
                u   = rv[idx]
                r   = T(nz[idx])
                c   = one(T) + model.α * r

                s  = zero(T)
                @inbounds @simd for f in 1:k
                    s += model.user_factors[f, u] * model.item_factors[f, j]
                end

                # Gradient of logistic loss
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

                # Loss: r·s - c·log(1+exp(s))
                total_loss += r * s - c * log1pexp(s)
            end

            # Negative sampling for item j
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

        # Regularization
        total_loss -= model.λ / 2 * (sum(abs2, model.user_factors) + sum(abs2, model.item_factors))
        @debug "LMF iter=$iter  loss=$total_loss"
    end
    model.is_fitted = true
    model
end

function predict(model::LMF{T}, X::SparseMatrixCSC; k::Int = 10) where {T}
    model.is_fitted || error("Model not fitted")
    scores = model.user_factors' * model.item_factors  # n_users × n_items
    n_users = size(scores, 1)
    n_items = size(scores, 2)
    k_actual = min(k, n_items)
    predictions = Matrix{Int}(undef, n_users, k_actual)
    for u in 1:n_users
        row = @view scores[u, :]
        perm = sortperm(row; rev=true)
        @inbounds predictions[u, :] .= perm[1:k_actual]
    end
    predictions
end
