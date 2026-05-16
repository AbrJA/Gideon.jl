# ──────────────────────────────────────────────────────────────────────────────
# FTRL — Follow The Regularized Leader (Proximal SGD)
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: McMahan et al. (2013)
#   "Ad Click Prediction: a View from the Trenches"
#
# Supports Elastic-Net (L1 + L2) regularization with per-coordinate
# adaptive learning rates.
# ──────────────────────────────────────────────────────────────────────────────

using SparseArrays, LinearAlgebra, Random

"""
    FTRL{T} <: AbstractSparseRegression

Follow The Regularized Leader proximal SGD for logistic regression on sparse data.

# Fields
- `learning_rate::T`
- `learning_rate_decay::T`  — controls per-coordinate learning-rate decay (β in paper)
- `λ::T`                   — overall regularization strength
- `l1_ratio::T`            — L1 vs L2 mix: 1.0 = Lasso, 0.0 = Ridge
- `dropout::T`             — feature dropout rate [0,1)
- `n_features::Int`        — set after first `partial_fit!`
- `z::Vector{T}`           — FTRL state vector
- `n::Vector{T}`           — per-coordinate sum of squared gradients
"""
mutable struct FTRL{T<:AbstractFloat} <: AbstractSparseRegression
    learning_rate::T
    learning_rate_decay::T
    λ::T
    l1_ratio::T
    dropout::T
    n_features::Int
    z::Vector{T}
    n::Vector{T}
    is_initialized::Bool
end

function FTRL(;
    learning_rate::Float64 = 0.1,
    learning_rate_decay::Float64 = 0.5,
    λ::Float64 = 0.0,
    l1_ratio::Float64 = 1.0,
    dropout::Float64 = 0.0,
)
    @assert 0.0 <= dropout < 1.0
    @assert 0.0 <= l1_ratio <= 1.0
    @assert λ >= 0.0
    @assert learning_rate > 0.0
    @assert learning_rate_decay > 0.0
    FTRL{Float64}(learning_rate, learning_rate_decay, λ, l1_ratio, dropout,
                  0, Float64[], Float64[], false)
end

# ──────────────────────────────────────────────────────────────────────────────
# partial_fit! — single epoch (online / streaming)
# ──────────────────────────────────────────────────────────────────────────────

function partial_fit!(model::FTRL{T}, X::SparseMatrixCSC{Tv,Ti}, y::AbstractVector;
                      weights::AbstractVector{T} = ones(T, length(y)),
                      rng::AbstractRNG = Random.default_rng()) where {T,Tv,Ti}
    n_samples, n_features = size(X)
    @assert n_samples == length(y) "X rows ($(n_samples)) ≠ length(y) ($(length(y)))"
    @assert !any(isnan, nonzeros(X)) "NaN values in input matrix"

    if !model.is_initialized
        model.n_features = n_features
        model.z = zeros(T, n_features)
        model.n = zeros(T, n_features)
        model.is_initialized = true
    end
    @assert n_features == model.n_features "Feature dimension mismatch"

    Xt = SparseMatrixCSC(X')  # n_features × n_samples — iterate by sample (column)

    z = model.z
    n_acc = model.n
    lr = model.learning_rate
    β  = model.learning_rate_decay
    λ  = model.λ
    λ1 = λ * model.l1_ratio
    λ2 = λ * (one(T) - model.l1_ratio)
    do_dropout = model.dropout > zero(T)

    rv = rowvals(Xt)
    nzv = nonzeros(Xt)

    for s in 1:n_samples
        # Compute w from z,n and form prediction
        pred = zero(T)
        col_range = nzrange(Xt, s)

        for idx in col_range
            j = rv[idx]
            xval = T(nzv[idx])
            if do_dropout && rand(rng) < model.dropout
                continue
            end
            wj = _ftrl_weight(z[j], n_acc[j], lr, β, λ1, λ2)
            pred += wj * xval
        end
        pred = sigmoid(pred)

        # Gradient: (pred - y) * x_j
        err = (pred - T(y[s])) * weights[s]

        for idx in col_range
            j = rv[idx]
            xval = T(nzv[idx])
            gj = err * xval
            σj = (sqrt(n_acc[j] + gj^2) - sqrt(n_acc[j])) / lr
            z[j] += gj - σj * _ftrl_weight(z[j], n_acc[j], lr, β, λ1, λ2)
            n_acc[j] += gj^2
        end
    end
    model
end

function fit!(model::FTRL{T}, X::SparseMatrixCSC, y::AbstractVector;
              n_iter::Int = 1, kwargs...) where {T}
    for i in 1:n_iter
        @debug "FTRL epoch $i"
        partial_fit!(model, X, y; kwargs...)
    end
    model
end

# ──────────────────────────────────────────────────────────────────────────────
# predict / coef
# ──────────────────────────────────────────────────────────────────────────────

function predict(model::FTRL{T}, X::SparseMatrixCSC) where {T}
    model.is_initialized || error("Model not fitted")
    n_samples = size(X, 1)
    @assert size(X, 2) == model.n_features

    w = coef(model)
    Xt = SparseMatrixCSC(X')
    rv = rowvals(Xt)
    nzv = nonzeros(Xt)

    preds = Vector{T}(undef, n_samples)
    @inbounds for s in 1:n_samples
        v = zero(T)
        for idx in nzrange(Xt, s)
            j = rv[idx]
            v += w[j] * T(nzv[idx])
        end
        preds[s] = sigmoid(v)
    end
    preds
end

function coef(model::FTRL{T}) where {T}
    model.is_initialized || error("Model not fitted")
    w = Vector{T}(undef, model.n_features)
    lr = model.learning_rate
    β  = model.learning_rate_decay
    λ1 = model.λ * model.l1_ratio
    λ2 = model.λ * (one(T) - model.l1_ratio)
    @inbounds for j in 1:model.n_features
        w[j] = _ftrl_weight(model.z[j], model.n[j], lr, β, λ1, λ2)
    end
    w
end

# ──────────────────────────────────────────────────────────────────────────────
# Internal: compute effective weight from FTRL state
# ──────────────────────────────────────────────────────────────────────────────

@inline function _ftrl_weight(zj::T, nj::T, lr::T, β::T, λ1::T, λ2::T) where {T}
    if abs(zj) <= λ1
        return zero(T)
    end
    sign_z = zj > zero(T) ? one(T) : -one(T)
    -((zj - sign_z * λ1) / ((β + sqrt(nj)) / lr + λ2))
end
