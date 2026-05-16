# ──────────────────────────────────────────────────────────────────────────────
# Factorization Machines (2nd-order) — SGD with AdaGrad
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: Rendle (2010)
#   "Factorization Machines"
#
# Prediction:
#   ŷ(x) = w₀ + Σ_j wⱼ xⱼ + ½ Σ_{f=1}^{k} [ (Σ_j v_{j,f} xⱼ)² - Σ_j v²_{j,f} x²ⱼ ]
# ──────────────────────────────────────────────────────────────────────────────

using SparseArrays, LinearAlgebra, Random

"""
    FactorizationMachine{T} <: AbstractSparseRegression

Second-order Factorization Machine trained via SGD with AdaGrad.
"""
mutable struct FactorizationMachine{T<:AbstractFloat} <: AbstractSparseRegression
    rank::Int
    learning_rate_w::T
    learning_rate_v::T
    λ_w::T
    λ_v::T
    family::Symbol          # :binomial or :gaussian
    intercept::Bool
    n_features::Int
    w0::T
    w::Vector{T}
    V::Matrix{T}            # rank × n_features
    grad_w2::Vector{T}      # AdaGrad accumulators
    grad_v2::Matrix{T}
    is_initialized::Bool
end

function FactorizationMachine(;
    rank::Int = 4,
    learning_rate_w::Float64 = 0.2,
    learning_rate_v::Float64 = learning_rate_w,
    λ_w::Float64 = 0.0,
    λ_v::Float64 = 0.0,
    family::Symbol = :binomial,
    intercept::Bool = true,
)
    @assert family in (:binomial, :gaussian)
    @assert rank >= 1
    FactorizationMachine{Float64}(
        rank, learning_rate_w, learning_rate_v, λ_w, λ_v, family, intercept,
        0, 0.0, Float64[], Matrix{Float64}(undef,0,0),
        Float64[], Matrix{Float64}(undef,0,0), false,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# partial_fit! — single SGD epoch
# ──────────────────────────────────────────────────────────────────────────────

function partial_fit!(model::FactorizationMachine{T}, X::SparseMatrixCSC{Tv,Ti},
                      y::AbstractVector;
                      weights::AbstractVector{T} = ones(T, length(y)),
                      rng::AbstractRNG = Random.default_rng()) where {T,Tv,Ti}
    n_samples, n_features = size(X)
    @assert n_samples == length(y)

    if !model.is_initialized
        model.n_features = n_features
        model.w0 = zero(T)
        model.w  = randn(rng, T, n_features) .* T(0.001)
        model.V  = randn(rng, T, model.rank, n_features) .* T(0.001)
        model.grad_w2 = ones(T, n_features)
        model.grad_v2 = ones(T, model.rank, n_features)
        model.is_initialized = true
    end
    @assert n_features == model.n_features

    Xt = SparseMatrixCSC(X')
    rv = rowvals(Xt)
    nzv = nonzeros(Xt)
    k = model.rank

    for s in 1:n_samples
        col_range = nzrange(Xt, s)
        # ---- Forward pass ----
        pred = model.intercept ? model.w0 : zero(T)

        # Linear term
        for idx in col_range
            j = rv[idx]
            xval = T(nzv[idx])
            pred += model.w[j] * xval
        end

        # Interaction term: ½ Σ_f [ (Σ_j v_{jf} xⱼ)² - Σ_j v²_{jf} x²ⱼ ]
        interaction = zero(T)
        # Pre-compute sum_vx[f] = Σ_j v_{jf} xⱼ  for each factor
        sum_vx = zeros(T, k)
        sum_v2x2 = zeros(T, k)
        for idx in col_range
            j = rv[idx]
            xval = T(nzv[idx])
            @inbounds for f in 1:k
                vfj = model.V[f, j]
                sum_vx[f]   += vfj * xval
                sum_v2x2[f] += vfj^2 * xval^2
            end
        end
        @inbounds for f in 1:k
            interaction += sum_vx[f]^2 - sum_v2x2[f]
        end
        pred += interaction / 2

        # ---- Compute gradient multiplier ----
        if model.family == :binomial
            # For binomial: y should be in {0,1}, convert to {-1,+1}
            y_s = T(y[s]) > zero(T) ? one(T) : -one(T)
            grad_mult = -y_s * sigmoid(-y_s * pred) * weights[s]
        else  # gaussian
            grad_mult = (pred - T(y[s])) * weights[s]
        end

        # ---- Backward pass (AdaGrad updates) ----
        # Intercept
        if model.intercept
            model.w0 -= model.learning_rate_w * grad_mult
        end

        # Linear weights
        for idx in col_range
            j = rv[idx]
            xval = T(nzv[idx])
            gj = grad_mult * xval + model.λ_w * model.w[j]
            model.grad_w2[j] += gj^2
            model.w[j] -= model.learning_rate_w * gj / sqrt(model.grad_w2[j])
        end

        # Interaction factors
        for idx in col_range
            j = rv[idx]
            xval = T(nzv[idx])
            @inbounds for f in 1:k
                g_vfj = grad_mult * (sum_vx[f] * xval - model.V[f, j] * xval^2) + model.λ_v * model.V[f, j]
                model.grad_v2[f, j] += g_vfj^2
                model.V[f, j] -= model.learning_rate_v * g_vfj / sqrt(model.grad_v2[f, j])
            end
        end
    end
    model
end

function fit!(model::FactorizationMachine{T}, X::SparseMatrixCSC, y::AbstractVector;
              n_iter::Int = 1, kwargs...) where {T}
    for i in 1:n_iter
        @debug "FM epoch $i"
        partial_fit!(model, X, y; kwargs...)
    end
    model
end

# ──────────────────────────────────────────────────────────────────────────────
# predict
# ──────────────────────────────────────────────────────────────────────────────

function predict(model::FactorizationMachine{T}, X::SparseMatrixCSC) where {T}
    model.is_initialized || error("Model not fitted")
    n_samples = size(X, 1)
    @assert size(X, 2) == model.n_features

    Xt = SparseMatrixCSC(X')
    rv = rowvals(Xt)
    nzv = nonzeros(Xt)
    k = model.rank

    preds = Vector{T}(undef, n_samples)
    for s in 1:n_samples
        col_range = nzrange(Xt, s)
        pred = model.intercept ? model.w0 : zero(T)

        for idx in col_range
            j = rv[idx]
            pred += model.w[j] * T(nzv[idx])
        end

        interaction = zero(T)
        @inbounds for f in 1:k
            s_vx  = zero(T)
            s_v2x2 = zero(T)
            for idx in col_range
                j = rv[idx]
                xval = T(nzv[idx])
                vfj = model.V[f, j]
                s_vx   += vfj * xval
                s_v2x2 += vfj^2 * xval^2
            end
            interaction += s_vx^2 - s_v2x2
        end
        pred += interaction / 2

        if model.family == :binomial
            preds[s] = sigmoid(pred)
        else
            preds[s] = pred
        end
    end
    preds
end
