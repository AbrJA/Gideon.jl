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

"""
    FactorizationMachine{T} <: AbstractSparseRegression

Second-order Factorization Machine trained via SGD with AdaGrad.

Supports both classification (`BINOMIAL`) and regression (`GAUSSIAN`) via
the `family` parameter. Uses per-coordinate adaptive learning rates (AdaGrad).

# Constructor
```julia
FactorizationMachine(; rank=4, learning_rate_w=0.2, learning_rate_v=learning_rate_w,
                       λ_w=0.0, λ_v=0.0, family=BINOMIAL, intercept=true,
                       n_iter=10, convergence_tol=-1.0, verbose=true)
```

# Example
```julia
using SparseArrays, Gideon

X = sprand(10000, 1000, 0.01)
y = rand([0.0, 1.0], 10000)
model = FactorizationMachine(rank=8, family=BINOMIAL)
fit!(model, X, y; n_iter=20)
preds = predict(model, X)
```
"""
mutable struct FactorizationMachine{T<:AbstractFloat} <: AbstractSparseRegression
    rank::Int
    learning_rate_w::T
    learning_rate_v::T
    λ_w::T
    λ_v::T
    family::Family
    intercept::Bool
    n_iter::Int
    convergence_tol::T
    verbose::Bool
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
    family::Family = BINOMIAL,
    intercept::Bool = true,
    n_iter::Int = 10,
    convergence_tol::Float64 = -1.0,
    verbose::Bool = true,
)
    @assert rank >= 1 "rank must be ≥ 1"
    @assert family in (BINOMIAL, GAUSSIAN) "FM supports BINOMIAL or GAUSSIAN"
    FactorizationMachine{Float64}(
        rank, learning_rate_w, learning_rate_v, λ_w, λ_v, family, intercept,
        n_iter, convergence_tol, verbose,
        0, 0.0, Float64[], Matrix{Float64}(undef,0,0),
        Float64[], Matrix{Float64}(undef,0,0), false,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# partial_fit! — single SGD epoch
# ──────────────────────────────────────────────────────────────────────────────

"""
    partial_fit!(model::FactorizationMachine, X, y; weights, rng) -> model

Run a single SGD epoch over the data.
"""
function partial_fit!(model::FactorizationMachine{T}, X::SparseMatrixCSC{Tv,Ti},
                      y::AbstractVector;
                      weights::AbstractVector{T} = ones(T, length(y)),
                      rng::AbstractRNG = Random.default_rng()) where {T,Tv,Ti}
    iter_start = time_ns()
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
    @assert n_features == model.n_features "Feature dimension mismatch"

    Xt = SparseMatrixCSC(X')
    rv = rowvals(Xt)
    nzv = nonzeros(Xt)
    k = model.rank

    # Pre-allocate per-sample buffers
    sum_vx   = Vector{T}(undef, k)
    sum_v2x2 = Vector{T}(undef, k)

    for s in 1:n_samples
        col_range = nzrange(Xt, s)
        # ---- Forward pass ----
        pred = model.intercept ? model.w0 : zero(T)

        for idx in col_range
            j = rv[idx]
            xval = T(nzv[idx])
            pred += model.w[j] * xval
        end

        # Interaction term: ½ Σ_f [ (Σ_j v_{jf} xⱼ)² - Σ_j v²_{jf} x²ⱼ ]
        fill!(sum_vx, zero(T))
        fill!(sum_v2x2, zero(T))
        for idx in col_range
            j = rv[idx]
            xval = T(nzv[idx])
            @inbounds for f in 1:k
                vfj = model.V[f, j]
                sum_vx[f]   += vfj * xval
                sum_v2x2[f] += vfj^2 * xval^2
            end
        end
        interaction = zero(T)
        @inbounds for f in 1:k
            interaction += sum_vx[f]^2 - sum_v2x2[f]
        end
        pred += interaction / 2

        # ---- Compute gradient multiplier ----
        if model.family == BINOMIAL
            y_s = T(y[s]) > zero(T) ? one(T) : -one(T)
            grad_mult = -y_s * sigmoid(-y_s * pred) * weights[s]
        else  # GAUSSIAN
            grad_mult = (pred - T(y[s])) * weights[s]
        end

        # ---- Backward pass (AdaGrad updates) ----
        if model.intercept
            model.w0 -= model.learning_rate_w * grad_mult
        end

        for idx in col_range
            j = rv[idx]
            xval = T(nzv[idx])
            gj = grad_mult * xval + model.λ_w * model.w[j]
            model.grad_w2[j] += gj^2
            model.w[j] -= model.learning_rate_w * gj / sqrt(model.grad_w2[j])
        end

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

    if model.verbose
        pass_seconds = (time_ns() - iter_start) / 1e9
        @info @sprintf("[FM] partial_fit: %d samples, %d features | time=%s",
                       n_samples, n_features, elapsed_str(pass_seconds))
    end
    model
end

"""
    fit!(model::FactorizationMachine, X, y; n_iter, kwargs...) -> model

Train the FM for `n_iter` epochs (defaults to `model.n_iter`).
"""
function fit!(model::FactorizationMachine{T}, X::SparseMatrixCSC, y::AbstractVector;
              n_iter::Int = model.n_iter, kwargs...) where {T}
    train_start = time_ns()
    prev_loss = T(Inf)

    for i in 1:n_iter
        epoch_start = time_ns()
        partial_fit!(model, X, y; kwargs...)
        epoch_seconds = (time_ns() - epoch_start) / 1e9
        total_seconds = (time_ns() - train_start) / 1e9

        # Compute training loss for convergence check
        if model.convergence_tol > zero(T)
            preds = predict(model, X)
            loss = if model.family == BINOMIAL
                -sum(y .* log.(preds .+ T(1e-10)) .+ (one(T) .- y) .* log.(one(T) .- preds .+ T(1e-10))) / length(y)
            else
                sum((preds .- y).^2) / length(y)
            end
            if model.verbose
                log_iteration("FM", i, n_iter, Float64(loss), epoch_seconds, total_seconds)
            end
            if i > 1 && abs(prev_loss - loss) / (abs(prev_loss) + T(1e-12)) < model.convergence_tol
                model.verbose && @info "[FM] converged at epoch $i"
                break
            end
            prev_loss = loss
        elseif model.verbose
            @info @sprintf("[FM] epoch %d/%d | epoch=%s | total=%s",
                           i, n_iter, elapsed_str(epoch_seconds), elapsed_str(total_seconds))
        end
    end
    model
end

# ──────────────────────────────────────────────────────────────────────────────
# predict
# ──────────────────────────────────────────────────────────────────────────────

"""
    predict(model::FactorizationMachine, X) -> Vector

Generate predictions. Output depends on family:
- `BINOMIAL` → probabilities in [0,1]
- `GAUSSIAN` → real-valued predictions
"""
function predict(model::FactorizationMachine{T}, X::SparseMatrixCSC) where {T}
    model.is_initialized || error("Model not fitted")
    n_samples = size(X, 1)
    @assert size(X, 2) == model.n_features "Feature dimension mismatch"

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

        if model.family == BINOMIAL
            preds[s] = sigmoid(pred)
        else
            preds[s] = pred
        end
    end
    preds
end
