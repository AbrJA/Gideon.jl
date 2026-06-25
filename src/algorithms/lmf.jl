# ──────────────────────────────────────────────────────────────────────────────
# Logistic Matrix Factorization (LogisticMF)
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
    LogisticMF{T} <: AbstractMatrixFactorization

Logistic Matrix Factorization for implicit feedback via Adagrad with negative sampling.

# Constructor
```julia
LogisticMF(; rank=10, λ=0.6, α=1.0, learning_rate=1.0, max_iter=30,
    n_negative=30, convergence_tol=-1.0, verbose=true)
```

# Example
```julia
using SparseArrays, Gideon

X = sprand(1000, 500, 0.01)
model = LogisticMF(rank=32, max_iter=20, learning_rate=0.01)
fit!(model, X)
top_items = recommend(model, X; k=10)
```
"""
mutable struct LogisticMF{T<:AbstractFloat} <: AbstractMatrixFactorization
    const rank::Int
    const λ::T
    const α::T
    learning_rate::T
    const max_iter::Int
    const n_negative::Int
    const convergence_tol::T
    const verbose::Bool
    user_factors::Matrix{T}
    item_factors::Matrix{T}
    is_fitted::Bool
end

function LogisticMF(;
    rank::Int = 10,
    λ::Float64 = 0.6,
    α::Float64 = 1.0,
    learning_rate::Float64 = 1.0,
    max_iter::Int = 30,
    n_negative::Int = 30,
    convergence_tol::Float64 = -1.0,
    verbose::Bool = true,
    dtype::Type{<:AbstractFloat} = Float64,
)
    rank >= 1 || throw(ArgumentError("rank must be ≥ 1, got $rank"))
    λ >= 0.0 || throw(ArgumentError("λ must be non-negative, got $λ"))
    learning_rate > 0.0 || throw(ArgumentError("learning_rate must be positive, got $learning_rate"))
    n_negative >= 1 || throw(ArgumentError("n_negative must be ≥ 1, got $n_negative"))
    Td = dtype
    LogisticMF{Td}(rank, Td(λ), Td(α), Td(learning_rate), max_iter, n_negative, Td(convergence_tol),
            verbose, Matrix{Td}(undef,0,0), Matrix{Td}(undef,0,0), false)
end

"""
    fit!(model::LogisticMF, X; rng) -> model

Fit the LogisticMF model on user-item interaction matrix `X` (n_users × n_items).
Uses per-user/per-item batched Adagrad updates matching the implicit library's approach:
each epoch alternates a user-update phase and an item-update phase, with one batched
gradient accumulation and single Adagrad step per entity.
"""
function fit!(model::LogisticMF{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG = Random.default_rng(),
              callbacks::Vector{<:AbstractCallback} = AbstractCallback[]) where {T,Tv,Ti}
    n_users, n_items = size(X)
    k = model.rank

    # Standard normal initialization (matching implicit)
    model.user_factors = randn(rng, T, k, n_users)
    model.item_factors = randn(rng, T, k, n_items)

    U = model.user_factors  # k × n_users
    V = model.item_factors  # k × n_items

    # Build CSR for user→item access and CSC (= item→user) access
    X_csr = to_csr(X)
    n_interactions = nnz(X)

    # Flat interaction arrays for negative sampling (popularity-biased, like implicit)
    all_items = Vector{Int32}(undef, n_interactions)
    pos = 1
    for u in 1:n_users
        for idx in nzrange(X_csr, u)
            all_items[pos] = Int32(X_csr.colval[idx])
            pos += 1
        end
    end

    # Item→user CSR (transpose structure)
    Xt = sparse(X')  # n_items × n_users
    Xt_csr = to_csr(Xt)

    # Flat arrays for item-side negative sampling
    all_users = Vector{Int32}(undef, n_interactions)
    pos = 1
    for j in 1:n_items
        for idx in nzrange(Xt_csr, j)
            all_users[pos] = Int32(Xt_csr.colval[idx])
            pos += 1
        end
    end

    lr = model.learning_rate
    λ  = model.λ
    n_neg = model.n_negative
    ada_eps = T(1e-6)

    # Adagrad accumulators
    grad2_U = zeros(T, k, n_users)::Matrix{T}
    grad2_V = zeros(T, k, n_items)::Matrix{T}

    monitor = ConvergenceMonitor{T}(tol=T(model.convergence_tol), min_iter=2)

    # Per-thread RNGs
    nt = Threads.maxthreadid()
    thread_rngs = [Random.Xoshiro(rand(rng, UInt64)) for _ in 1:nt]

    for epoch in 1:model.max_iter
        epoch_start = time_ns()

        # ── Phase 1: Update user factors (items fixed) ──
        _lmf_update_users!(U, V, X_csr, all_items, grad2_U,
                           lr, λ, n_neg, ada_eps, k, n_users, n_interactions, thread_rngs)

        # ── Phase 2: Update item factors (users fixed) ──
        _lmf_update_items!(V, U, Xt_csr, all_users, grad2_V,
                           lr, λ, n_neg, ada_eps, k, n_items, n_interactions, thread_rngs)

        # ── Compute epoch loss (sampled estimate) ──
        loss = _lmf_loss_estimate(U, V, X_csr, n_users, k)

        iter_seconds = (time_ns() - epoch_start) / 1e9
        total_seconds = elapsed_seconds(monitor)
        if model.verbose
            log_iteration("LogisticMF", epoch, model.max_iter, Float64(loss),
                         iter_seconds, total_seconds)
        end

        if record!(monitor, loss)
            model.verbose && @info "[LogisticMF] converged at iteration $epoch"
            break
        end

        if !isempty(callbacks)
            info = CallbackInfo(epoch, Float64(loss), total_seconds, model)
            run_callbacks(callbacks, info) && break
        end
    end
    model.is_fitted = true
    model
end

"""
Update all user factors with one batched Adagrad step per user.
Matches implicit's lmf_update: accumulate gradient from positives + negatives + reg,
then single Adagrad update.
"""
function _lmf_update_users!(U::Matrix{T}, V::Matrix{T},
                            X_csr::SparseMatricesCSR.SparseMatrixCSR,
                            all_items::Vector{Int32},
                            grad2_U::Matrix{T},
                            lr::T, λ::T, n_neg::Int, ada_eps::T,
                            k::Int, n_users::Int, n_interactions::Int,
                            thread_rngs::Vector) where {T}
    n_items = size(V, 2)

    Threads.@threads :static for u in 1:n_users
        tid = Threads.threadid()
        local_rng = thread_rngs[tid]
        rng_u = nzrange(X_csr, u)
        user_seen = length(rng_u)
        user_seen == 0 && continue

        # Accumulate batched gradient matching implicit's lmf_update:
        # deriv = Σ_i c_i * v_i - Σ_i σ(s_ui)*c_i * v_i - Σ_neg σ(s_uj)*v_j - λ*u

        # Allocate per-user deriv buffer
        deriv = zeros(T, k)

        # Phase A: + Σ c_i * v_i[f]
        @inbounds for idx in rng_u
            j = Int(X_csr.colval[idx])
            c_uj = T(X_csr.nzval[idx])
            for f in 1:k
                deriv[f] += c_uj * V[f, j]
            end
        end

        # Phase B: - Σ σ(s_ui) * c_i * v_i[f]
        @inbounds for idx in rng_u
            j = Int(X_csr.colval[idx])
            c_uj = T(X_csr.nzval[idx])
            s = zero(T)
            for g in 1:k
                s += U[g, u] * V[g, j]
            end
            σ_s = one(T) / (one(T) + exp(-s))
            z = σ_s * c_uj
            for f in 1:k
                deriv[f] -= z * V[f, j]
            end
        end

        # Phase C: - Σ_neg σ(s_uj) * v_j[f]  (negatives sampled from global interactions)
        n_neg_samples = min(n_items, user_seen * n_neg)
        @inbounds for _ in 1:n_neg_samples
            neg_idx = rand(local_rng, 1:n_interactions)
            j = Int(all_items[neg_idx])
            s = zero(T)
            for g in 1:k
                s += U[g, u] * V[g, j]
            end
            σ_s = one(T) / (one(T) + exp(-s))
            for f in 1:k
                deriv[f] -= σ_s * V[f, j]
            end
        end

        # Phase D: regularization (once per user)
        @inbounds for f in 1:k
            deriv[f] -= λ * U[f, u]
        end

        # Adagrad update (one step per user)
        @inbounds for f in 1:k
            grad2_U[f, u] += deriv[f] * deriv[f]
            U[f, u] += (lr / sqrt(ada_eps + grad2_U[f, u])) * deriv[f]
        end
    end
end

"""
Update all item factors with one batched Adagrad step per item.
"""
function _lmf_update_items!(V::Matrix{T}, U::Matrix{T},
                            Xt_csr::SparseMatricesCSR.SparseMatrixCSR,
                            all_users::Vector{Int32},
                            grad2_V::Matrix{T},
                            lr::T, λ::T, n_neg::Int, ada_eps::T,
                            k::Int, n_items::Int, n_interactions::Int,
                            thread_rngs::Vector) where {T}
    n_users = size(U, 2)

    Threads.@threads :static for j in 1:n_items
        tid = Threads.threadid()
        local_rng = thread_rngs[tid]
        rng_j = nzrange(Xt_csr, j)
        item_seen = length(rng_j)
        item_seen == 0 && continue

        deriv = zeros(T, k)

        # Phase A: + Σ c_ui * u_i[f]
        @inbounds for idx in rng_j
            u = Int(Xt_csr.colval[idx])
            c_uj = T(Xt_csr.nzval[idx])
            for f in 1:k
                deriv[f] += c_uj * U[f, u]
            end
        end

        # Phase B: - Σ σ(s_ui) * c_ui * u_i[f]
        @inbounds for idx in rng_j
            u = Int(Xt_csr.colval[idx])
            c_uj = T(Xt_csr.nzval[idx])
            s = zero(T)
            for g in 1:k
                s += U[g, u] * V[g, j]
            end
            σ_s = one(T) / (one(T) + exp(-s))
            z = σ_s * c_uj
            for f in 1:k
                deriv[f] -= z * U[f, u]
            end
        end

        # Phase C: negatives (sampled from global interactions)
        n_neg_samples = min(n_users, item_seen * n_neg)
        @inbounds for _ in 1:n_neg_samples
            neg_idx = rand(local_rng, 1:n_interactions)
            u = Int(all_users[neg_idx])
            s = zero(T)
            for g in 1:k
                s += U[g, u] * V[g, j]
            end
            σ_s = one(T) / (one(T) + exp(-s))
            for f in 1:k
                deriv[f] -= σ_s * U[f, u]
            end
        end

        # Phase D: regularization
        @inbounds for f in 1:k
            deriv[f] -= λ * V[f, j]
        end

        # Adagrad update
        @inbounds for f in 1:k
            grad2_V[f, j] += deriv[f] * deriv[f]
            V[f, j] += (lr / sqrt(ada_eps + grad2_V[f, j])) * deriv[f]
        end
    end
end

function _lmf_loss_estimate(U::Matrix{T}, V::Matrix{T},
                            X_csr::SparseMatricesCSR.SparseMatrixCSR, n_users::Int, k::Int) where {T}
    loss = zero(T)
    for u in 1:n_users
        for idx in nzrange(X_csr, u)
            j = Int(X_csr.colval[idx])
            s = zero(T)
            @inbounds @simd for f in 1:k
                s += U[f, u] * V[f, j]
            end
            loss -= log(one(T) / (one(T) + exp(-s)) + T(1e-10))
        end
    end
    loss / max(one(T), T(nnz(X_csr)))
end


