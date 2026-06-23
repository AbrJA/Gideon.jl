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
top_items = recommend(model, X; k=10)
```
"""
mutable struct LMF{T<:AbstractFloat} <: AbstractMatrixFactorization
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

function LMF(;
    rank::Int = 10,
    λ::Float64 = 0.1,
    α::Float64 = 1.0,
    learning_rate::Float64 = 0.01,
    max_iter::Int = 10,
    n_negative::Int = 4,
    convergence_tol::Float64 = -1.0,
    verbose::Bool = true,
    dtype::Type{<:AbstractFloat} = Float64,
)
    rank >= 1 || throw(ArgumentError("rank must be ≥ 1, got $rank"))
    λ >= 0.0 || throw(ArgumentError("λ must be non-negative, got $λ"))
    learning_rate > 0.0 || throw(ArgumentError("learning_rate must be positive, got $learning_rate"))
    n_negative >= 1 || throw(ArgumentError("n_negative must be ≥ 1, got $n_negative"))
    Td = dtype
    LMF{Td}(rank, Td(λ), Td(α), Td(learning_rate), max_iter, n_negative, Td(convergence_tol),
            verbose, Matrix{Td}(undef,0,0), Matrix{Td}(undef,0,0), false)
end

"""
    fit!(model::LMF, X; rng) -> model

Fit the LMF model on user-item interaction matrix `X` (n_users × n_items).
Uses Hogwild!-style lock-free parallel SGD with per-interaction negative sampling.
"""
function fit!(model::LMF{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG = Random.default_rng(),
              callbacks::Vector{<:AbstractCallback} = AbstractCallback[]) where {T,Tv,Ti}
    n_users, n_items = size(X)
    k = model.rank

    # Xavier initialization — scale ~ 1/sqrt(rank)
    scale = T(1.0 / sqrt(k))
    model.user_factors = randn(rng, T, k, n_users) .* scale
    model.item_factors = randn(rng, T, k, n_items) .* scale

    U = model.user_factors
    V = model.item_factors

    # Build flat sampling structure (like BPR) — sample (user, pos_item) uniformly
    X_csr = to_csr(X)
    n_interactions = nnz(X)

    userids = Vector{Int32}(undef, n_interactions)
    itemids = Vector{Int32}(undef, n_interactions)
    conf_vals = Vector{T}(undef, n_interactions)  # confidence values
    pos = 1
    for u in 1:n_users
        for idx in nzrange(X_csr, u)
            userids[pos] = Int32(u)
            itemids[pos] = Int32(X_csr.colval[idx])
            conf_vals[pos] = T(X_csr.nzval[idx])
            pos += 1
        end
    end

    # Build sorted item lists per user for negative sampling
    user_item_sorted = Vector{Vector{Int32}}(undef, n_users)
    for u in 1:n_users
        items = Int32[]
        for idx in nzrange(X_csr, u)
            push!(items, Int32(X_csr.colval[idx]))
        end
        sort!(items)
        user_item_sorted[u] = items
    end

    lr = model.learning_rate
    λ  = model.λ
    α  = model.α
    n_neg = model.n_negative
    samples_per_epoch = n_interactions

    monitor = ConvergenceMonitor{T}(tol=T(model.convergence_tol), min_iter=2)

    # Per-thread RNGs
    nt = Threads.nthreads()
    thread_rngs = [Random.Xoshiro(rand(rng, UInt64)) for _ in 1:nt]

    for epoch in 1:model.max_iter
        epoch_start = time_ns()

        # Hogwild! parallel SGD
        epoch_losses = zeros(T, nt)

        Threads.@threads :static for chunk in 1:nt
            local_rng = thread_rngs[chunk]
            local_loss = zero(T)
            chunk_size = cld(samples_per_epoch, nt)
            chunk_start = (chunk - 1) * chunk_size + 1
            chunk_end = min(chunk * chunk_size, samples_per_epoch)

            @fastmath @inbounds for _ in chunk_start:chunk_end
                # Sample a random positive interaction
                liked_index = rand(local_rng, 1:n_interactions)
                u = Int(userids[liked_index])
                i = Int(itemids[liked_index])
                r = conf_vals[liked_index]
                c = one(T) + α * r

                # ── Positive update: grad = c * (1 - σ(s)) ──
                # Johnson (2014): maximize c * log σ(s) for positive pairs
                s = zero(T)
                @simd for f in 1:k
                    s += U[f, u] * V[f, i]
                end
                σ_s = one(T) / (one(T) + exp(-s))
                grad_pos = c * (one(T) - σ_s)

                for f in 1:k
                    u_f = U[f, u]
                    i_f = V[f, i]
                    U[f, u] = u_f + lr * (grad_pos * i_f - λ * u_f)
                    V[f, i] = i_f + lr * (grad_pos * u_f - λ * i_f)
                end

                local_loss -= c * log1pexp(-s)

                # ── Negative sampling: sample items user hasn't seen ──
                # Johnson (2014): maximize log(1 - σ(s)) for negative pairs
                local sorted_items = user_item_sorted[u]
                for _ in 1:n_neg
                    j = rand(local_rng, Int32(1):Int32(n_items))
                    while _insorted(sorted_items, j)
                        j = rand(local_rng, Int32(1):Int32(n_items))
                    end
                    j_int = Int(j)

                    # Negative gradient: -σ(s_neg)
                    s_neg = zero(T)
                    @simd for f in 1:k
                        s_neg += U[f, u] * V[f, j_int]
                    end
                    σ_neg = one(T) / (one(T) + exp(-s_neg))

                    for f in 1:k
                        u_f = U[f, u]
                        j_f = V[f, j_int]
                        U[f, u] = u_f + lr * (-σ_neg * j_f - λ * u_f)
                        V[f, j_int] = j_f + lr * (-σ_neg * u_f - λ * j_f)
                    end

                    local_loss -= log1pexp(s_neg)
                end
            end

            epoch_losses[chunk] = local_loss
        end

        total_loss = sum(epoch_losses) / samples_per_epoch

        iter_seconds = (time_ns() - epoch_start) / 1e9
        total_seconds = elapsed_seconds(monitor)
        if model.verbose
            log_iteration("LMF", epoch, model.max_iter, Float64(total_loss),
                         iter_seconds, total_seconds)
        end

        if record!(monitor, total_loss)
            model.verbose && @info "[LMF] converged at iteration $epoch"
            break
        end

        if !isempty(callbacks)
            info = CallbackInfo(epoch, Float64(total_loss), total_seconds, model)
            run_callbacks(callbacks, info) && break
        end
    end
    model.is_fitted = true
    model
end


