# ──────────────────────────────────────────────────────────────────────────────
# BPR — Bayesian Personalized Ranking
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: Rendle, Freudenthaler, Gantner, Schmidt-Thieme (2009)
#   "BPR: Bayesian Personalized Ranking from Implicit Feedback" (UAI 2009)
#   arXiv:1205.2618
#
# Optimizes the AUC-related BPR-Opt criterion:
#   Σ_{(u,i,j) ∈ D_S} ln σ(x̂_{uij}) - λ‖Θ‖²
#
# where x̂_{uij} = x̂_{ui} - x̂_{uj} (score difference between positive and
# negative item), and D_S is the set of triplets (user, positive_item, negative_item).
#
# Learning: Stochastic Gradient Descent with bootstrap sampling of triplets.
# ──────────────────────────────────────────────────────────────────────────────

"""
    BPR{T} <: AbstractMatrixFactorization

Bayesian Personalized Ranking via Matrix Factorization.

Learns user and item embeddings optimized for ranking (AUC) rather than
pointwise prediction. Uses SGD with negative sampling of (user, pos, neg) triplets.

# Negative Sampling Strategies
- `:uniform` — standard uniform random sampling (Rendle et al. 2009)
- `:popular` — popularity-biased sampling (items sampled proportional to sqrt of frequency)
- `:dns` — Dynamic Negative Sampling: sample `dns_candidates` negatives, pick the
  one with highest score as the "hardest" negative (Zhang et al. 2013)

# Constructor
```julia
BPR(; rank=64, λ_user=0.01, λ_pos=0.01, λ_neg=0.01,
      learning_rate=0.05, max_iter=100, n_samples=0,
      negative_sampling=:uniform, dns_candidates=5,
      convergence_tol=-1.0, verbose=true)
```

# Fields
- `rank::Int` — embedding dimension
- `λ_user::T` — L2 regularization for user factors
- `λ_pos::T` — L2 regularization for positive item factors
- `λ_neg::T` — L2 regularization for negative item factors
- `learning_rate::T` — SGD step size
- `max_iter::Int` — number of epochs
- `n_samples::Int` — samples per epoch (0 = nnz(X))
- `negative_sampling::Symbol` — `:uniform`, `:popular`, or `:dns`
- `dns_candidates::Int` — number of candidates for DNS strategy
- `convergence_tol::T` — AUC-based early stopping tolerance (-1 disables)
"""
mutable struct BPR{T<:AbstractFloat} <: AbstractMatrixFactorization
    rank::Int
    λ_user::T
    λ_pos::T
    λ_neg::T
    learning_rate::T
    max_iter::Int
    n_samples::Int
    negative_sampling::Symbol
    dns_candidates::Int
    convergence_tol::T
    verbose::Bool
    # Factors
    user_factors::Matrix{T}
    item_factors::Matrix{T}
    loss_history::Vector{T}
    is_fitted::Bool
end

function BPR(;
    rank::Int = 64,
    λ_user::Float64 = 0.01,
    λ_pos::Float64 = 0.01,
    λ_neg::Float64 = 0.01,
    learning_rate::Float64 = 0.05,
    max_iter::Int = 100,
    n_samples::Int = 0,
    negative_sampling::Symbol = :uniform,
    dns_candidates::Int = 5,
    convergence_tol::Float64 = -1.0,
    verbose::Bool = true,
    dtype::Type{<:AbstractFloat} = Float32,
)
    @assert rank >= 1
    @assert learning_rate > 0.0
    @assert negative_sampling in (:uniform, :popular, :dns)
    @assert dns_candidates >= 1
    T = dtype
    BPR{T}(rank, T(λ_user), T(λ_pos), T(λ_neg), T(learning_rate), max_iter, n_samples,
            negative_sampling, dns_candidates, T(convergence_tol), verbose,
            Matrix{T}(undef,0,0), Matrix{T}(undef,0,0), T[], false)
end

# ──────────────────────────────────────────────────────────────────────────────
# fit!
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit!(model::BPR, X; rng) -> model

Fit BPR-MF on implicit feedback matrix `X` (users × items).
Non-zero entries are treated as positive interactions.

Uses Hogwild!-style lock-free parallel SGD (Niu et al. 2011) for massive speedup
on multi-core systems. Each thread processes independent samples with concurrent
writes to shared factor matrices — safe for sparse problems where collision
probability is low.
"""
function fit!(model::BPR{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG = Random.default_rng()) where {T,Tv,Ti}
    n_users, n_items = size(X)
    k = model.rank

    # Initialize factors with small random values
    model.user_factors = randn(rng, T, k, n_users) .* T(0.01)
    model.item_factors = randn(rng, T, k, n_items) .* T(0.01)
    model.loss_history = T[]

    U = model.user_factors
    V = model.item_factors

    # ── Build flat sampling structure (like implicit) ──
    # userids[s] = user who made interaction s
    # itemids[s] = item of interaction s
    # This lets us sample a (user, positive_item) pair with a single random index
    X_csr = to_csr(X)
    n_interactions = nnz(X)

    userids = Vector{Int32}(undef, n_interactions)
    itemids = Vector{Int32}(undef, n_interactions)
    pos = 1
    for u in 1:n_users
        for idx in nzrange(X_csr, u)
            userids[pos] = Int32(u)
            itemids[pos] = Int32(X_csr.colval[idx])
            pos += 1
        end
    end

    # Build sorted item lists per user for binary-search negative verification
    user_item_sorted = Vector{Vector{Int32}}(undef, n_users)
    for u in 1:n_users
        items = Int32[]
        for idx in nzrange(X_csr, u)
            push!(items, Int32(X_csr.colval[idx]))
        end
        sort!(items)
        user_item_sorted[u] = items
    end

    # Build popularity-based sampling distribution (sqrt-frequency smoothing)
    item_pop = zeros(T, n_items)
    for j in axes(X, 2)
        item_pop[j] = T(length(nzrange(X, j)))
    end
    pop_weights = sqrt.(item_pop)
    pop_cumsum = cumsum(pop_weights)
    pop_total = pop_cumsum[end]

    samples_per_epoch = model.n_samples > 0 ? model.n_samples : n_interactions
    monitor = ConvergenceMonitor{T}(tol=T(model.convergence_tol), min_iter=3)

    lr = model.learning_rate
    λ_u = model.λ_user
    λ_p = model.λ_pos
    λ_n = model.λ_neg
    neg_strategy = model.negative_sampling
    dns_k = model.dns_candidates

    # ── Per-thread RNGs for thread safety ──
    nt = Threads.nthreads()
    thread_rngs = [Random.Xoshiro(rand(rng, UInt64)) for _ in 1:nt]

    for epoch in 1:model.max_iter
        epoch_start = time_ns()

        # ── Hogwild! parallel SGD — lock-free concurrent updates ──
        epoch_losses = zeros(T, nt)
        epoch_correct = zeros(Int, nt)

        Threads.@threads :static for chunk in 1:nt
            local_rng = thread_rngs[chunk]
            local_loss = zero(T)
            local_correct = 0
            chunk_size = cld(samples_per_epoch, nt)
            chunk_start = (chunk - 1) * chunk_size + 1
            chunk_end = min(chunk * chunk_size, samples_per_epoch)

            @fastmath @inbounds for _ in chunk_start:chunk_end
                # Sample a random interaction → gives (user, positive_item)
                liked_index = rand(local_rng, 1:n_interactions)
                u = Int(userids[liked_index])
                i = Int(itemids[liked_index])

                # Sample negative item (inline for uniform; call function for others)
                local sorted_items = user_item_sorted[u]
                local j_int::Int
                if neg_strategy == :uniform
                    j = rand(local_rng, Int32(1):Int32(n_items))
                    while _insorted(sorted_items, j)
                        j = rand(local_rng, Int32(1):Int32(n_items))
                    end
                    j_int = Int(j)
                else
                    j_int = _bpr_sample_negative_fast(local_rng, n_items, sorted_items,
                                                     neg_strategy, dns_k,
                                                     pop_cumsum, pop_total,
                                                     U, V, u, k)
                end

                # Compute x̂_uij = x̂_ui - x̂_uj
                x_uij = zero(T)
                @simd for f in 1:k
                    x_uij += U[f, u] * (V[f, i] - V[f, j_int])
                end

                # σ(-x_uij) = 1/(1 + exp(x_uij))
                sig = one(T) / (one(T) + exp(x_uij))

                if sig < T(0.5)
                    local_correct += 1
                end

                local_loss += -log(one(T) - sig + T(1e-10))

                # SGD updates (lock-free Hogwild! — races are acceptable)
                for f in 1:k
                    u_f = U[f, u]
                    i_f = V[f, i]
                    j_f = V[f, j_int]
                    diff = i_f - j_f

                    U[f, u] = u_f + lr * (sig * diff - λ_u * u_f)
                    V[f, i] = i_f + lr * (sig * u_f - λ_p * i_f)
                    V[f, j_int] = j_f + lr * (-sig * u_f - λ_n * j_f)
                end
            end

            epoch_losses[chunk] = local_loss
            epoch_correct[chunk] = local_correct
        end

        total_loss = sum(epoch_losses)
        avg_loss = total_loss / samples_per_epoch
        push!(model.loss_history, avg_loss)

        iter_seconds = (time_ns() - epoch_start) / 1e9
        total_seconds = elapsed_seconds(monitor)

        if model.verbose
            total_correct = sum(epoch_correct)
            auc_pct = 100.0 * total_correct / samples_per_epoch
            log_iteration("BPR", epoch, model.max_iter, Float64(avg_loss),
                         iter_seconds, total_seconds;
                         extra="auc≈$(round(auc_pct; digits=1))%")
        end

        if record!(monitor, avg_loss)
            model.verbose && @info "[BPR] converged at epoch $epoch"
            break
        end
    end

    model.is_fitted = true
    model
end

"""
Sample a negative item using binary search on sorted item lists (O(log n) verification).
Much faster than Set-based lookup for cache-friendly access patterns.
"""
function _bpr_sample_negative_fast(rng::AbstractRNG, n_items::Int,
                                   sorted_items::Vector{Int32},
                                   strategy::Symbol, dns_k::Int,
                                   pop_cumsum::Vector{T}, pop_total::T,
                                   U::Matrix{T}, V::Matrix{T},
                                   u::Int, k::Int) where {T}
    if strategy == :uniform
        # Rejection sampling with binary search verification
        j = rand(rng, Int32(1):Int32(n_items))
        while _insorted(sorted_items, j)
            j = rand(rng, Int32(1):Int32(n_items))
        end
        return Int(j)
    elseif strategy == :popular
        j = Int32(_sample_from_cumsum(rng, pop_cumsum, pop_total, n_items))
        while _insorted(sorted_items, j)
            j = Int32(_sample_from_cumsum(rng, pop_cumsum, pop_total, n_items))
        end
        return Int(j)
    else  # :dns
        best_j = Int32(0)
        best_score = T(-Inf)
        candidates_found = 0
        max_tries = dns_k * 5
        tries = 0
        while candidates_found < dns_k && tries < max_tries
            tries += 1
            j = rand(rng, Int32(1):Int32(n_items))
            _insorted(sorted_items, j) && continue
            candidates_found += 1
            score = zero(T)
            @inbounds @simd for f in 1:k
                score += U[f, u] * V[f, Int(j)]
            end
            if score > best_score
                best_score = score
                best_j = j
            end
        end
        if best_j == Int32(0)
            best_j = rand(rng, Int32(1):Int32(n_items))
            while _insorted(sorted_items, best_j)
                best_j = rand(rng, Int32(1):Int32(n_items))
            end
        end
        return Int(best_j)
    end
end

"""
Binary search in a sorted Int32 vector. O(log n) and cache-friendly.
"""
@inline function _insorted(sorted::Vector{Int32}, val::Int32)
    lo, hi = 1, length(sorted)
    @inbounds while lo <= hi
        mid = (lo + hi) >>> 1
        if sorted[mid] < val
            lo = mid + 1
        elseif sorted[mid] > val
            hi = mid - 1
        else
            return true
        end
    end
    return false
end

"""
Legacy: Sample a negative item according to the specified strategy.
Kept for backward compatibility with tests.
"""
function _bpr_sample_negative(rng::AbstractRNG, n_items::Int, items_u_set::Set{Int},
                              strategy::Symbol, dns_k::Int,
                              pop_cumsum::Vector{T}, pop_total::T,
                              U::Matrix{T}, V::Matrix{T},
                              u::Int, k::Int) where {T}
    if strategy == :uniform
        # Standard uniform rejection sampling
        j = rand(rng, 1:n_items)
        while j in items_u_set
            j = rand(rng, 1:n_items)
        end
        return j
    elseif strategy == :popular
        # Popularity-biased: sample proportional to sqrt(item frequency)
        j = _sample_from_cumsum(rng, pop_cumsum, pop_total, n_items)
        while j in items_u_set
            j = _sample_from_cumsum(rng, pop_cumsum, pop_total, n_items)
        end
        return j
    else  # :dns — Dynamic Negative Sampling
        best_j = 0
        best_score = T(-Inf)
        candidates_found = 0
        max_tries = dns_k * 5  # avoid infinite loop
        tries = 0
        while candidates_found < dns_k && tries < max_tries
            tries += 1
            j = rand(rng, 1:n_items)
            j in items_u_set && continue
            candidates_found += 1
            # Compute score for this candidate
            score = zero(T)
            @inbounds @simd for f in 1:k
                score += U[f, u] * V[f, j]
            end
            if score > best_score
                best_score = score
                best_j = j
            end
        end
        # Fallback if no candidates found
        if best_j == 0
            best_j = rand(rng, 1:n_items)
            while best_j in items_u_set
                best_j = rand(rng, 1:n_items)
            end
        end
        return best_j
    end
end

"""
Sample an index from a cumulative weight distribution via binary search.
"""
function _sample_from_cumsum(rng::AbstractRNG, cumsum::Vector{T},
                             total::T, n::Int) where {T}
    r = rand(rng) * total
    # Binary search for the position
    lo, hi = 1, n
    while lo < hi
        mid = (lo + hi) >> 1
        if @inbounds cumsum[mid] < r
            lo = mid + 1
        else
            hi = mid
        end
    end
    lo
end

# ──────────────────────────────────────────────────────────────────────────────
# predict
# ──────────────────────────────────────────────────────────────────────────────

"""
    predict(model::BPR, X; k=10) -> Matrix{Int}

Return top-k item indices per user, excluding already-interacted items.
Uses batched GEMM for efficient score computation.
"""
function predict(model::BPR{T}, X::SparseMatrixCSC; k::Int=10) where {T}
    model.is_fitted || error("Model not fitted")
    n_users = size(X, 1)
    n_items = size(model.item_factors, 2)
    k_out = min(k, n_items)

    predictions = Matrix{Int}(undef, n_users, k_out)
    X_csr = to_csr(X)

    # Batched GEMM approach — compute scores for batches of users at once
    max_batch_mem = 2 * 1024^3  # 2 GB target
    batch_size = max(1, min(n_users, Int(floor(max_batch_mem / (n_items * sizeof(T))))))

    for batch_start in 1:batch_size:n_users
        batch_end = min(batch_start + batch_size - 1, n_users)
        batch_users = batch_start:batch_end
        n_batch = length(batch_users)

        # Batched matrix multiply: (n_batch × rank) × (rank × n_items) = (n_batch × n_items)
        scores = model.user_factors[:, batch_users]' * model.item_factors

        # Mask seen items and get top-k per user (threaded)
        Threads.@threads for local_u in 1:n_batch
            global_u = batch_users[local_u]
            @inbounds for idx in nzrange(X_csr, global_u)
                j = Int(X_csr.colval[idx])
                scores[local_u, j] = T(-Inf)
            end
            row = @view scores[local_u, :]
            perm = partialsortperm(row, 1:k_out; rev=true)
            @inbounds predictions[global_u, :] .= perm
        end
    end
    predictions
end

"""
    predict_scores(model::BPR, user_indices, item_indices) -> Vector

Return raw scores for specific (user, item) pairs.
"""
function predict_scores(model::BPR{T}, user_indices::AbstractVector{<:Integer},
                        item_indices::AbstractVector{<:Integer}) where {T}
    model.is_fitted || error("Model not fitted")
    @assert length(user_indices) == length(item_indices)
    n = length(user_indices)
    scores = Vector{T}(undef, n)
    k = size(model.user_factors, 1)
    @inbounds for idx in 1:n
        u = user_indices[idx]
        i = item_indices[idx]
        s = zero(T)
        @simd for f in 1:k
            s += model.user_factors[f, u] * model.item_factors[f, i]
        end
        scores[idx] = s
    end
    scores
end
