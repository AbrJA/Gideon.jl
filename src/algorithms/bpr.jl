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
)
    @assert rank >= 1
    @assert learning_rate > 0.0
    @assert negative_sampling in (:uniform, :popular, :dns)
    @assert dns_candidates >= 1
    T = Float64
    BPR{T}(rank, λ_user, λ_pos, λ_neg, learning_rate, max_iter, n_samples,
            negative_sampling, dns_candidates, convergence_tol, verbose,
            Matrix{T}(undef,0,0), Matrix{T}(undef,0,0), T[], false)
end

# ──────────────────────────────────────────────────────────────────────────────
# fit!
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit!(model::BPR, X; rng) -> model

Fit BPR-MF on implicit feedback matrix `X` (users × items).
Non-zero entries are treated as positive interactions.
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

    # Build user -> positive items index for fast sampling
    user_items = Vector{Vector{Int}}(undef, n_users)
    for u in 1:n_users
        user_items[u] = Int[]
    end
    rv = rowvals(X)
    for j in axes(X, 2)
        for idx in nzrange(X, j)
            push!(user_items[rv[idx]], j)
        end
    end

    # Build user -> Set for O(1) membership testing (avoids slow `in` on Vector)
    user_item_sets = [Set{Int}(items) for items in user_items]

    # Filter users with at least 1 positive item
    active_users = findall(u -> !isempty(user_items[u]), 1:n_users)
    n_active = length(active_users)

    # Build popularity-based sampling distribution (sqrt-frequency smoothing)
    item_pop = zeros(T, n_items)
    for j in axes(X, 2)
        item_pop[j] = T(length(nzrange(X, j)))
    end
    pop_weights = sqrt.(item_pop)
    pop_cumsum = cumsum(pop_weights)
    pop_total = pop_cumsum[end]

    samples_per_epoch = model.n_samples > 0 ? model.n_samples : nnz(X)
    monitor = ConvergenceMonitor{T}(tol=T(model.convergence_tol), min_iter=3)

    lr = model.learning_rate
    λ_u = model.λ_user
    λ_p = model.λ_pos
    λ_n = model.λ_neg
    neg_strategy = model.negative_sampling
    dns_k = model.dns_candidates

    for epoch in 1:model.max_iter
        epoch_start = time_ns()
        epoch_loss = zero(T)

        for _ in 1:samples_per_epoch
            # Sample a random user with at least one positive item
            u = active_users[rand(rng, 1:n_active)]
            items_u = user_items[u]
            items_u_set = user_item_sets[u]

            # Sample positive item
            i = items_u[rand(rng, 1:length(items_u))]

            # Sample negative item based on strategy
            j = _bpr_sample_negative(rng, n_items, items_u_set,
                                     neg_strategy, dns_k,
                                     pop_cumsum, pop_total,
                                     U, V, u, k)

            # Compute x̂_uij = x̂_ui - x̂_uj
            x_uij = zero(T)
            @inbounds @simd for f in 1:k
                x_uij += U[f, u] * (V[f, i] - V[f, j])
            end

            # BPR loss: -ln σ(x̂_uij)
            sig = one(T) / (one(T) + exp(x_uij))  # σ(-x_uij)
            epoch_loss += -log(one(T) - sig + T(1e-10))

            # SGD updates
            @inbounds @simd for f in 1:k
                u_f = U[f, u]
                i_f = V[f, i]
                j_f = V[f, j]

                U[f, u] += lr * (sig * (i_f - j_f) - λ_u * u_f)
                V[f, i] += lr * (sig * u_f - λ_p * i_f)
                V[f, j] += lr * (sig * (-u_f) - λ_n * j_f)
            end
        end

        avg_loss = epoch_loss / samples_per_epoch
        push!(model.loss_history, avg_loss)

        iter_seconds = (time_ns() - epoch_start) / 1e9
        total_seconds = elapsed_seconds(monitor)

        if model.verbose
            log_iteration("BPR", epoch, model.max_iter, Float64(avg_loss),
                         iter_seconds, total_seconds)
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
Sample a negative item according to the specified strategy.
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
"""
function predict(model::BPR{T}, X::SparseMatrixCSC; k::Int=10) where {T}
    model.is_fitted || error("Model not fitted")
    n_users = size(X, 1)
    n_items = size(model.item_factors, 2)
    k_out = min(k, n_items)

    preds = Matrix{Int}(undef, n_users, k_out)
    X_csr = to_csr(X)

    Threads.@threads for u in 1:n_users
        scores = model.item_factors' * @view(model.user_factors[:, u])

        # Mask seen items using CSR row access
        @inbounds for idx in nzrange(X_csr, u)
            j = Int(X_csr.colval[idx])
            scores[j] = T(-Inf)
        end

        topk = partialsortperm(scores, 1:k_out; rev=true)
        preds[u, :] .= topk
    end
    preds
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
