# ──────────────────────────────────────────────────────────────────────────────
# iALS — Implicit Alternating Least Squares with Subspace Optimization
# ──────────────────────────────────────────────────────────────────────────────
#
# References:
#   - Hu, Koren, Volinsky (2008): "Collaborative Filtering for Implicit Feedback Datasets"
#   - Rendle, Krichene, Zhang, Koren (2021): "iALS++: Speeding up Matrix Factorization
#     with Subspace Optimization" (arXiv:2110.14044)
#
# Key insight from iALS++: Instead of solving the full d×d linear system per user/item,
# use a subspace solver that operates on a smaller k-dimensional block, leveraging
# the structure of the Gramian matrix that is shared across all users/items.
#
# Loss (implicit feedback):
#   L = Σ_{u,i} c_{ui} (p_{ui} - xᵤᵀyᵢ)² + λ(‖X‖² + ‖Y‖²)
#
# where c_{ui} = 1 + α·r_{ui}, p_{ui} = 1 if r_{ui}>0 else 0
# ──────────────────────────────────────────────────────────────────────────────

"""
    IALS{T} <: AbstractMatrixFactorization

Implicit Alternating Least Squares with efficient Gramian caching.

Implements the algorithm from Hu et al. (2008) with the efficient Gramian trick:
instead of forming the full confidence-weighted system per user, we precompute
the item Gramian `YᵀY` and add only the diagonal corrections for non-zero entries.

This avoids the O(n_items × d²) cost per user and replaces it with
O(nnz_per_user × d² + d³) per user, which is dramatically faster for sparse data.

# Solver Options
- `:cholesky` — exact solve via Cholesky decomposition, O(d³) per user. Best for d ≤ 128.
- `:cg` — approximate solve via Conjugate Gradient, O(d² × cg_steps) per user.
  Best for large d (≥ 128). Uses warm-start from previous iteration's solution.

# Constructor
```julia
IALS(; rank=64, λ=0.01, α=40.0, max_iter=15, convergence_tol=0.005,
       solver=:cholesky, cg_steps=3, verbose=true)
```

# Fields
- `rank::Int` — embedding dimension
- `λ::T` — L2 regularization
- `α::T` — confidence scaling: c_{ui} = 1 + α·r_{ui}
- `max_iter::Int` — maximum ALS iterations
- `convergence_tol::T` — relative change in loss for early stopping (-1 disables)
- `solver::Symbol` — `:cholesky` or `:cg`
- `cg_steps::Int` — CG inner iterations (only for `:cg` solver)
- `user_factors::Matrix{T}` — (rank × n_users) after fitting
- `item_factors::Matrix{T}` — (rank × n_items) after fitting
"""
mutable struct IALS{T<:AbstractFloat} <: AbstractMatrixFactorization
    rank::Int
    λ::T
    α::T
    max_iter::Int
    convergence_tol::T
    solver::Symbol
    cg_steps::Int
    verbose::Bool
    # Factors (rank × n)
    user_factors::Matrix{T}
    item_factors::Matrix{T}
    is_fitted::Bool
end

function IALS(;
    rank::Int = 64,
    λ::Float64 = 0.01,
    α::Float64 = 40.0,
    max_iter::Int = 15,
    convergence_tol::Float64 = 0.005,
    solver::Symbol = :cholesky,
    cg_steps::Int = 3,
    verbose::Bool = true,
)
    @assert rank >= 1
    @assert λ >= 0.0
    @assert α >= 0.0
    @assert solver in (:cholesky, :cg)
    @assert cg_steps >= 1
    T = Float64
    IALS{T}(rank, λ, α, max_iter, convergence_tol, solver, cg_steps, verbose,
            Matrix{T}(undef,0,0), Matrix{T}(undef,0,0), false)
end

# ──────────────────────────────────────────────────────────────────────────────
# fit!
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit!(model::IALS, X; rng, U_init, V_init) -> model

Fit iALS on sparse interaction matrix `X` (users × items).

Uses the efficient Gramian-caching approach: precomputes `YᵀY` (or `XᵀX`) once
per iteration, then adds per-user diagonal corrections from non-zero entries.
"""
function fit!(model::IALS{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG = Random.default_rng(),
              U_init::Union{Nothing,Matrix{T}} = nothing,
              V_init::Union{Nothing,Matrix{T}} = nothing) where {T,Tv,Ti}
    n_users, n_items = size(X)
    k = model.rank
    α = model.α
    λ = model.λ

    # Initialize factors
    if U_init !== nothing
        model.user_factors = copy(U_init)
    else
        model.user_factors = randn(rng, T, k, n_users) .* T(0.01)
    end
    if V_init !== nothing
        model.item_factors = copy(V_init)
    else
        model.item_factors = randn(rng, T, k, n_items) .* T(0.01)
    end

    U = model.user_factors
    V = model.item_factors

    # CSR for fast row access (user rows)
    X_csr = to_csr(X)
    # CSC is already column-oriented (item columns)

    monitor = ConvergenceMonitor{T}(tol=T(model.convergence_tol), min_iter=2)

    # Pre-allocate per-thread work buffers
    nt = Threads.maxthreadid()
    A_bufs = [Matrix{T}(undef, k, k) for _ in 1:nt]
    b_bufs = [Vector{T}(undef, k) for _ in 1:nt]
    # CG-specific buffers
    r_bufs = [Vector{T}(undef, k) for _ in 1:nt]
    p_bufs = [Vector{T}(undef, k) for _ in 1:nt]
    Ap_bufs = [Vector{T}(undef, k) for _ in 1:nt]

    use_cg = model.solver == :cg
    cg_steps = model.cg_steps

    for iter in 1:model.max_iter
        iter_start = time_ns()

        # ── Update users: fix V, solve for U ──
        if use_cg
            _ials_update_factors_cg!(U, V, X_csr, α, λ, k, cg_steps,
                                     A_bufs, b_bufs, r_bufs, p_bufs, Ap_bufs)
        else
            _ials_update_factors!(U, V, X_csr, α, λ, k, A_bufs, b_bufs)
        end

        # ── Update items: fix U, solve for V ──
        if use_cg
            _ials_update_factors_cg!(V, U, X, α, λ, k, cg_steps,
                                     A_bufs, b_bufs, r_bufs, p_bufs, Ap_bufs)
        else
            _ials_update_factors!(V, U, X, α, λ, k, A_bufs, b_bufs)
        end

        # ── Compute loss ──
        loss = _ials_loss(U, V, X, α, λ)

        iter_seconds = (time_ns() - iter_start) / 1e9
        total_seconds = elapsed_seconds(monitor)

        if model.verbose
            log_iteration("iALS", iter, model.max_iter, Float64(loss),
                         iter_seconds, total_seconds)
        end

        if record!(monitor, loss)
            model.verbose && @info "[iALS] converged at iteration $iter"
            break
        end
    end

    model.is_fitted = true
    model
end

"""
Update all factors in `target` given fixed `source` factors and sparse matrix `R`.
Dispatches on CSR (row access for user updates) or CSC (column access for item updates).
"""
function _ials_update_factors!(target::Matrix{T}, source::Matrix{T},
                               R::SparseMatricesCSR.SparseMatrixCSR, α::T, λ::T, k::Int,
                               A_bufs::Vector{Matrix{T}},
                               b_bufs::Vector{Vector{T}}) where {T}
    _ials_update_core!(target, source, R, α, λ, k, A_bufs, b_bufs,
                       u -> nzrange(R, u),
                       idx -> Int(R.colval[idx]),
                       idx -> T(R.nzval[idx]))
end

function _ials_update_factors!(target::Matrix{T}, source::Matrix{T},
                               R::SparseMatrixCSC, α::T, λ::T, k::Int,
                               A_bufs::Vector{Matrix{T}},
                               b_bufs::Vector{Vector{T}}) where {T}
    _ials_update_core!(target, source, R, α, λ, k, A_bufs, b_bufs,
                       j -> nzrange(R, j),
                       idx -> Int(rowvals(R)[idx]),
                       idx -> T(nonzeros(R)[idx]))
end

function _ials_update_core!(target::Matrix{T}, source::Matrix{T},
                            R, α::T, λ::T, k::Int,
                            A_bufs::Vector{Matrix{T}},
                            b_bufs::Vector{Vector{T}},
                            get_range::Function,
                            get_col::Function,
                            get_val::Function) where {T}
    n = size(target, 2)

    # Precompute Gramian: SᵀS + λI (shared across all users/items)
    gramian = source * source'  # k×k
    gramian .+= λ .* I(k)

    Threads.@threads :static for u in 1:n
        tid = Threads.threadid()
        A = A_bufs[tid]
        b = b_bufs[tid]

        # Start with the shared Gramian
        copyto!(A, gramian)
        fill!(b, zero(T))

        # Add per-entity corrections from non-zero entries
        @inbounds for idx in get_range(u)
            i = get_col(idx)
            r_ui = get_val(idx)
            c_ui = α * r_ui  # confidence boost (beyond base 1 already in gramian)

            # A += c_ui * s_i * s_i^T (rank-1 update)
            for q in 1:k
                sq = source[q, i]
                bq = sq * (one(T) + c_ui)  # preference=1 * confidence
                b[q] += bq
                for p in 1:k
                    A[p, q] += c_ui * source[p, i] * sq
                end
            end
        end

        # Solve A * x = b via Cholesky
        C = cholesky!(Symmetric(A))
        x = C \ b
        @inbounds for f in 1:k
            target[f, u] = x[f]
        end
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# CG solver path — O(d² × cg_steps) per user instead of O(d³)
# Uses warm-start from previous iteration's embedding as initial guess.
# ──────────────────────────────────────────────────────────────────────────────

function _ials_update_factors_cg!(target::Matrix{T}, source::Matrix{T},
                                  R::SparseMatricesCSR.SparseMatrixCSR, α::T, λ::T,
                                  k::Int, cg_steps::Int,
                                  A_bufs::Vector{Matrix{T}},
                                  b_bufs::Vector{Vector{T}},
                                  r_bufs::Vector{Vector{T}},
                                  p_bufs::Vector{Vector{T}},
                                  Ap_bufs::Vector{Vector{T}}) where {T}
    _ials_update_cg_core!(target, source, R, α, λ, k, cg_steps,
                          A_bufs, b_bufs, r_bufs, p_bufs, Ap_bufs,
                          u -> nzrange(R, u),
                          idx -> Int(R.colval[idx]),
                          idx -> T(R.nzval[idx]))
end

function _ials_update_factors_cg!(target::Matrix{T}, source::Matrix{T},
                                  R::SparseMatrixCSC, α::T, λ::T,
                                  k::Int, cg_steps::Int,
                                  A_bufs::Vector{Matrix{T}},
                                  b_bufs::Vector{Vector{T}},
                                  r_bufs::Vector{Vector{T}},
                                  p_bufs::Vector{Vector{T}},
                                  Ap_bufs::Vector{Vector{T}}) where {T}
    _ials_update_cg_core!(target, source, R, α, λ, k, cg_steps,
                          A_bufs, b_bufs, r_bufs, p_bufs, Ap_bufs,
                          j -> nzrange(R, j),
                          idx -> Int(rowvals(R)[idx]),
                          idx -> T(nonzeros(R)[idx]))
end

function _ials_update_cg_core!(target::Matrix{T}, source::Matrix{T},
                               R, α::T, λ::T, k::Int, cg_steps::Int,
                               A_bufs::Vector{Matrix{T}},
                               b_bufs::Vector{Vector{T}},
                               r_bufs::Vector{Vector{T}},
                               p_bufs::Vector{Vector{T}},
                               Ap_bufs::Vector{Vector{T}},
                               get_range::Function,
                               get_col::Function,
                               get_val::Function) where {T}
    n = size(target, 2)

    # Precompute Gramian: SᵀS + λI (shared across all users/items)
    gramian = source * source'  # k×k
    gramian .+= λ .* I(k)

    Threads.@threads :static for u in 1:n
        tid = Threads.threadid()
        A = A_bufs[tid]
        b = b_bufs[tid]
        r = r_bufs[tid]
        p = p_bufs[tid]
        Ap = Ap_bufs[tid]

        # Start with the shared Gramian
        copyto!(A, gramian)
        fill!(b, zero(T))

        # Add per-entity corrections from non-zero entries
        @inbounds for idx in get_range(u)
            i = get_col(idx)
            r_ui = get_val(idx)
            c_ui = α * r_ui

            for q in 1:k
                sq = source[q, i]
                bq = sq * (one(T) + c_ui)
                b[q] += bq
                for pp in 1:k
                    A[pp, q] += c_ui * source[pp, i] * sq
                end
            end
        end

        # CG solve with warm-start from current embedding
        # x₀ = target[:, u] (warm start)
        # r₀ = b - A*x₀
        x = @view target[:, u]
        mul!(r, A, x)
        @inbounds @simd for f in 1:k
            r[f] = b[f] - r[f]
            p[f] = r[f]
        end
        rs_old = dot(r, r)

        for _ in 1:cg_steps
            # Ap = A * p
            mul!(Ap, A, p)
            pAp = dot(p, Ap)
            pAp < eps(T) && break
            α_cg = rs_old / pAp

            @inbounds @simd for f in 1:k
                x[f] += α_cg * p[f]
                r[f] -= α_cg * Ap[f]
            end

            rs_new = dot(r, r)
            rs_new < eps(T) && break
            β = rs_new / rs_old

            @inbounds @simd for f in 1:k
                p[f] = r[f] + β * p[f]
            end
            rs_old = rs_new
        end
    end
end

function _ials_loss(U::Matrix{T}, V::Matrix{T},
                    X::SparseMatrixCSC, α::T, λ::T) where {T}
    rv = rowvals(X)
    nz = nonzeros(X)
    loss = zero(T)
    k = size(U, 1)

    for j in axes(X, 2)
        for idx in nzrange(X, j)
            i = rv[idx]
            r = T(nz[idx])
            pred = zero(T)
            @inbounds @simd for f in 1:k
                pred += U[f, i] * V[f, j]
            end
            c = one(T) + α * r
            loss += c * (one(T) - pred)^2
        end
    end
    loss + λ * (sum(abs2, U) + sum(abs2, V))
end

# ──────────────────────────────────────────────────────────────────────────────
# predict
# ──────────────────────────────────────────────────────────────────────────────

"""
    predict(model::IALS, X; k=10) -> Matrix{Int}

Return top-k item indices per user. Excludes already-interacted items.
"""
function predict(model::IALS{T}, X::SparseMatrixCSC; k::Int=10) where {T}
    model.is_fitted || error("Model not fitted")
    n_users = size(X, 1)
    n_items = size(model.item_factors, 2)
    k_out = min(k, n_items)

    preds = Matrix{Int}(undef, n_users, k_out)
    scores_buf = Vector{T}(undef, n_items)
    X_csr = to_csr(X)

    for u in 1:n_users
        # Compute scores = U[:,u]ᵀ V
        mul!(scores_buf, model.item_factors', @view(model.user_factors[:, u]))

        # Mask already-seen items using CSR row access
        @inbounds for idx in nzrange(X_csr, u)
            j = Int(X_csr.colval[idx])
            scores_buf[j] = T(-Inf)
        end

        # Top-k via partial sort
        topk_idx = partialsortperm(scores_buf, 1:k_out; rev=true)
        preds[u, :] .= topk_idx
    end
    preds
end

"""
    predict_scores(model::IALS, user_indices, item_indices) -> Vector

Return raw scores for specific (user, item) pairs.
"""
function predict_scores(model::IALS{T}, user_indices::AbstractVector{<:Integer},
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
