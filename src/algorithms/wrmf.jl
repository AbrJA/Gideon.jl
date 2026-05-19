# ──────────────────────────────────────────────────────────────────────────────
# WRMF — Weighted Regularized Matrix Factorization (Implicit ALS)
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: Hu, Koren, Volinsky (2008)
#   "Collaborative Filtering for Implicit Feedback Datasets"
#
# Loss:
#   L = Σ_{u,i} c_{ui}(p_{ui} - xᵤᵀ yᵢ)² + λ(Σ_u ||xᵤ||² + Σ_i ||yᵢ||²)
#
# where c_{ui} = 1 + α·r_{ui}  and  p_{ui} = r_{ui} > 0
#
# Optimisations:
#   • Per-thread pre-allocated gram/rhs/Chol buffers → zero inner-loop allocs
#   • BLAS.syr! for O(k²) rank-1 gram accumulation (vectorised BLAS-2)
#   • BLAS.axpy! for O(k) rhs accumulation
#   • In-place LAPACK.potrf! + LAPACK.potrs! → no extra matrices in Chol path
#   • Coordinate-descent NNLS (true bounded NNLS, not a clamp)
#   • BLAS.syrk! for YᵀY (symmetric rank-k update)
#   • Base.Threads.@threads :static for stable thread IDs
# ──────────────────────────────────────────────────────────────────────────────

"""
    WRMF{T} <: AbstractMatrixFactorization

Weighted Regularized Matrix Factorization via Alternating Least Squares.

Supports implicit feedback (Hu et al. 2008) and explicit feedback (MSE).
Three solvers available: Cholesky (exact), Conjugate Gradient (approximate, fast),
and NNLS (non-negative matrix factorization).

# Constructor
```julia
WRMF(; rank=10, λ=0.1, α=1.0, max_iter=10, convergence_tol=0.005,
       solver=CONJUGATE_GRADIENT, cg_steps=3, feedback=IMPLICIT, verbose=true)
```

# Fields
- `rank::Int`          — latent dimension
- `λ::T`              — regularisation strength
- `α::T`              — confidence weight for implicit feedback
- `max_iter::Int`      — maximum ALS iterations
- `convergence_tol::T` — relative loss tolerance for early stopping (<0 disables)
- `solver::ALSSolver`  — `CHOLESKY`, `CONJUGATE_GRADIENT`, or `NNLS`
- `cg_steps::Int`      — max CG inner iterations (only for CG solver)
- `feedback::FeedbackType` — `IMPLICIT` or `EXPLICIT`
- `user_factors::Matrix{T}`  — rank × n_users (set after `fit!`)
- `item_factors::Matrix{T}`  — rank × n_items (set after `fit!`)

# Example
```julia
using SparseArrays, Gideon

X = sprand(1000, 500, 0.01)
model = WRMF(rank=64, λ=0.1, α=40.0, max_iter=15, solver=CONJUGATE_GRADIENT)
fit!(model, X)
scores = predict(model, X; k=10)
```
"""
mutable struct WRMF{T<:AbstractFloat} <: AbstractMatrixFactorization
    rank::Int
    λ::T
    α::T
    max_iter::Int
    convergence_tol::T
    solver::ALSSolver
    cg_steps::Int
    feedback::FeedbackType
    verbose::Bool
    user_factors::Matrix{T}
    item_factors::Matrix{T}
    is_fitted::Bool
end

function WRMF(;
    rank::Int = 10,
    λ::Float64 = 0.1,
    α::Float64 = 1.0,
    max_iter::Int = 10,
    convergence_tol::Float64 = 0.005,
    solver::ALSSolver = CONJUGATE_GRADIENT,
    cg_steps::Int = 3,
    feedback::FeedbackType = IMPLICIT,
    verbose::Bool = true,
)
    @assert rank >= 1 "rank must be ≥ 1"
    @assert λ >= 0.0 "λ must be non-negative"
    @assert α >= 0.0 "α must be non-negative"
    @assert max_iter >= 1 "max_iter must be ≥ 1"
    @assert cg_steps >= 1 "cg_steps must be ≥ 1"
    WRMF{Float64}(
        rank, λ, α, max_iter, convergence_tol, solver, cg_steps, feedback, verbose,
        Matrix{Float64}(undef, 0, 0),
        Matrix{Float64}(undef, 0, 0),
        false,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# fit!
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit!(model::WRMF, X::SparseMatrixCSC; rng, U_init, V_init) -> model

Fit the WRMF model on user-item sparse matrix `X` (n_users × n_items).

# Keyword Arguments
- `rng::AbstractRNG = Random.default_rng()` — random number generator
- `U_init::Union{Nothing, Matrix}` — warm-start user factors (rank × n_users)
- `V_init::Union{Nothing, Matrix}` — warm-start item factors (rank × n_items)
"""
function fit!(model::WRMF{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG = Random.default_rng(),
              U_init::Union{Nothing, Matrix{T}} = nothing,
              V_init::Union{Nothing, Matrix{T}} = nothing) where {T,Tv,Ti}
    n_users, n_items = size(X)
    k = model.rank

    # Initialise factor matrices
    model.user_factors = isnothing(U_init) ? init_factors(rng, k, n_users) : copy(U_init)
    model.item_factors = isnothing(V_init) ? init_factors(rng, k, n_items) : copy(V_init)

    # Build transpose for fast row access
    Xt = SparseMatrixCSC(X')  # n_items × n_users

    monitor = ConvergenceMonitor{T}(tol=T(model.convergence_tol), min_iter=2)

    for iter in 1:model.max_iter
        iter_start = time_ns()

        # Update user factors (fixing items)
        _als_sweep!(model, Xt, model.user_factors, model.item_factors, n_users)
        # Update item factors (fixing users)
        _als_sweep!(model, X, model.item_factors, model.user_factors, n_items)

        loss = _compute_loss(model, X)
        iter_seconds = (time_ns() - iter_start) / 1e9
        total_seconds = elapsed_seconds(monitor)

        if model.verbose
            log_iteration("WRMF", iter, model.max_iter, Float64(loss),
                         iter_seconds, total_seconds)
        end

        if record!(monitor, loss)
            model.verbose && @info "[WRMF] converged at iteration $iter"
            break
        end
    end
    model.is_fitted = true
    model
end

# ──────────────────────────────────────────────────────────────────────────────
# ALS sweep — update one side of factors, multithreaded
# ──────────────────────────────────────────────────────────────────────────────

function _als_sweep!(
    model::WRMF{T},
    A::SparseMatrixCSC,
    factors::Matrix{T},
    fixed::Matrix{T},
    n_entities::Int,
) where {T}
    if model.solver == CHOLESKY || model.solver == NNLS
        _als_sweep_cholesky!(model, A, factors, fixed, n_entities)
    else
        _als_sweep_cg!(model, A, factors, fixed, n_entities)
    end
end

# ──────────────── Cholesky path ────────────────

function _als_sweep_cholesky!(
    model::WRMF{T},
    A::SparseMatrixCSC,
    factors::Matrix{T},
    fixed::Matrix{T},
    n_entities::Int,
) where {T}
    k = model.rank
    λ = model.λ
    α = model.α
    is_implicit = model.feedback == IMPLICIT
    is_nnls     = model.solver == NNLS

    # YᵀY via BLAS syrk (symmetric rank-k: C = α·A·Aᵀ + β·C)
    YtY = Matrix{T}(undef, k, k)
    BLAS.syrk!('U', 'N', one(T), fixed, zero(T), YtY)
    LinearAlgebra.copytri!(YtY, 'U')

    rv = rowvals(A)
    nz = nonzeros(A)

    # Pre-allocate per-thread buffers
    nt = Threads.maxthreadid()
    gram_bufs = [Matrix{T}(undef, k, k) for _ in 1:nt]
    rhs_bufs  = [Vector{T}(undef, k)    for _ in 1:nt]

    Base.Threads.@threads :static for u in 1:n_entities
        tid  = Threads.threadid()
        gram = gram_bufs[tid]
        rhs  = rhs_bufs[tid]

        # gram ← YᵀY + λI
        copyto!(gram, YtY)
        @inbounds for d in 1:k
            gram[d, d] += λ
        end
        fill!(rhs, zero(T))

        for idx in nzrange(A, u)
            i   = rv[idx]
            rui = T(nz[idx])
            yi  = @view fixed[:, i]

            if is_implicit
                cui = one(T) + α * rui
                BLAS.syr!('U', cui - one(T), yi, gram)
                BLAS.axpy!(cui, yi, rhs)
            else
                BLAS.axpy!(rui, yi, rhs)
            end
        end

        # Mirror upper triangle
        LinearAlgebra.copytri!(gram, 'U')

        # In-place Cholesky solve
        LAPACK.potrf!('U', gram)
        LAPACK.potrs!('U', gram, rhs)

        if is_nnls
            rhs .= max.(rhs, zero(T))
            _nnls_cd!(rhs, YtY, fixed, rv, nz, nzrange(A, u), k, α, λ, is_implicit)
        end

        @inbounds factors[:, u] .= rhs
    end
end

"""
    _nnls_cd!(x, YtY, Y, rv, nz, col_range, k, α, λ, is_implicit; max_iter=50)

Block-coordinate descent NNLS: minimises ‖Ax - b‖² s.t. x ≥ 0.
Updates `x` in-place.
"""
function _nnls_cd!(
    x::Vector{T}, YtY::Matrix{T}, Y::Matrix{T},
    rv, nz, col_range, k::Int,
    α::T, λ::T, is_implicit::Bool; max_iter::Int = 50,
) where {T}
    gram = copy(YtY)
    @inbounds for d in 1:k; gram[d,d] += λ; end
    rhs = zeros(T, k)
    for idx in col_range
        i   = rv[idx]
        rui = T(nz[idx])
        yi  = @view Y[:, i]
        if is_implicit
            cui = one(T) + α * rui
            BLAS.syr!('U', cui - one(T), yi, gram)
            BLAS.axpy!(cui, yi, rhs)
        else
            BLAS.axpy!(rui, yi, rhs)
        end
    end
    LinearAlgebra.copytri!(gram, 'U')

    # Coordinate descent
    for _ in 1:max_iter
        max_change = zero(T)
        for d in 1:k
            @inbounds numer = rhs[d] - BLAS.dot(k, @view(gram[:,d]), 1, x, 1) + gram[d,d]*x[d]
            new_val = max(zero(T), numer / gram[d, d])
            @inbounds max_change = max(max_change, abs(new_val - x[d]))
            @inbounds x[d] = new_val
        end
        max_change < T(1e-8) && break
    end
end

# ──────────────── Conjugate Gradient path ────────────────

function _als_sweep_cg!(
    model::WRMF{T},
    A::SparseMatrixCSC,
    factors::Matrix{T},
    fixed::Matrix{T},
    n_entities::Int,
) where {T}
    k = model.rank
    λ = model.λ
    α = model.α
    cg_steps = model.cg_steps
    is_implicit = model.feedback == IMPLICIT

    # Base gram (shared, read-only)
    YtY = Matrix{T}(undef, k, k)
    BLAS.syrk!('U', 'N', one(T), fixed, zero(T), YtY)
    LinearAlgebra.copytri!(YtY, 'U')
    base_gram = copy(YtY)
    @inbounds for d in 1:k; base_gram[d,d] += λ; end

    rv = rowvals(A)
    nz = nonzeros(A)

    # Per-thread CG workspace
    nt = Threads.maxthreadid()
    max_nnz = maximum(length(nzrange(A, u)) for u in 1:n_entities; init=0)
    rhs_bufs  = [Vector{T}(undef, k)        for _ in 1:nt]
    idx_bufs  = [Vector{Int}(undef, max_nnz) for _ in 1:nt]
    wgt_bufs  = [Vector{T}(undef, max_nnz)   for _ in 1:nt]
    r_bufs    = [Vector{T}(undef, k)        for _ in 1:nt]
    p_bufs    = [Vector{T}(undef, k)        for _ in 1:nt]
    Ap_bufs   = [Vector{T}(undef, k)        for _ in 1:nt]

    Base.Threads.@threads :static for u in 1:n_entities
        tid  = Threads.threadid()
        rhs  = rhs_bufs[tid]
        idxs = idx_bufs[tid]
        wgts = wgt_bufs[tid]

        fill!(rhs, zero(T))
        col_range = nzrange(A, u)
        n_nz = length(col_range)

        for (pos, idx) in enumerate(col_range)
            i   = rv[idx]
            rui = T(nz[idx])
            idxs[pos] = i
            if is_implicit
                cui = one(T) + α * rui
                wgts[pos] = cui - one(T)
                BLAS.axpy!(cui, @view(fixed[:, i]), rhs)
            else
                wgts[pos] = zero(T)
                BLAS.axpy!(rui, @view(fixed[:, i]), rhs)
            end
        end

        xu = @view factors[:, u]
        _cg_solve!(xu, base_gram, fixed,
                   view(idxs, 1:n_nz), view(wgts, 1:n_nz),
                   rhs, k, cg_steps,
                   r_bufs[tid], p_bufs[tid], Ap_bufs[tid])
    end
end

"""
    _cg_solve!(x, base_gram, Y, indices, weights, b, k, max_steps, r, p, Ap)

Solve `(base_gram + Σ_j w_j y_j y_jᵀ) x = b` via Conjugate Gradient.
All vectors are pre-allocated per-thread — zero heap allocation.
"""
function _cg_solve!(
    x::AbstractVector{T},
    base_gram::Matrix{T},
    Y::Matrix{T},
    indices::AbstractVector{Int},
    weights::AbstractVector{T},
    b::Vector{T},
    k::Int,
    max_steps::Int,
    r::Vector{T}, p::Vector{T}, Ap::Vector{T},
) where {T}
    _implicit_matvec!(Ap, base_gram, Y, indices, weights, x, k)
    @inbounds for a in 1:k
        r[a] = b[a] - Ap[a]
        p[a] = r[a]
    end
    rs_old = dot(r, r)

    for _ in 1:max_steps
        _implicit_matvec!(Ap, base_gram, Y, indices, weights, p, k)
        pAp = dot(p, Ap)
        pAp < eps(T) && break
        α_cg = rs_old / pAp
        BLAS.axpy!(α_cg, p, x)
        BLAS.axpy!(-α_cg, Ap, r)
        rs_new = dot(r, r)
        rs_new < eps(T) && break
        β = rs_new / rs_old
        @inbounds @simd for a in 1:k
            p[a] = r[a] + β * p[a]
        end
        rs_old = rs_new
    end
end

function _implicit_matvec!(
    result::Vector{T},
    base_gram::Matrix{T},
    Y::Matrix{T},
    indices::AbstractVector{Int},
    weights::AbstractVector{T},
    v::AbstractVector{T},
    k::Int,
) where {T}
    BLAS.gemv!('N', one(T), base_gram, v, zero(T), result)
    n_nz = length(indices)
    n_nz == 0 && return

    # For very sparse entities (< 32 nnz), avoid BLAS overhead with manual dot+axpy
    if n_nz < 32
        @inbounds for pos in 1:n_nz
            i = indices[pos]
            w = weights[pos]
            iszero(w) && continue
            d = zero(T)
            @simd for f in 1:k
                d += Y[f, i] * v[f]
            end
            wd = w * d
            @simd for f in 1:k
                result[f] += wd * Y[f, i]
            end
        end
    else
        @inbounds for pos in eachindex(indices)
            i = indices[pos]
            w = weights[pos]
            iszero(w) && continue
            d = BLAS.dot(k, @view(Y[:, i]), 1, v, 1)
            BLAS.axpy!(w * d, @view(Y[:, i]), result)
        end
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Loss computation
# ──────────────────────────────────────────────────────────────────────────────

function _compute_loss(model::WRMF{T}, X::SparseMatrixCSC) where {T}
    U = model.user_factors
    V = model.item_factors
    λ = model.λ
    α = model.α
    k = model.rank

    loss = zero(T)
    rv = rowvals(X)
    nz = nonzeros(X)

    for j in axes(X, 2)
        for idx in nzrange(X, j)
            i = rv[idx]
            r = T(nz[idx])
            pred = zero(T)
            @inbounds @simd for f in 1:k
                pred += U[f, i] * V[f, j]
            end
            if model.feedback == IMPLICIT
                c = one(T) + α * r
                loss += c * (one(T) - pred)^2
            else
                loss += (r - pred)^2
            end
        end
    end

    loss += λ * (sum(abs2, U) + sum(abs2, V))
    loss
end

# ──────────────────────────────────────────────────────────────────────────────
# transform / predict
# ──────────────────────────────────────────────────────────────────────────────

"""
    transform(model::WRMF, X::SparseMatrixCSC) -> Matrix

Compute user embeddings for new users given their interaction matrix `X` (n_new × n_items).
Returns a `rank × n_new` factor matrix.
"""
function transform(model::WRMF{T}, X::SparseMatrixCSC) where {T}
    model.is_fitted || error("Model not fitted. Call fit! first.")
    n_users_new = size(X, 1)
    k = model.rank
    new_user_factors = Matrix{T}(undef, k, n_users_new)
    fill!(new_user_factors, zero(T))

    Xt = SparseMatrixCSC(X')
    _als_sweep!(model, Xt, new_user_factors, model.item_factors, n_users_new)
    new_user_factors
end

"""
    predict(model::WRMF, X::SparseMatrixCSC; k=10) -> Matrix{Int}

Return top-k item indices for each user. Returns `n_users × k` matrix.
Processes users in batches to avoid allocating the full score matrix.
"""
function predict(model::WRMF{T}, X::SparseMatrixCSC; k::Int = 10) where {T}
    model.is_fitted || error("Model not fitted. Call fit! first.")
    n_users = size(X, 1)
    n_items = size(model.item_factors, 2)
    k_actual = min(k, n_items)

    # Use cached user factors if dimensions match (training data), else re-embed
    if size(model.user_factors, 2) == n_users
        user_emb = model.user_factors
    else
        user_emb = transform(model, X)
    end

    predictions = Matrix{Int}(undef, n_users, k_actual)

    # Process in batches to limit memory usage
    max_batch_mem = 2 * 1024^3  # 2 GB target
    batch_size = max(1, min(n_users, Int(floor(max_batch_mem / (n_items * sizeof(T))))))

    X_csr = to_csr(X)

    # Pre-allocate buffers — scores stored as n_items × batch_size (column-per-user)
    # so that each user's scores vector is a contiguous column (cache-friendly top-k)
    nt = Threads.maxthreadid()
    topk_bufs = [Vector{Int}(undef, k_actual) for _ in 1:nt]
    scores_buf = Matrix{T}(undef, n_items, batch_size)

    # Use multi-threaded BLAS for the large GEMM
    old_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(Threads.nthreads())

    for batch_start in 1:batch_size:n_users
        batch_end = min(batch_start + batch_size - 1, n_users)
        batch_users = batch_start:batch_end
        n_batch = length(batch_users)

        # GEMM: scores_buf[:,1:n_batch] = item_factors' * user_emb[:,batch_users]
        # Result: n_items × n_batch, each column = scores for one user (contiguous!)
        scores = @view scores_buf[:, 1:n_batch]
        mul!(scores, model.item_factors', @view(user_emb[:, batch_users]))

        # Mask seen items and get top-k per user (threaded)
        Threads.@threads for local_u in 1:n_batch
            tid = Threads.threadid()
            global_u = batch_users[local_u]
            @inbounds for idx in nzrange(X_csr, global_u)
                j = Int(X_csr.colval[idx])
                scores_buf[j, local_u] = T(-Inf)
            end
            col = @view scores_buf[:, local_u]
            topk = topk_bufs[tid]
            _topk_indices!(topk, col, k_actual)
            @inbounds for i in 1:k_actual
                predictions[global_u, i] = topk[i]
            end
        end
    end

    BLAS.set_num_threads(old_blas)
    predictions
end

"""
    predict_scores(model::WRMF, X) -> Matrix

Return the full score matrix (n_users × n_items) without top-k filtering.
Uses `transform` to embed users, then computes inner products with item factors.
"""
function predict_scores(model::WRMF{T}, X::SparseMatrixCSC) where {T}
    model.is_fitted || error("Model not fitted. Call fit! first.")
    user_emb = transform(model, X)
    user_emb' * model.item_factors
end

"""
    predict_scores(model::WRMF, user_indices, item_indices) -> Vector

Return raw scores for specific (user, item) pairs using pre-fitted factors.
"""
function predict_scores(model::WRMF{T}, user_indices::AbstractVector{<:Integer},
                        item_indices::AbstractVector{<:Integer}) where {T}
    model.is_fitted || error("Model not fitted. Call fit! first.")
    @assert length(user_indices) == length(item_indices)
    n = length(user_indices)
    scores = Vector{T}(undef, n)
    k = model.rank
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
