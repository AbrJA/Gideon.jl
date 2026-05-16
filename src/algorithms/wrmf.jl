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

using LinearAlgebra, SparseArrays, Random

"""
    WRMF{T} <: AbstractMatrixFactorization

Weighted Regularized Matrix Factorization via Alternating Least Squares.

# Fields
- `rank::Int`          — latent dimension
- `λ::T`              — regularisation strength
- `α::T`              — confidence weight for implicit feedback
- `max_iter::Int`      — maximum ALS iterations
- `solver::ALSSolver`  — `CHOLESKY` or `CONJUGATE_GRADIENT`
- `cg_steps::Int`      — max CG inner iterations (only used when solver == CONJUGATE_GRADIENT)
- `feedback::FeedbackType` — `IMPLICIT` or `EXPLICIT`
- `user_factors::Matrix{T}`  — rank × n_users  (set after `fit!`)
- `item_factors::Matrix{T}`  — rank × n_items  (set after `fit!`)
- `user_bias::Vector{T}`     — per-user bias   (length n_users)
- `item_bias::Vector{T}`     — per-item bias   (length n_items)
- `global_bias::T`           — global mean
"""
mutable struct WRMF{T<:AbstractFloat} <: AbstractMatrixFactorization
    rank::Int
    λ::T
    α::T
    max_iter::Int
    solver::ALSSolver
    cg_steps::Int
    feedback::FeedbackType
    user_factors::Matrix{T}
    item_factors::Matrix{T}
    user_bias::Vector{T}
    item_bias::Vector{T}
    global_bias::T
    is_fitted::Bool
end

function WRMF(;
    rank::Int = 10,
    λ::Float64 = 0.1,
    α::Float64 = 1.0,
    max_iter::Int = 10,
    solver::ALSSolver = CONJUGATE_GRADIENT,
    cg_steps::Int = 3,
    feedback::FeedbackType = IMPLICIT,
)
    WRMF{Float64}(
        rank, λ, α, max_iter, solver, cg_steps, feedback,
        Matrix{Float64}(undef, 0, 0),   # user_factors placeholder
        Matrix{Float64}(undef, 0, 0),   # item_factors placeholder
        Float64[], Float64[], 0.0, false,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# fit!
# ──────────────────────────────────────────────────────────────────────────────

function fit!(model::WRMF{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG = Random.default_rng(),
              convergence_tol::Float64 = 0.005) where {T,Tv,Ti}
    n_users, n_items = size(X)
    k = model.rank

    # Initialise factor matrices (rank × n)
    model.user_factors = init_factors(rng, k, n_users)
    model.item_factors = init_factors(rng, k, n_items)
    model.user_bias    = zeros(T, n_users)
    model.item_bias    = zeros(T, n_items)
    model.global_bias  = zero(T)

    # Build transpose for fast row access
    Xt = SparseMatrixCSC(X')  # n_items × n_users

    prev_loss = T(Inf)

    for iter in 1:model.max_iter
        # ---- Update user factors (fixing items) ----
        # Xt columns = users, row values = item indices → indexes into item_factors
        _als_sweep!(model, Xt, model.user_factors, model.item_factors, n_users, true)

        # ---- Update item factors (fixing users) ----
        # X columns = items, row values = user indices → indexes into user_factors
        _als_sweep!(model, X, model.item_factors, model.user_factors, n_items, false)

        loss = _compute_loss(model, X)
        @debug "WRMF iter=$iter  loss=$loss"

        if iter > 1 && abs(prev_loss - loss) / (abs(prev_loss) + T(1e-12)) < convergence_tol
            @debug "WRMF converged at iter=$iter"
            break
        end
        prev_loss = loss
    end
    model.is_fitted = true
    model
end

# ──────────────────────────────────────────────────────────────────────────────
# ALS sweep — update one side of factors, multithreaded
# ──────────────────────────────────────────────────────────────────────────────

function _als_sweep!(
    model::WRMF{T},
    A::SparseMatrixCSC,      # n_entities × n_fixed   (CSC — columns = fixed side)
    factors::Matrix{T},      # k × n_entities  (to be updated)
    fixed::Matrix{T},        # k × n_fixed
    n_entities::Int,
    is_user_side::Bool,
) where {T}
    k  = model.rank
    λ  = model.λ

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
    # Only upper triangle is valid; we symmetrize before use.
    YtY = Matrix{T}(undef, k, k)
    BLAS.syrk!('U', 'N', one(T), fixed, zero(T), YtY)
    LinearAlgebra.copytri!(YtY, 'U')   # mirror upper → lower

    rv = rowvals(A)
    nz = nonzeros(A)

    # Pre-allocate per-thread buffers.
    # maxthreadid() covers interactive threads (IDs > nthreads()) that may also
    # participate in @threads :static on the calling task's thread.
    nt = Threads.maxthreadid()
    gram_bufs = [Matrix{T}(undef, k, k) for _ in 1:nt]
    rhs_bufs  = [Vector{T}(undef, k)    for _ in 1:nt]

    Base.Threads.@threads :static for u in 1:n_entities
        tid  = Threads.threadid()
        gram = gram_bufs[tid]
        rhs  = rhs_bufs[tid]

        # gram ← YᵀY + λI   (copy from shared, then add diagonal)
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
                # gram += (cᵤᵢ - 1) · yᵢ yᵢᵀ  via BLAS rank-1 symmetric update
                BLAS.syr!('U', cui - one(T), yi, gram)
                # rhs += cᵤᵢ · yᵢ
                BLAS.axpy!(cui, yi, rhs)
            else
                BLAS.axpy!(rui, yi, rhs)
            end
        end

        # Mirror upper triangle (syr! only writes upper)
        LinearAlgebra.copytri!(gram, 'U')

        # In-place Cholesky: potrf! factors gram → U, potrs! solves using U
        LAPACK.potrf!('U', gram)
        LAPACK.potrs!('U', gram, rhs)   # rhs is overwritten with solution

        if is_nnls
            # Coordinate-descent NNLS refinement  (true NNLS, not a clamp)
            # Restart from the Cholesky solution projected to non-negative orthant
            rhs .= max.(rhs, zero(T))
            _nnls_cd!(rhs, YtY, fixed, rv, nz, nzrange(A, u), k, α, λ, is_implicit)
        end

        @inbounds factors[:, u] .= rhs
    end
end

"""
    _nnls_cd!(x, YtY, Y, rv, nz, col_range, k, α, λ, is_implicit; max_iter=50)

Block-coordinate descent NNLS: minimises ‖Ax - b‖² s.t. x ≥ 0
where A = YtY + λI + Σ wᵢ yᵢyᵢᵀ.  Updates `x` in-place.
"""
function _nnls_cd!(
    x::Vector{T}, YtY::Matrix{T}, Y::Matrix{T},
    rv, nz, col_range, k::Int,
    α::T, λ::T, is_implicit::Bool; max_iter::Int = 50,
) where {T}
    # Build gram = YtY + λI + Σ wᵢ yᵢyᵢᵀ  and  rhs = Σ cᵢ yᵢ
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
        prev = copy(x)
        for d in 1:k
            @inbounds numer = rhs[d] - BLAS.dot(k, @view(gram[:,d]), 1, x, 1) + gram[d,d]*x[d]
            @inbounds x[d] = max(zero(T), numer / gram[d, d])
        end
        norm(x .- prev) < T(1e-8) * (norm(x) + T(1e-12)) && break
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

    # Base gram (shared, read-only inside @threads)
    YtY = Matrix{T}(undef, k, k)
    BLAS.syrk!('U', 'N', one(T), fixed, zero(T), YtY)
    LinearAlgebra.copytri!(YtY, 'U')
    base_gram = copy(YtY)
    @inbounds for d in 1:k; base_gram[d,d] += λ; end

    rv = rowvals(A)
    nz = nonzeros(A)

    # Pre-allocate per-thread CG workspace buffers.
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

Solve `(base_gram + Σ_j w_j y_j y_jᵀ) x = b` via CG.
`x`, `r`, `p`, `Ap` are pre-allocated per-thread buffers — no heap allocation.
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
    rs_old = dot(r, r)   # LinearAlgebra.dot — type-stable, avoids BLAS union split

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
    @inbounds for pos in eachindex(indices)
        i = indices[pos]
        w = weights[pos]
        iszero(w) && continue
        d = BLAS.dot(k, @view(Y[:, i]), 1, v, 1)
        BLAS.axpy!(w * d, @view(Y[:, i]), result)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Loss computation
# ──────────────────────────────────────────────────────────────────────────────

function _compute_loss(model::WRMF{T}, X::SparseMatrixCSC) where {T}
    U = model.user_factors  # k × n_users
    V = model.item_factors  # k × n_items
    λ = model.λ
    α = model.α

    loss = zero(T)
    rv = rowvals(X)
    nz = nonzeros(X)

    for j in axes(X, 2)
        for idx in nzrange(X, j)
            i = rv[idx]
            r = T(nz[idx])
            pred = dot(@view(U[:, i]), @view(V[:, j]))
            if model.feedback == IMPLICIT
                c = one(T) + α * r
                loss += c * (one(T) - pred)^2
            else
                loss += (r - pred)^2
            end
        end
    end

    # Regularization
    loss += λ * (sum(abs2, U) + sum(abs2, V))
    loss
end

# ──────────────────────────────────────────────────────────────────────────────
# transform / predict
# ──────────────────────────────────────────────────────────────────────────────

function transform(model::WRMF{T}, X::SparseMatrixCSC) where {T}
    model.is_fitted || error("Model not fitted. Call fit! first.")
    n_users_new = size(X, 1)
    k = model.rank
    new_user_factors = Matrix{T}(undef, k, n_users_new)
    fill!(new_user_factors, zero(T))

    Xt = SparseMatrixCSC(X')
    # Use Xt so columns = users, row values = item indices
    _als_sweep!(model, Xt, new_user_factors, model.item_factors, n_users_new, true)
    new_user_factors
end

function predict(model::WRMF{T}, X::SparseMatrixCSC; k::Int = 10) where {T}
    model.is_fitted || error("Model not fitted. Call fit! first.")
    user_emb = transform(model, X)
    # scores = Uᵀ V  →  n_users × n_items
    scores = user_emb' * model.item_factors
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
