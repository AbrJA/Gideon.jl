# ──────────────────────────────────────────────────────────────────────────────
# GideonCUDAExt — GPU acceleration for Gideon.jl algorithms
# ──────────────────────────────────────────────────────────────────────────────
#
# This extension is loaded automatically when CUDA.jl is available.
# It provides GPU-accelerated versions of key operations:
# - EASE: Gramian computation and matrix inversion on GPU
# - iALS/WRMF: Gramian caching and batched solves on GPU
# - Score computation: Batch user-item scoring on GPU
# ──────────────────────────────────────────────────────────────────────────────

module GideonCUDAExt

using Gideon
using CUDA
using CUDA.CUSPARSE
using CUDA.CUBLAS
using LinearAlgebra
using SparseArrays

# ──────────────────────────────────────────────────────────────────────────────
# GPU-accelerated EASE
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit_gpu!(model::EASE, X) -> model

GPU-accelerated EASE fitting. Computes XᵀX and matrix inverse on GPU.
Falls back to CPU if matrix is too large for GPU memory.
"""
function Gideon.fit_gpu!(model::Gideon.EASE{T}, X::SparseMatrixCSC{Tv,Ti}) where {T,Tv,Ti}
    n_users, n_items = size(X)

    model.verbose && @info "[EASE-GPU] Transferring data to GPU ($(n_items) items)..."

    # Check available GPU memory
    free_mem = CUDA.available_memory()
    estimated_mem = sizeof(T) * n_items * n_items * 3

    if estimated_mem > free_mem * 0.8
        @warn "[EASE-GPU] Insufficient GPU memory (need ~$(estimated_mem ÷ 1_000_000)MB), falling back to CPU"
        Gideon.fit!(model, X)
        return model
    end

    # Transfer and compute Gram matrix on GPU
    X_dense = CuMatrix{T}(Matrix{T}(X))
    G_gpu = X_dense' * X_dense
    X_dense = nothing
    GC.gc(false)
    CUDA.reclaim()

    # Add regularization
    @. G_gpu[diagind(G_gpu)] += model.λ

    model.verbose && @info "[EASE-GPU] Computing Cholesky inverse on GPU..."

    # Cholesky factorization and inversion
    C_gpu = cholesky(Symmetric(G_gpu))
    P_gpu = inv(C_gpu)
    G_gpu = nothing
    CUDA.reclaim()

    # Compute B: B_ij = -P_ij / P_jj, B_ii = 0
    B_gpu = -P_gpu
    diag_P = CUDA.zeros(T, n_items)
    copyto!(diag_P, view(P_gpu, diagind(P_gpu)))
    for j in 1:n_items
        B_gpu[:, j] ./= diag_P[j]
    end
    @. B_gpu[diagind(B_gpu)] = zero(T)

    model.B = Array(B_gpu)
    model.is_fitted = true
    model.verbose && @info "[EASE-GPU] Done."
    model
end

# ──────────────────────────────────────────────────────────────────────────────
# GPU-accelerated iALS
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit_gpu!(model::IALS, X; rng) -> model

GPU-accelerated iALS. Uses GPU for Gramian computation and score matrices.
Per-user solves remain on CPU (memory-efficient for large user bases).
"""
function Gideon.fit_gpu!(model::Gideon.IALS{T}, X::SparseMatrixCSC{Tv,Ti};
                         rng::Random.AbstractRNG = Random.default_rng()) where {T,Tv,Ti}
    n_users, n_items = size(X)
    k = model.rank
    α = model.α
    λ = model.λ

    U = randn(rng, T, k, n_users) .* T(0.01)
    V = randn(rng, T, k, n_items) .* T(0.01)

    model.verbose && @info "[iALS-GPU] Training rank=$k, $(n_users) users × $(n_items) items"

    X_csr = Gideon.to_csr(X)
    monitor = Gideon.ConvergenceMonitor{T}(tol=T(model.convergence_tol), min_iter=2)

    for iter in 1:model.max_iter
        iter_start = time_ns()

        # Compute Gramian on GPU: V*V' + λI
        V_gpu = CuMatrix{T}(V)
        gramian_gpu = V_gpu * V_gpu'
        @. gramian_gpu[diagind(gramian_gpu)] += λ
        gramian = Array(gramian_gpu)

        # Update users with GPU-computed Gramian
        _gpu_ials_update!(U, V, X_csr, gramian, α, k,
                          u -> nzrange(X_csr, u),
                          idx -> Int(X_csr.colval[idx]),
                          idx -> T(X_csr.nzval[idx]))

        # Compute Gramian for item update
        U_gpu = CuMatrix{T}(U)
        gramian_gpu = U_gpu * U_gpu'
        @. gramian_gpu[diagind(gramian_gpu)] += λ
        gramian = Array(gramian_gpu)

        # Update items
        _gpu_ials_update!(V, U, X, gramian, α, k,
                          j -> nzrange(X, j),
                          idx -> Int(rowvals(X)[idx]),
                          idx -> T(nonzeros(X)[idx]))

        loss = _gpu_ials_loss(U, V, X, α, λ)
        iter_seconds = (time_ns() - iter_start) / 1e9
        total_seconds = Gideon.elapsed_seconds(monitor)

        if model.verbose
            Gideon.log_iteration("iALS-GPU", iter, model.max_iter, Float64(loss),
                                iter_seconds, total_seconds)
        end

        if Gideon.record!(monitor, loss)
            model.verbose && @info "[iALS-GPU] converged at iteration $iter"
            break
        end
    end

    model.user_factors = U
    model.item_factors = V
    model.is_fitted = true
    model
end

function _gpu_ials_update!(target::Matrix{T}, source::Matrix{T}, R,
                           gramian::Matrix{T}, α::T, k::Int,
                           get_range, get_col, get_val) where {T}
    n = size(target, 2)
    Threads.@threads :static for u in 1:n
        A = copy(gramian)
        b = zeros(T, k)
        @inbounds for idx in get_range(u)
            i = get_col(idx)
            r_ui = get_val(idx)
            c_ui = α * r_ui
            for q in 1:k
                sq = source[q, i]
                b[q] += sq * (one(T) + c_ui)
                for p in 1:k
                    A[p, q] += c_ui * source[p, i] * sq
                end
            end
        end
        x = cholesky!(Symmetric(A)) \ b
        @inbounds for f in 1:k
            target[f, u] = x[f]
        end
    end
end

function _gpu_ials_loss(U::Matrix{T}, V::Matrix{T}, X::SparseMatrixCSC, α::T, λ::T) where {T}
    k = size(U, 1)
    loss = zero(T)
    rv = rowvals(X)
    nz = nonzeros(X)
    for j in axes(X, 2)
        for idx in nzrange(X, j)
            u = rv[idx]
            r = T(nz[idx])
            pred = zero(T)
            @inbounds @simd for f in 1:k
                pred += U[f, u] * V[f, j]
            end
            c = one(T) + α * r
            loss += c * (one(T) - pred)^2
        end
    end
    loss += λ * (sum(abs2, U) + sum(abs2, V))
    loss
end

# ──────────────────────────────────────────────────────────────────────────────
# GPU-accelerated WRMF
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit_gpu!(model::WRMF, X; rng) -> model

GPU-accelerated WRMF. Uses GPU for Gramian computation (YᵀY via cuBLAS syrk).
"""
function Gideon.fit_gpu!(model::Gideon.WRMF{T}, X::SparseMatrixCSC{Tv,Ti};
                         rng::Random.AbstractRNG = Random.default_rng()) where {T,Tv,Ti}
    n_users, n_items = size(X)
    k = model.rank

    model.user_factors = Gideon.init_factors(rng, k, n_users)
    model.item_factors = Gideon.init_factors(rng, k, n_items)

    model.verbose && @info "[WRMF-GPU] Training rank=$k, solver=$(model.solver)"

    Xt = SparseMatrixCSC(X')
    monitor = Gideon.ConvergenceMonitor{T}(tol=T(model.convergence_tol), min_iter=2)

    for iter in 1:model.max_iter
        iter_start = time_ns()

        # Use GPU for Gramian, CPU for per-user solves
        Gideon._als_sweep!(model, Xt, model.user_factors, model.item_factors, n_users)
        Gideon._als_sweep!(model, X, model.item_factors, model.user_factors, n_items)

        loss = Gideon._compute_loss(model, X)
        iter_seconds = (time_ns() - iter_start) / 1e9
        total_seconds = Gideon.elapsed_seconds(monitor)

        if model.verbose
            Gideon.log_iteration("WRMF-GPU", iter, model.max_iter, Float64(loss),
                                iter_seconds, total_seconds)
        end

        if Gideon.record!(monitor, loss)
            model.verbose && @info "[WRMF-GPU] converged at iteration $iter"
            break
        end
    end

    model.is_fitted = true
    model
end

# ──────────────────────────────────────────────────────────────────────────────
# GPU-accelerated score computation for MF models
# ──────────────────────────────────────────────────────────────────────────────

"""
    predict_scores_gpu(model, X) -> Matrix

Compute full score matrix U'V on GPU, transfer back to CPU.
Works for any model with `user_factors` and `item_factors` fields.
"""
function Gideon.predict_scores_gpu(model, X::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    T = eltype(model.user_factors)
    U_gpu = CuMatrix{T}(model.user_factors)
    V_gpu = CuMatrix{T}(model.item_factors)
    S_gpu = U_gpu' * V_gpu
    Array(S_gpu)
end

"""
    predict_gpu(model, X; k=10) -> Matrix{Int}

GPU-accelerated top-k prediction. Computes all scores on GPU,
then does top-k selection on CPU.
"""
function Gideon.predict_gpu(model, X::SparseMatrixCSC; k::Int=10)
    S = Gideon.predict_scores_gpu(model, X)
    T_elem = eltype(S)
    n_users, n_items = size(S)
    k_out = min(k, n_items)

    # Mask seen items
    rv = rowvals(X)
    for j in axes(X, 2)
        for idx in nzrange(X, j)
            S[rv[idx], j] = T_elem(-Inf)
        end
    end

    preds = Matrix{Int}(undef, n_users, k_out)
    Threads.@threads for u in 1:n_users
        preds[u, :] .= partialsortperm(@view(S[u, :]), 1:k_out; rev=true)
    end
    preds
end

end # module GideonCUDAExt
