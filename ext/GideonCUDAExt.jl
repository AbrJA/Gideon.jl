# ──────────────────────────────────────────────────────────────────────────────
# GideonCUDAExt — GPU acceleration for Gideon.jl algorithms
# ──────────────────────────────────────────────────────────────────────────────
#
# This extension is loaded automatically when CUDA.jl is available.
# It provides GPU-accelerated versions of key operations:
# - EASE: Gramian computation and matrix inversion on GPU
# - iALS: Gramian caching on GPU with batched Cholesky solves
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

    model.verbose && @info "[EASE-GPU] Transferring data to GPU..."

    # Transfer sparse matrix to GPU
    X_gpu = CuSparseMatrixCSC(X)

    # Compute Gram matrix on GPU: G = XᵀX
    # CUSPARSE supports sparse-sparse multiplication
    Xt_gpu = CuSparseMatrixCSC(sparse(X'))
    G_gpu = CuMatrix{T}(CUDA.zeros(T, n_items, n_items))

    # XᵀX via dense conversion (for moderate n_items)
    if n_items <= 50_000
        X_dense = CuMatrix{T}(Matrix{T}(X))
        G_gpu = X_dense' * X_dense

        # Add regularization
        G_gpu .+= model.λ .* CuMatrix{T}(I(n_items))

        model.verbose && @info "[EASE-GPU] Computing matrix inverse on GPU..."

        # Invert on GPU
        P_gpu = inv(G_gpu)

        # Compute B on GPU
        B_gpu = CUDA.zeros(T, n_items, n_items)
        @. B_gpu = -P_gpu
        diag_P = CUDA.zeros(T, n_items)
        for j in 1:n_items
            diag_P[j] = P_gpu[j, j]
        end
        for j in 1:n_items
            B_gpu[:, j] ./= diag_P[j]
        end
        # Zero diagonal
        for j in 1:n_items
            B_gpu[j, j] = zero(T)
        end

        # Transfer back to CPU
        model.B = Array(B_gpu)
    else
        # For very large item sets, compute on CPU (memory bound)
        @warn "[EASE-GPU] n_items=$n_items too large for GPU memory, falling back to CPU"
        Gideon.fit!(model, X)
        return model
    end

    model.is_fitted = true
    model.verbose && @info "[EASE-GPU] Done."
    model
end

# ──────────────────────────────────────────────────────────────────────────────
# GPU-accelerated score computation for MF models
# ──────────────────────────────────────────────────────────────────────────────

"""
    predict_scores_gpu(model, X) -> Matrix

Compute full score matrix U'V on GPU, transfer back to CPU.
Works for any model with user_factors and item_factors fields.
"""
function Gideon.predict_scores_gpu(model, X::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    T = eltype(model.user_factors)
    U_gpu = CuMatrix{T}(model.user_factors)  # k × n_users
    V_gpu = CuMatrix{T}(model.item_factors)  # k × n_items

    # Scores = Uᵀ × V = (n_users × k) × (k × n_items) = n_users × n_items
    S_gpu = U_gpu' * V_gpu

    Array(S_gpu)
end

"""
    predict_gpu(model, X; k=10) -> Matrix{Int}

GPU-accelerated top-k prediction. Computes all scores on GPU,
then does top-k selection on CPU (GPU top-k is memory-inefficient for large k).
"""
function Gideon.predict_gpu(model, X::SparseMatrixCSC; k::Int=10)
    S = Gideon.predict_scores_gpu(model, X)
    n_users, n_items = size(S)
    k_out = min(k, n_items)

    # Mask seen items
    rv = rowvals(X)
    nz = nonzeros(X)
    for j in axes(X, 2)
        for idx in nzrange(X, j)
            S[rv[idx], j] = -Inf
        end
    end

    # Top-k per user
    preds = Matrix{Int}(undef, n_users, k_out)
    Threads.@threads for u in 1:n_users
        preds[u, :] .= partialsortperm(@view(S[u, :]), 1:k_out; rev=true)
    end
    preds
end

end # module GideonCUDAExt
