# ──────────────────────────────────────────────────────────────────────────────
# SoftImpute / SoftSVD — matrix completion via alternating least squares
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: Hastie, Mazumder, Lee, Zadeh (2014)
#   "Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares"
# ──────────────────────────────────────────────────────────────────────────────

"""
    SoftImputeResult{T}

Holds the SVD-like result from SoftImpute: `U`, `d`, `V` such that
the low-rank approximation is `U * Diagonal(d) * V'`.
"""
struct SoftImputeResult{T<:AbstractFloat}
    U::Matrix{T}
    d::Vector{T}
    V::Matrix{T}
end

"""
    soft_impute(X; rank, λ, max_iter, convergence_tol, final_svd, verbose) -> SoftImputeResult

Fit the SoftImpute algorithm on sparse matrix `X`.

# Example
```julia
using SparseArrays, Gideon

X = sprand(1000, 500, 0.05)
result = soft_impute(X; rank=20, λ=1.0, max_iter=50)
reconstruction = result.U * Diagonal(result.d) * result.V'
```
"""
function soft_impute(
    X::SparseMatrixCSC{Tv,Ti};
    rank::Int = 10,
    λ::Float64 = 0.0,
    max_iter::Int = 100,
    convergence_tol::Float64 = 1e-3,
    final_svd::Bool = true,
    verbose::Bool = true,
) where {Tv,Ti}
    _soft_als(X; rank, λ, max_iter, convergence_tol, final_svd, target=:soft_impute, verbose)
end

"""
    soft_svd(X; rank, λ, max_iter, convergence_tol, final_svd, verbose) -> SoftImputeResult

Fit the SoftSVD algorithm on sparse matrix `X` (power-iteration style).
"""
function soft_svd(
    X::SparseMatrixCSC{Tv,Ti};
    rank::Int = 10,
    λ::Float64 = 0.0,
    max_iter::Int = 100,
    convergence_tol::Float64 = 1e-3,
    final_svd::Bool = true,
    verbose::Bool = true,
) where {Tv,Ti}
    _soft_als(X; rank, λ, max_iter, convergence_tol, final_svd, target=:svd, verbose)
end

function _soft_als(
    X::SparseMatrixCSC{Tv,Ti};
    rank::Int,
    λ::Float64,
    max_iter::Int,
    convergence_tol::Float64,
    final_svd::Bool,
    target::Symbol,
    verbose::Bool,
) where {Tv,Ti}
    T = Float64
    m, n = size(X)
    k = min(rank, m, n)

    # Initialize with random orthonormal bases
    Q_u, _ = qr(randn(T, m, k))
    U_cur = Matrix(Q_u)[:, 1:k]
    Q_v, _ = qr(randn(T, n, k))
    V_cur = Matrix(Q_v)[:, 1:k]
    d_cur = ones(T, k)

    monitor = ConvergenceMonitor{T}(tol=T(convergence_tol), min_iter=2)

    for iter in 1:max_iter
        iter_start = time_ns()

        if target == :svd
            B = X * V_cur
            F_b = svd(B)
            kk = min(k, length(F_b.S))
            U_cur = F_b.U[:, 1:kk]
            d_cur = _soft_threshold.(F_b.S[1:kk], T(λ))

            A = X' * U_cur
            F_a = svd(A)
            kk2 = min(k, length(F_a.S))
            V_cur = F_a.U[:, 1:kk2]
            d_cur = _soft_threshold.(F_a.S[1:kk2], T(λ))
            U_cur = U_cur[:, 1:kk2]
        else
            # SoftImpute: correct for observed entries
            UdVt_vals = _sparse_approx_values(X, U_cur .* d_cur', V_cur)
            X_corrected = copy(X)
            nonzeros(X_corrected) .= nonzeros(X) .- Tv.(UdVt_vals)

            B = X_corrected * V_cur .+ U_cur .* d_cur'
            F_b = svd(B)
            kk = min(k, length(F_b.S))
            U_cur = F_b.U[:, 1:kk]
            d_cur = _soft_threshold.(F_b.S[1:kk], T(λ))

            UdVt_vals2 = _sparse_approx_values(X, U_cur .* d_cur', V_cur[:, 1:kk])
            X_corrected2 = copy(X)
            nonzeros(X_corrected2) .= nonzeros(X) .- Tv.(UdVt_vals2)

            A = X_corrected2' * U_cur .+ V_cur[:, 1:kk] .* d_cur'
            F_a = svd(A)
            kk2 = min(kk, length(F_a.S))
            V_cur = F_a.U[:, 1:kk2]
            d_cur = _soft_threshold.(F_a.S[1:kk2], T(λ))
            U_cur = U_cur[:, 1:kk2]
        end

        cur_frob = sum(abs2, d_cur)
        iter_seconds = (time_ns() - iter_start) / 1e9
        total_seconds = elapsed_seconds(monitor)

        if verbose
            log_iteration("SoftImpute", iter, max_iter, cur_frob,
                         iter_seconds, total_seconds;
                         extra="target=$target")
        end

        if record!(monitor, T(cur_frob))
            verbose && @info "[SoftImpute] converged at iteration $iter"
            break
        end
    end

    if final_svd
        M_approx = U_cur .* d_cur' * V_cur'
        F_final = svd(M_approx)
        kk = min(k, length(F_final.S))
        return SoftImputeResult{T}(
            F_final.U[:, 1:kk],
            F_final.S[1:kk],
            F_final.Vt[1:kk, :]',
        )
    end
    SoftImputeResult{T}(U_cur, d_cur, V_cur)
end

@inline _soft_threshold(x::T, λ::T) where {T} = max(x - λ, zero(T))

"""
    _sparse_approx_values(X, A, B)

Compute the values of `A * B'` at the non-zero positions of sparse matrix `X`.
"""
function _sparse_approx_values(X::SparseMatrixCSC{Tv,Ti}, A::Matrix{T}, B::Matrix{T}) where {Tv,Ti,T}
    k = size(A, 2)
    rv = rowvals(X)
    result = Vector{T}(undef, nnz(X))
    pos = 1
    @inbounds for col in axes(X, 2)
        for idx in nzrange(X, col)
            row = rv[idx]
            val = zero(T)
            for f in 1:k
                val += A[row, f] * B[col, f]
            end
            result[pos] = val
            pos += 1
        end
    end
    result
end
