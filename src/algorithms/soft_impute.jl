# ──────────────────────────────────────────────────────────────────────────────
# SoftImpute — matrix completion via alternating least squares
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: Hastie, Mazumder, Lee, Zadeh (2014)
#   "Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares"
# ──────────────────────────────────────────────────────────────────────────────

"""
    SoftImpute{T} <: AbstractMatrixFactorization

Low-rank matrix completion via alternating least squares with nuclear-norm
regularization. Learns a factorization `U * Diagonal(d) * V'` from partially
observed entries.

Supports two modes:
- `:soft_impute` — corrects for observed entries at each iteration (default)
- `:svd` — power-iteration style (SoftSVD)

# Constructor
```julia
SoftImpute(; rank=10, λ=0.0, max_iter=100, convergence_tol=1e-3,
             target=:soft_impute, final_svd=true, verbose=true)
```

# Fields
- `rank::Int` — target rank for the low-rank approximation
- `λ::T` — nuclear-norm penalty (soft-threshold on singular values)
- `max_iter::Int` — maximum iterations
- `convergence_tol::T` — relative Frobenius norm change for early stopping
- `target::Symbol` — `:soft_impute` or `:svd`
- `final_svd::Bool` — re-factorize result via full SVD at end
- `U::Matrix{T}` — left singular vectors (m × rank) after fitting
- `d::Vector{T}` — singular values after fitting
- `V::Matrix{T}` — right singular vectors (n × rank) after fitting

# Example
```julia
using SparseArrays, Gideon

X = sprand(1000, 500, 0.05)
model = SoftImpute(rank=20, λ=1.0, max_iter=50)
fit!(model, X)
reconstruction = model.U * Diagonal(model.d) * model.V'
```
"""
mutable struct SoftImpute{T<:AbstractFloat} <: AbstractMatrixFactorization
    const rank::Int
    const λ::T
    const max_iter::Int
    const convergence_tol::T
    const target::Symbol
    const final_svd::Bool
    const verbose::Bool
    U::Matrix{T}
    d::Vector{T}
    V::Matrix{T}
    user_factors::Matrix{T}
    item_factors::Matrix{T}
    is_fitted::Bool
end

function SoftImpute(;
    rank::Int = 10,
    λ::Float64 = 0.0,
    max_iter::Int = 100,
    convergence_tol::Float64 = 1e-3,
    target::Symbol = :soft_impute,
    final_svd::Bool = true,
    verbose::Bool = true,
)
    rank >= 1 || throw(ArgumentError("rank must be ≥ 1, got $rank"))
    λ >= 0.0 || throw(ArgumentError("λ must be non-negative, got $λ"))
    target in (:soft_impute, :svd) || throw(ArgumentError("target must be :soft_impute or :svd, got :$target"))
    T = Float64
    SoftImpute{T}(rank, T(λ), max_iter, T(convergence_tol), target, final_svd, verbose,
                  Matrix{T}(undef,0,0), T[], Matrix{T}(undef,0,0),
                  Matrix{T}(undef,0,0), Matrix{T}(undef,0,0), false)
end

# ──────────────────────────────────────────────────────────────────────────────
# fit!
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit!(model::SoftImpute, X; rng, callbacks) -> model

Fit the SoftImpute model on sparse matrix `X`.
"""
function fit!(model::SoftImpute{T}, X::SparseMatrixCSC{Tv,Ti};
              rng::AbstractRNG = Random.default_rng(),
              callbacks::Vector{<:AbstractCallback} = AbstractCallback[]) where {T,Tv,Ti}
    m, n = size(X)
    k = min(model.rank, m, n)

    # Initialize with random orthonormal bases
    Q_u, _ = qr(randn(rng, T, m, k))
    U_cur = Matrix(Q_u)[:, 1:k]
    Q_v, _ = qr(randn(rng, T, n, k))
    V_cur = Matrix(Q_v)[:, 1:k]
    d_cur = ones(T, k)

    monitor = ConvergenceMonitor{T}(tol=T(model.convergence_tol), min_iter=2)

    for iter in 1:model.max_iter
        iter_start = time_ns()

        if model.target == :svd
            B = X * V_cur
            F_b = svd(B)
            kk = min(k, length(F_b.S))
            U_cur = F_b.U[:, 1:kk]
            d_cur = _soft_threshold.(F_b.S[1:kk], T(model.λ))

            A = X' * U_cur
            F_a = svd(A)
            kk2 = min(k, length(F_a.S))
            V_cur = F_a.U[:, 1:kk2]
            d_cur = _soft_threshold.(F_a.S[1:kk2], T(model.λ))
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
            d_cur = _soft_threshold.(F_b.S[1:kk], T(model.λ))

            UdVt_vals2 = _sparse_approx_values(X, U_cur .* d_cur', V_cur[:, 1:kk])
            X_corrected2 = copy(X)
            nonzeros(X_corrected2) .= nonzeros(X) .- Tv.(UdVt_vals2)

            A = X_corrected2' * U_cur .+ V_cur[:, 1:kk] .* d_cur'
            F_a = svd(A)
            kk2 = min(kk, length(F_a.S))
            V_cur = F_a.U[:, 1:kk2]
            d_cur = _soft_threshold.(F_a.S[1:kk2], T(model.λ))
            U_cur = U_cur[:, 1:kk2]
        end

        cur_frob = sum(abs2, d_cur)
        iter_seconds = (time_ns() - iter_start) / 1e9
        total_seconds = elapsed_seconds(monitor)

        if model.verbose
            log_iteration("SoftImpute", iter, model.max_iter, cur_frob,
                         iter_seconds, total_seconds;
                         extra="target=$(model.target)")
        end

        if record!(monitor, T(cur_frob))
            model.verbose && @info "[SoftImpute] converged at iteration $iter"
            break
        end

        if !isempty(callbacks)
            info = CallbackInfo(iter, Float64(cur_frob), total_seconds, model)
            run_callbacks(callbacks, info) && break
        end
    end

    if model.final_svd
        M_approx = U_cur .* d_cur' * V_cur'
        F_final = svd(M_approx)
        kk = min(k, length(F_final.S))
        model.U = F_final.U[:, 1:kk]
        model.d = F_final.S[1:kk]
        model.V = F_final.Vt[1:kk, :]'
    else
        model.U = U_cur
        model.d = d_cur
        model.V = V_cur
    end

    # Set user/item factors for compatibility with AbstractMatrixFactorization
    model.user_factors = (model.U .* sqrt.(model.d)')'  # rank × m
    model.item_factors = (model.V .* sqrt.(model.d)')'  # rank × n
    model.is_fitted = true
    model
end

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
