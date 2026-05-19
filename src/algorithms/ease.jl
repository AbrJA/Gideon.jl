# ──────────────────────────────────────────────────────────────────────────────
# EASE — Embarrassingly Shallow Autoencoders for Sparse Data
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: Harald Steck (2019)
#   "Embarrassingly Shallow Autoencoders for Sparse Data" (WWW 2019)
#   arXiv:1905.03375
#
# EASE learns an item-item weight matrix B by solving:
#   min_B ‖X - XB‖²_F + λ‖B‖²_F  subject to diag(B) = 0
#
# Closed-form solution:
#   P = (XᵀX + λI)⁻¹
#   B = I - P · diag(1/diag(P))
#
# This is a linear autoencoder that achieves state-of-the-art results on
# implicit feedback benchmarks, often outperforming deep models.
# ──────────────────────────────────────────────────────────────────────────────

"""
    EASE{T} <: AbstractSparseModel

Embarrassingly Shallow Autoencoders (EASE^R) for collaborative filtering.

A closed-form linear model that learns an item-item similarity matrix B
with the constraint that diag(B) = 0 (items cannot recommend themselves).
Despite its simplicity, EASE consistently outperforms deep models on
standard benchmarks.

# Constructor
```julia
EASE(; λ=500.0)
```

# Fields
- `λ::T` — L2 regularization (higher = more smoothing, typical range: 100-1000)
- `B::Matrix{T}` — item-item weight matrix (n_items × n_items) after fitting

# Example
```julia
using SparseArrays, Gideon
X = sprand(1000, 500, 0.02)  # users × items
model = EASE(λ=200.0)
fit!(model, X)
preds = predict(model, X; k=10)
```
"""
mutable struct EASE{T<:AbstractFloat} <: AbstractSparseModel
    λ::T
    verbose::Bool
    B::Matrix{T}
    is_fitted::Bool
end

function EASE(; λ::Float64=500.0, verbose::Bool=true)
    @assert λ > 0.0 "λ must be positive"
    EASE{Float64}(λ, verbose, Matrix{Float64}(undef, 0, 0), false)
end

# ──────────────────────────────────────────────────────────────────────────────
# fit!
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit!(model::EASE, X) -> model

Compute the closed-form EASE solution on interaction matrix `X` (users × items).

Complexity: O(n_items² × n_users) for XᵀX, then O(n_items³) for the inverse.
Memory: O(n_items²) for the weight matrix B.
"""
function fit!(model::EASE{T}, X::SparseMatrixCSC{Tv,Ti};
              kwargs...) where {T,Tv,Ti}
    n_users, n_items = size(X)

    model.verbose && @info "[EASE] Computing Gram matrix ($(n_items) items)..."

    # G = XᵀX + λI
    G = Matrix{T}(X' * X)
    @inbounds for i in 1:n_items
        G[i, i] += model.λ
    end

    model.verbose && @info "[EASE] Computing inverse via Cholesky ($(n_items)×$(n_items))..."

    # Use Cholesky factorization for numerical stability (G is SPD)
    C = cholesky(Symmetric(G))
    P = inv(C)

    # B = I - P · diag(1/diag(P))
    # Equivalent to: B_ij = -P_ij / P_jj for i≠j, B_ii = 0
    B = Matrix{T}(undef, n_items, n_items)
    @inbounds for j in 1:n_items
        inv_pjj = one(T) / P[j, j]
        for i in 1:n_items
            B[i, j] = -P[i, j] * inv_pjj
        end
        B[j, j] = zero(T)
    end

    model.B = B
    model.is_fitted = true

    model.verbose && @info "[EASE] Fitted. B matrix: $(n_items)×$(n_items)"
    model
end

# ──────────────────────────────────────────────────────────────────────────────
# predict
# ──────────────────────────────────────────────────────────────────────────────

"""
    predict(model::EASE, X; k=10) -> Matrix{Int}

Return top-k item indices per user. Scores are computed as X * B.
Already-interacted items are excluded.
"""
function predict(model::EASE{T}, X::SparseMatrixCSC; k::Int=10) where {T}
    model.is_fitted || error("Model not fitted")
    n_users = size(X, 1)
    n_items = size(model.B, 1)
    k_out = min(k, n_items)

    # Compute full score matrix via efficient sparse × dense
    S = Matrix{T}(X * model.B)
    preds = Matrix{Int}(undef, n_users, k_out)
    X_csr = to_csr(X)

    Threads.@threads for u in 1:n_users
        scores = @view S[u, :]

        # Mask seen items using CSR row access
        @inbounds for idx in nzrange(X_csr, u)
            j = Int(X_csr.colval[idx])
            S[u, j] = T(-Inf)
        end

        topk = partialsortperm(Vector(scores), 1:k_out; rev=true)
        preds[u, :] .= topk
    end
    preds
end

"""
    predict_scores(model::EASE, X) -> Matrix{T}

Return the full score matrix S = X * B (dense, n_users × n_items).
"""
function predict_scores(model::EASE{T}, X::SparseMatrixCSC) where {T}
    model.is_fitted || error("Model not fitted")
    Matrix{T}(X * model.B)
end
