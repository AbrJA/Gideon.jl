# ──────────────────────────────────────────────────────────────────────────────
# Abstract type hierarchy
# ──────────────────────────────────────────────────────────────────────────────

"""
    AbstractSparseModel

Root abstract type for all Gideon models that operate on sparse matrices.
"""
abstract type AbstractSparseModel end

"""
    AbstractMatrixFactorization <: AbstractSparseModel

Abstract type for matrix factorization models (WRMF, GloVe, SoftImpute, LMF, etc.).
"""
abstract type AbstractMatrixFactorization <: AbstractSparseModel end

"""
    AbstractSparseRegression <: AbstractSparseModel

Abstract type for sparse regression models (FTRL, Factorization Machines, etc.).
"""
abstract type AbstractSparseRegression <: AbstractSparseModel end

# ──────────────────────────────────────────────────────────────────────────────
# Solver enum
# ──────────────────────────────────────────────────────────────────────────────

@enum ALSSolver begin
    CHOLESKY
    CONJUGATE_GRADIENT
    NNLS
end

# ──────────────────────────────────────────────────────────────────────────────
# Feedback enum
# ──────────────────────────────────────────────────────────────────────────────

@enum FeedbackType begin
    IMPLICIT
    EXPLICIT
end

# ──────────────────────────────────────────────────────────────────────────────
# Generic API — every model must implement these
# ──────────────────────────────────────────────────────────────────────────────

"""
    fit!(model::AbstractSparseModel, X; kwargs...)

Fit `model` in-place on sparse matrix `X`.
"""
function fit! end

"""
    transform(model::AbstractMatrixFactorization, X)

Return user embeddings for a fitted matrix factorization model.
"""
function transform end

"""
    predict(model::AbstractSparseModel, X; kwargs...)

Generate predictions from a fitted model.
"""
function predict end
