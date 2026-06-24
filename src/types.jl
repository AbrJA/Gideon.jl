# ──────────────────────────────────────────────────────────────────────────────
# Abstract type hierarchy
# ──────────────────────────────────────────────────────────────────────────────

"""
    AbstractSparseModel

Root abstract type for all Gideon models that operate on sparse matrices.
"""
abstract type AbstractSparseModel end

"""
    AbstractRecommender <: AbstractSparseModel

Abstract type for recommendation models that produce top-k item lists.
Models inheriting from this type must implement `recommend` and `score`.
"""
abstract type AbstractRecommender <: AbstractSparseModel end

"""
    AbstractMatrixFactorization <: AbstractRecommender

Abstract type for matrix factorization models.
Also provides `embeddings`, `similar_items`, and `similar_users`.
"""
abstract type AbstractMatrixFactorization <: AbstractRecommender end

"""
    AbstractItemSimilarity <: AbstractRecommender

Abstract type for item-similarity (neighborhood) models.
"""
abstract type AbstractItemSimilarity <: AbstractRecommender end

"""
    AbstractSparseRegression <: AbstractSparseModel

Abstract type for sparse regression models.
These implement `predict` (regression output), not `recommend`.
"""
abstract type AbstractSparseRegression <: AbstractSparseModel end

# ──────────────────────────────────────────────────────────────────────────────
# Solver enum
# ──────────────────────────────────────────────────────────────────────────────

"""
    ALSSolver

Enum for ALS solver type: `CHOLESKY`, `CONJUGATE_GRADIENT`, or `NNLS`.
"""
@enum ALSSolver begin
    CHOLESKY
    CONJUGATE_GRADIENT
    NNLS
end

# ──────────────────────────────────────────────────────────────────────────────
# Feedback enum
# ──────────────────────────────────────────────────────────────────────────────

"""
    FeedbackType

Enum for feedback type: `IMPLICIT` or `EXPLICIT`.
"""
@enum FeedbackType begin
    IMPLICIT
    EXPLICIT
end

# ──────────────────────────────────────────────────────────────────────────────
# Family enum (for GLM link functions)
# ──────────────────────────────────────────────────────────────────────────────

"""
    Family

Enum for GLM family: `BINOMIAL` (logistic), `GAUSSIAN` (identity), or `POISSON` (log).
"""
@enum Family begin
    BINOMIAL
    GAUSSIAN
    POISSON
end

# ──────────────────────────────────────────────────────────────────────────────
# Negative sampling enum (for BayesianPersonalizedRanking)
# ──────────────────────────────────────────────────────────────────────────────

"""
    NegativeSampling

Enum for negative sampling strategy: `UNIFORM`, `POPULAR`, or `DYNAMIC`.
"""
@enum NegativeSampling begin
    UNIFORM
    POPULAR
    DYNAMIC
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
    recommend(model::AbstractRecommender, X; k=10)

Return top-k item indices per user, excluding already-interacted items.
Returns a `Matrix{Int}` of shape (n_users, k).
"""
function recommend end

"""
    score(model::AbstractRecommender, X)
    score(model::AbstractRecommender, user_indices, item_indices)

Return raw prediction scores. The full-matrix variant returns a dense `Matrix{T}`
(n_users × n_items). The pairwise variant returns a `Vector{T}` for specific
(user, item) pairs.
"""
function score end

"""
    predict(model::AbstractSparseRegression, X)

Generate regression predictions from a fitted model. Returns `Vector{T}`.
"""
function predict end

"""
    update!(model, X, y; kwargs...)

Run a single epoch of online/incremental learning. For streaming models
(OnlineRegressor, FactorizationMachine, ElementwiseALS).
"""
function update! end

"""
    embeddings(model::AbstractMatrixFactorization)

Return the embedding matrix for a fitted model.
"""
function embeddings end

"""
    similar_items(model::AbstractMatrixFactorization, item_id; k=10)

Find the k most similar items to `item_id` based on embedding cosine similarity.
Returns `(ids::Vector{Int}, scores::Vector{T})`.
"""
function similar_items end

"""
    similar_users(model::AbstractMatrixFactorization, user_id; k=10)

Find the k most similar users to `user_id` based on embedding cosine similarity.
Returns `(ids::Vector{Int}, scores::Vector{T})`.
"""
function similar_users end


