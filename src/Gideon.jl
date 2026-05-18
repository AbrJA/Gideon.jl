module Gideon

using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using Random
using Logging
using Printf

# ── Core types & API ──
include("types.jl")
include("utils.jl")
include("sparse_utils.jl")
include("progress.jl")

# ── Algorithms ──
include("algorithms/wrmf.jl")
include("algorithms/ftrl.jl")
include("algorithms/fm.jl")
include("algorithms/glove.jl")
include("algorithms/lmf.jl")
include("algorithms/soft_impute.jl")

# ── Metrics ──
include("metrics/ranking.jl")

# ── Public API ──
export
    # Types
    AbstractSparseModel,
    AbstractMatrixFactorization,
    AbstractSparseRegression,
    ALSSolver, CHOLESKY, CONJUGATE_GRADIENT, NNLS,
    FeedbackType, IMPLICIT, EXPLICIT,
    Family, BINOMIAL, GAUSSIAN, POISSON,

    # Models
    WRMF,
    FTRL,
    FactorizationMachine,
    GloVe,
    LMF,
    SoftImputeResult,

    # Generic API
    fit!,
    transform,
    predict,
    partial_fit!,
    coef,

    # Convenience functions
    soft_impute,
    soft_svd,
    get_embeddings,

    # Metrics
    ap_at_k,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,

    # Sparse utilities
    to_csr,
    sparse_row_norms,
    sparse_col_nnz,
    sparse_row_nnz,
    dual_representation,

    # Helpers
    sigmoid,
    init_factors

end # module Gideon
