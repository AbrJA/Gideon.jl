module Gideon

using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using Random
using Printf
using Serialization
using PrecompileTools

# ── Core types & API ──
include("types.jl")
include("utils.jl")
include("sparse_utils.jl")
include("progress.jl")
include("callbacks.jl")
include("serialization.jl")

# ── Algorithms ──
include("algorithms/wrmf.jl")
include("algorithms/ials.jl")
include("algorithms/eals.jl")
include("algorithms/ftrl.jl")
include("algorithms/fm.jl")
include("algorithms/glove.jl")
include("algorithms/lmf.jl")
include("algorithms/bpr.jl")
include("algorithms/ease.jl")
include("algorithms/slim.jl")
include("algorithms/soft_impute.jl")

# ── Metrics & evaluation ──
include("metrics/ranking.jl")
include("crossval.jl")

# ── Tables.jl integration ──
include("tables.jl")

# ── Precompilation ──
include("precompile.jl")

# ── Public API ──
export
    # Types
    AbstractSparseModel,
    AbstractRecommender,
    AbstractMatrixFactorization,
    AbstractItemSimilarity,
    AbstractSparseRegression,
    ALSSolver, CHOLESKY, CONJUGATE_GRADIENT, NNLS,
    FeedbackType, IMPLICIT, EXPLICIT,
    Family, BINOMIAL, GAUSSIAN, POISSON,

    # Models
    WRMF,
    IALS,
    EALS,
    FTRL,
    FactorizationMachine,
    GloVe,
    LMF,
    BPR,
    EASE,
    SLIM,
    SoftImputeResult,

    # Generic API
    fit!,
    transform,
    recommend,
    score,
    predict,
    predict_scores,
    predict_scores_gpu,
    predict_gpu,
    fit_gpu!,
    partial_fit!,
    coef,
    similar_items,
    similar_users,

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

    # Cross-validation & search
    temporal_split,
    cv_evaluate,
    grid_search,
    random_search,

    # Callbacks
    AbstractCallback,
    CallbackInfo,
    on_epoch_end,
    on_train_begin,
    on_train_end,
    EarlyStoppingCallback,
    LossHistoryCallback,
    CheckpointCallback,
    LearningRateScheduler,
    run_callbacks,
    run_callbacks_train_begin,
    run_callbacks_train_end,

    # Serialization
    save_model,
    load_model,

    # Sparse utilities
    to_csr,
    sparse_row_norms,
    sparse_col_nnz,

    # Tables.jl integration
    interactions_to_sparse,
    sparse_to_interactions,
    sparse_row_nnz,
    dual_representation,

    # Helpers
    sigmoid,
    init_factors

# ── GPU stubs (implemented by ext/GideonCUDAExt.jl when CUDA is loaded) ──
function fit_gpu! end
function predict_gpu end
function predict_scores_gpu end

end # module Gideon
