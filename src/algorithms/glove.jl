# ──────────────────────────────────────────────────────────────────────────────
# GloVe — Global Vectors for co-occurrence matrix factorization
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: Pennington, Socher, Manning (2014)
#   "GloVe: Global Vectors for Word Representation"
#
# Loss:
#   L = Σ_{i,j} f(X_{ij}) (wᵢᵀ w̃ⱼ + bᵢ + b̃ⱼ - log X_{ij})²
#
# where f(x) = (x/x_max)^α if x < x_max, else 1.
# ──────────────────────────────────────────────────────────────────────────────

using SparseArrays, LinearAlgebra, Random, LoopVectorization, Dates

"""
    GloVe{T} <: AbstractMatrixFactorization

GloVe matrix factorization with AdaGrad-based SGD.
"""
mutable struct GloVe{T<:AbstractFloat} <: AbstractMatrixFactorization
    rank::Int
    x_max::T
    learning_rate::T
    α::T
    λ::T
    shuffle::Bool
    verbose::Bool
    # Embeddings (rank × n)
    W_main::Matrix{T}
    W_ctx::Matrix{T}
    b_main::Vector{T}
    b_ctx::Vector{T}
    # AdaGrad accumulators
    grad_W_main::Matrix{T}
    grad_W_ctx::Matrix{T}
    grad_b_main::Vector{T}
    grad_b_ctx::Vector{T}
    cost_history::Vector{T}
    is_fitted::Bool
end

function GloVe(;
    rank::Int = 50,
    x_max::Float64 = 100.0,
    learning_rate::Float64 = 0.15,
    α::Float64 = 0.75,
    λ::Float64 = 0.0,
    shuffle::Bool = false,
    verbose::Bool = true,
)
    T = Float64
    GloVe{T}(
        rank, x_max, learning_rate, α, λ, shuffle, verbose,
        Matrix{T}(undef,0,0), Matrix{T}(undef,0,0),
        T[], T[],
        Matrix{T}(undef,0,0), Matrix{T}(undef,0,0),
        T[], T[],
        T[], false,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# fit! — iterative SGD on COO representation of the co-occurrence matrix
# ──────────────────────────────────────────────────────────────────────────────

function fit!(model::GloVe{T}, X::SparseMatrixCSC{Tv,Ti};
              n_iter::Int = 10,
              convergence_tol::Float64 = -1.0,
              rng::AbstractRNG = Random.default_rng()) where {T,Tv,Ti}
    n = size(X, 1)
    @assert size(X, 1) == size(X, 2) "GloVe requires a square co-occurrence matrix"
    @assert all(x -> x > 0, nonzeros(X)) "All co-occurrence values must be positive"

    k = model.rank

    # Initialize embeddings
    model.W_main    = (rand(rng, T, k, n) .- T(0.5))
    model.W_ctx     = (rand(rng, T, k, n) .- T(0.5))
    model.b_main    = (rand(rng, T, n) .- T(0.5))
    model.b_ctx     = (rand(rng, T, n) .- T(0.5))
    model.grad_W_main = ones(T, k, n)
    model.grad_W_ctx  = ones(T, k, n)
    model.grad_b_main = ones(T, n)
    model.grad_b_ctx  = ones(T, n)
    model.cost_history = T[]

    # Extract COO triplets
    rows, cols, vals = findnz(X)
    nnz_count = length(rows)
    train_start = now()

    for iter in 1:n_iter
        iter_start = now()
        order = model.shuffle ? randperm(rng, nnz_count) : (1:nnz_count)
        epoch_cost = _glove_epoch!(model, rows, cols, vals, order)

        if isnan(epoch_cost)
            error("GloVe: cost became NaN — try a smaller learning_rate")
        end

        avg_cost = epoch_cost / nnz_count
        push!(model.cost_history, avg_cost)
        iter_seconds = Dates.value(now() - iter_start) / 1000.0
        total_seconds = Dates.value(now() - train_start) / 1000.0
        if model.verbose
            @info "GloVe iteration" iter=iter avg_cost=avg_cost iter_seconds=iter_seconds total_seconds=total_seconds
        end
        @debug "GloVe iter=$iter  cost=$avg_cost"

        if iter > 1 && convergence_tol > 0
            improvement = model.cost_history[iter-1] / model.cost_history[iter] - 1
            if improvement < convergence_tol
                @debug "GloVe converged at iter=$iter"
                break
            end
        end
    end
    model.is_fitted = true
    model
end

function _glove_epoch!(model::GloVe{T}, rows, cols, vals, order) where {T}
    k  = model.rank
    lr = model.learning_rate
    x_max = model.x_max
    α  = model.α
    λ  = model.λ

    W  = model.W_main
    Wc = model.W_ctx
    b  = model.b_main
    bc = model.b_ctx
    gW  = model.grad_W_main
    gWc = model.grad_W_ctx
    gb  = model.grad_b_main
    gbc = model.grad_b_ctx

    # Thread-local cost accumulators — maxthreadid() covers interactive threads.
    nt = Threads.maxthreadid()
    local_costs = zeros(T, nt)

    # Hogwild parallel SGD — benign races on independent rows/cols, same as R rsparse
    Base.Threads.@threads :static for idx in order
        tid  = Threads.threadid()
        i = rows[idx]
        j = cols[idx]
        x_ij = T(vals[idx])

        # Weighting function
        weight = x_ij < x_max ? (x_ij / x_max)^α : one(T)

        # Inner product + biases
        diff = b[i] + bc[j] - log(x_ij)
        @inbounds for f in 1:k
            diff += W[f, i] * Wc[f, j]
        end

        local_costs[tid] += weight * diff^2
        grad_common = T(2) * weight * diff

        # AdaGrad update — Hogwild writes (no locks)
        # SIMD-vectorized gradient computation and weight updates
        @inbounds @simd for f in 1:k
            g_main = grad_common * Wc[f, j] + λ * W[f, i]
            g_ctx  = grad_common * W[f, i]  + λ * Wc[f, j]
            gW[f, i]  += g_main * g_main
            gWc[f, j] += g_ctx * g_ctx
            W[f, i]  -= lr * g_main / sqrt(gW[f, i])
            Wc[f, j] -= lr * g_ctx  / sqrt(gWc[f, j])
        end

        g_bi = grad_common
        g_bj = grad_common
        @inbounds begin
            gb[i]  += g_bi^2
            gbc[j] += g_bj^2
            b[i]  -= lr * g_bi / sqrt(gb[i])
            bc[j] -= lr * g_bj / sqrt(gbc[j])
        end
    end
    sum(local_costs)
end

"""
    get_embeddings(model::GloVe)

Return the combined word embeddings `W_main + W_ctx` (each column is an embedding).
"""
function get_embeddings(model::GloVe{T}) where {T}
    model.is_fitted || error("Model not fitted")
    model.W_main .+ model.W_ctx
end
