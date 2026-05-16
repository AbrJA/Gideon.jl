# ──────────────────────────────────────────────────────────────────────────────
# Ranking evaluation metrics — MAP@K, NDCG@K, Precision@K
# ──────────────────────────────────────────────────────────────────────────────
#
# All metrics follow the convention:
#   predictions : Matrix{Int} of shape (n_users, K)  — predicted item indices
#   actual      : SparseMatrixCSC (n_users × n_items) — non-zero = relevant
# ──────────────────────────────────────────────────────────────────────────────

using SparseArrays

# ──────────────── Average Precision @ K ────────────────

"""
    ap_at_k(predictions::AbstractMatrix{Int}, actual::SparseMatrixCSC; k=size(predictions,2))

Compute Average Precision @ K for each user.
Returns a vector of length `n_users`.
"""
function ap_at_k(predictions::AbstractMatrix{<:Integer},
                 actual::SparseMatrixCSC;
                 k::Int = size(predictions, 2))
    n_users = size(predictions, 1)
    @assert n_users == size(actual, 1) "Row count mismatch"
    result = Vector{Float64}(undef, n_users)
    @inbounds for u in 1:n_users
        result[u] = _ap_single(predictions, actual, u, k)
    end
    result
end

"""
    map_at_k(predictions, actual; k)

Mean Average Precision @ K (macro-averaged over all users).
"""
function map_at_k(predictions::AbstractMatrix{<:Integer},
                  actual::SparseMatrixCSC;
                  k::Int = size(predictions, 2))
    aps = ap_at_k(predictions, actual; k)
    sum(aps) / length(aps)
end

function _ap_single(predictions::AbstractMatrix{<:Integer},
                    actual::SparseMatrixCSC, u::Int, k::Int)
    relevant = Set(_relevant_items(actual, u))
    isempty(relevant) && return 0.0

    k_eff = min(k, size(predictions, 2))
    hits = 0
    cum_precision = 0.0
    @inbounds for pos in 1:k_eff
        item = predictions[u, pos]
        if item in relevant
            hits += 1
            cum_precision += hits / pos
        end
    end
    cum_precision / min(k_eff, length(relevant))
end

# ──────────────── NDCG @ K ────────────────

"""
    ndcg_at_k(predictions::AbstractMatrix{Int}, actual::SparseMatrixCSC; k)

Normalized Discounted Cumulative Gain @ K for each user.
Non-zero values in `actual` are treated as relevance scores.
"""
function ndcg_at_k(predictions::AbstractMatrix{<:Integer},
                   actual::SparseMatrixCSC;
                   k::Int = size(predictions, 2))
    n_users = size(predictions, 1)
    @assert n_users == size(actual, 1)
    result = Vector{Float64}(undef, n_users)
    @inbounds for u in 1:n_users
        result[u] = _ndcg_single(predictions, actual, u, k)
    end
    result
end

function _ndcg_single(predictions::AbstractMatrix{<:Integer},
                      actual::SparseMatrixCSC, u::Int, k::Int)
    items, rels = _relevant_items_with_scores(actual, u)
    isempty(items) && return 0.0

    item_rel = Dict(zip(items, rels))
    k_eff = min(k, size(predictions, 2))

    # DCG
    dcg = 0.0
    @inbounds for pos in 1:k_eff
        item = predictions[u, pos]
        rel = get(item_rel, item, 0.0)
        dcg += rel / log2(pos + 1)
    end

    # IDCG
    sorted_rels = sort(rels; rev=true)
    idcg = 0.0
    kk = min(k_eff, length(sorted_rels))
    @inbounds for pos in 1:kk
        idcg += sorted_rels[pos] / log2(pos + 1)
    end

    idcg ≈ 0.0 ? 0.0 : dcg / idcg
end

# ──────────────── Precision @ K ────────────────

"""
    precision_at_k(predictions::AbstractMatrix{Int}, actual::SparseMatrixCSC; k)

Precision @ K for each user.
"""
function precision_at_k(predictions::AbstractMatrix{<:Integer},
                        actual::SparseMatrixCSC;
                        k::Int = size(predictions, 2))
    n_users = size(predictions, 1)
    @assert n_users == size(actual, 1)
    result = Vector{Float64}(undef, n_users)
    @inbounds for u in 1:n_users
        result[u] = _precision_single(predictions, actual, u, k)
    end
    result
end

function _precision_single(predictions::AbstractMatrix{<:Integer},
                           actual::SparseMatrixCSC, u::Int, k::Int)
    relevant = Set(_relevant_items(actual, u))
    isempty(relevant) && return 0.0

    k_eff = min(k, size(predictions, 2))
    hits = 0
    @inbounds for pos in 1:k_eff
        if predictions[u, pos] in relevant
            hits += 1
        end
    end
    hits / k_eff
end

# ──────────────── Recall @ K ────────────────

"""
    recall_at_k(predictions::AbstractMatrix{Int}, actual::SparseMatrixCSC; k)

Recall @ K for each user.
"""
function recall_at_k(predictions::AbstractMatrix{<:Integer},
                     actual::SparseMatrixCSC;
                     k::Int = size(predictions, 2))
    n_users = size(predictions, 1)
    @assert n_users == size(actual, 1)
    result = Vector{Float64}(undef, n_users)
    @inbounds for u in 1:n_users
        relevant = Set(_relevant_items(actual, u))
        if isempty(relevant)
            result[u] = 0.0
            continue
        end
        k_eff = min(k, size(predictions, 2))
        hits = 0
        @inbounds for pos in 1:k_eff
            if predictions[u, pos] in relevant
                hits += 1
            end
        end
        result[u] = hits / length(relevant)
    end
    result
end

# ──────────────── Helpers: extract relevant items from sparse row ────────────────

function _relevant_items(actual::SparseMatrixCSC, u::Int)
    # For CSC, row `u` entries are scattered across columns.
    # Collect column indices where row `u` has a non-zero.
    items = Int[]
    rv = rowvals(actual)
    @inbounds for col in axes(actual, 2)
        for idx in nzrange(actual, col)
            if rv[idx] == u
                push!(items, col)
            end
        end
    end
    items
end

function _relevant_items_with_scores(actual::SparseMatrixCSC, u::Int)
    items = Int[]
    scores = Float64[]
    rv = rowvals(actual)
    nz = nonzeros(actual)
    @inbounds for col in axes(actual, 2)
        for idx in nzrange(actual, col)
            if rv[idx] == u
                push!(items, col)
                push!(scores, Float64(nz[idx]))
            end
        end
    end
    (items, scores)
end
