# ──────────────────────────────────────────────────────────────────────────────
# Tables.jl integration — accept interaction data as (user, item, value) triplets
# ──────────────────────────────────────────────────────────────────────────────

"""
    interactions_to_sparse(table; user_col=:user, item_col=:item, value_col=:value,
                           n_users=nothing, n_items=nothing) -> SparseMatrixCSC

Convert any Tables.jl-compatible source (DataFrames, NamedTuples of vectors, CSV rows, etc.)
to a sparse user-item interaction matrix.

Columns `user_col` and `item_col` must contain integer indices (1-based).
The `value_col` column provides the interaction strength (defaults to 1.0 if missing).

# Arguments
- `table` — any iterable of rows with named fields (Tables.jl compatible)
- `user_col::Symbol` — name of the user ID column
- `item_col::Symbol` — name of the item ID column
- `value_col::Symbol` — name of the value/rating column (use `nothing` for implicit=1)
- `n_users::Union{Nothing,Int}` — number of users (auto-detected if nothing)
- `n_items::Union{Nothing,Int}` — number of items (auto-detected if nothing)

# Example
```julia
using Gideon, DataFrames

df = DataFrame(user=[1,1,2,3,3,3], item=[2,5,3,1,2,4], rating=[1.0,1.0,1.0,1.0,1.0,1.0])
X = interactions_to_sparse(df; user_col=:user, item_col=:item, value_col=:rating)
```
"""
function interactions_to_sparse(table;
                                user_col::Symbol = :user,
                                item_col::Symbol = :item,
                                value_col::Union{Symbol,Nothing} = :value,
                                n_users::Union{Nothing,Int} = nothing,
                                n_items::Union{Nothing,Int} = nothing)
    # Handle column tables (NamedTuple of vectors) directly
    if _is_column_table(table, user_col, item_col)
        users = Int.(getproperty(table, user_col))
        items = Int.(getproperty(table, item_col))
        vals = if value_col === nothing || !hasproperty(table, value_col)
            ones(Float64, length(users))
        else
            Float64.(getproperty(table, value_col))
        end
    else
        # Row iteration (Vector of NamedTuples, DataFrameRows, etc.)
        users = Int[]
        items = Int[]
        vals = Float64[]
        for row in table
            push!(users, Int(getfield_or_getindex(row, user_col)))
            push!(items, Int(getfield_or_getindex(row, item_col)))
            push!(vals, value_col === nothing ? 1.0 : Float64(getfield_or_getindex(row, value_col)))
        end
    end

    isempty(users) && error("Empty interaction table")

    nu = something(n_users, maximum(users))
    ni = something(n_items, maximum(items))

    # Validate indices
    all(u -> 1 <= u <= nu, users) || throw(ArgumentError("User indices must be in [1, $nu]"))
    all(i -> 1 <= i <= ni, items) || throw(ArgumentError("Item indices must be in [1, $ni]"))

    sparse(users, items, vals, nu, ni)
end

"""
    sparse_to_interactions(X::SparseMatrixCSC) -> NamedTuple

Convert a sparse matrix back to (user, item, value) triplet vectors.
Returns a NamedTuple with fields `:user`, `:item`, `:value`.

# Example
```julia
using Gideon, SparseArrays
X = sprand(10, 5, 0.3)
triplets = sparse_to_interactions(X)
# triplets.user, triplets.item, triplets.value
```
"""
function sparse_to_interactions(X::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    users = Int[]
    items = Int[]
    vals = Tv[]
    rv = rowvals(X)
    nz = nonzeros(X)
    for j in axes(X, 2)
        for idx in nzrange(X, j)
            push!(users, Int(rv[idx]))
            push!(items, j)
            push!(vals, nz[idx])
        end
    end
    (user=users, item=items, value=vals)
end

# ──────────────────────────────────────────────────────────────────────────────
# Helpers for Tables.jl-compatible row iteration (no Tables.jl dependency)
# ──────────────────────────────────────────────────────────────────────────────

# Detect column tables: a NamedTuple whose values are AbstractVectors
function _is_column_table(table, user_col::Symbol, item_col::Symbol)
    table isa NamedTuple || return false
    haskey(table, user_col) || return false
    haskey(table, item_col) || return false
    return getproperty(table, user_col) isa AbstractVector
end

# Access a field from a row (works for NamedTuples, DataFrameRows, etc.)
@inline function getfield_or_getindex(row, col::Symbol)
    return getproperty(row, col)
end
