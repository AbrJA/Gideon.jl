# validation/validate_py.jl
# Optional validation against Python implicit implementations.

using Gideon, SparseArrays, LinearAlgebra, Random
using Test

const PY_FIXTURE_DIR = get(ENV, "GIDEON_PY_FIXTURE_DIR", "/tmp/gideon_fixtures/python")

function _read_matrix(path::String)
    lines = readlines(path)
    rows = [parse.(Float64, split(strip(l), ','))
            for l in lines if !isempty(strip(l))]
    isempty(rows) && return Matrix{Float64}(undef, 0, 0)
    reduce(vcat, [r' for r in rows])
end

function _load_sparse(triplet::String, dims::String)
    lines = readlines(triplet)
    rv = Int[]
    cv = Int[]
    vv = Float64[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        p = split(strip(line), ',')
        push!(rv, parse(Int, p[1]))
        push!(cv, parse(Int, p[2]))
        push!(vv, parse(Float64, p[3]))
    end
    dlines = readlines(dims)
    hdr = split(strip(dlines[1]), ',')
    dvals = split(strip(dlines[2]), ',')
    nr = parse(Int, dvals[findfirst(==("nr"), hdr)])
    nc = parse(Int, dvals[findfirst(==("nc"), hdr)])
    sparse(rv, cv, vv, nr, nc)
end

function _cor(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    x64 = Float64.(x)
    y64 = Float64.(y)
    mx = sum(x64) / length(x64)
    my = sum(y64) / length(y64)
    dx = x64 .- mx
    dy = y64 .- my
    dot(dx, dy) / (sqrt(dot(dx, dx) * dot(dy, dy)) + 1e-15)
end

function _row_topk(scores::AbstractMatrix{<:Real}, k::Int)
    n_users = size(scores, 1)
    out = Matrix{Int}(undef, n_users, k)
    for u in 1:n_users
        out[u, :] = partialsortperm(@view(scores[u, :]), 1:k; rev=true)
    end
    out
end

function _mean_topk_overlap(a::Matrix{Int}, b::Matrix{Int})
    n_users = size(a, 1)
    k = size(a, 2)
    s = 0.0
    for u in 1:n_users
        s += length(intersect(@view(a[u, :]), @view(b[u, :]))) / k
    end
    s / n_users
end

@testset "Python Reference Comparison" begin
    if !isdir(PY_FIXTURE_DIR)
        @warn """
        Python fixtures not found at: $PY_FIXTURE_DIR
        Generate fixtures with:
          python3 validation/fixtures_py.py
        """
        return
    end

    x_path = joinpath(PY_FIXTURE_DIR, "X_small.csv")
    d_path = joinpath(PY_FIXTURE_DIR, "X_small_dims.csv")
    als_path = joinpath(PY_FIXTURE_DIR, "py_als_scores.csv")
    bpr_path = joinpath(PY_FIXTURE_DIR, "py_bpr_scores.csv")

    if !(isfile(x_path) && isfile(d_path) && isfile(als_path) && isfile(bpr_path))
        @warn """
        Python fixture files are incomplete at: $PY_FIXTURE_DIR
        Expected: X_small.csv, X_small_dims.csv, py_als_scores.csv, py_bpr_scores.csv
        Regenerate with:
          python3 validation/fixtures_py.py
        """
        return
    end

    X = _load_sparse(x_path, d_path)
    py_als = _read_matrix(als_path)
    py_bpr = _read_matrix(bpr_path)

    @test size(py_als) == size(py_bpr) == size(X)

    rank = 16

    @testset "WMF (Julia) vs ALS (Python)" begin
        m = WMF(rank=rank, λ=0.1, α=40.0, max_iter=20,
                solver=CholeskySolver(), feedback=IMPLICIT, verbose=false)
        fit!(m, X; rng=MersenneTwister(42))
        jl_scores = Matrix(transpose(m.user_factors) * m.item_factors)

        c = _cor(vec(jl_scores), vec(py_als))
        k = 10
        n_users_eval = min(25, size(jl_scores, 1))
        jl_top = _row_topk(jl_scores[1:n_users_eval, :], k)
        py_top = _row_topk(py_als[1:n_users_eval, :], k)
        overlap = _mean_topk_overlap(jl_top, py_top)

        min_cor = parse(Float64, get(ENV, "GIDEON_PY_ALS_MIN_COR", "0.45"))
        min_ov = parse(Float64, get(ENV, "GIDEON_PY_ALS_MIN_OVERLAP", "0.20"))

        @test isfinite(c)
        @test c >= min_cor
        @test overlap >= min_ov
        println("  ALS score correlation: $c")
        println("  ALS top-$k overlap: $overlap")
    end

    @testset "BPR (Julia) vs BPR (Python)" begin
        m = BPR(rank=rank, max_iter=40, verbose=false)
        fit!(m, X; rng=MersenneTwister(42))
        jl_scores = Matrix(transpose(m.user_factors) * m.item_factors)

        c = _cor(vec(jl_scores), vec(py_bpr))
        k = 10
        n_users_eval = min(25, size(jl_scores, 1))
        jl_top = _row_topk(jl_scores[1:n_users_eval, :], k)
        py_top = _row_topk(py_bpr[1:n_users_eval, :], k)
        overlap = _mean_topk_overlap(jl_top, py_top)

        min_cor = parse(Float64, get(ENV, "GIDEON_PY_BPR_MIN_COR", "0.20"))
        min_ov = parse(Float64, get(ENV, "GIDEON_PY_BPR_MIN_OVERLAP", "0.10"))

        @test isfinite(c)
        @test c >= min_cor
        @test overlap >= min_ov
        println("  BPR score correlation: $c")
        println("  BPR top-$k overlap: $overlap")
    end
end
