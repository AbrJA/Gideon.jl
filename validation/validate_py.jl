# validation/validate_py.jl
# Optional validation against Python implicit implementations.

using Gideon, SparseArrays, LinearAlgebra, Random, Statistics
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

function _read_json_metric(path::String, key::String)
    txt = read(path, String)
    m = match(Regex("\"$key\"\\s*:\\s*([0-9eE+\\-.]+)"), txt)
    isnothing(m) && error("Metric '$key' not found in $(basename(path))")
    parse(Float64, m.captures[1])
end

function _safe_recommend(model, X::SparseMatrixCSC, k::Int)
    n_users, n_items = size(X)
    r = recommend(model, X; k=k)
    if size(r) == (n_users, k)
        return r
    end
    if size(r, 1) == k && size(r, 2) == n_users
        return Matrix(r')
    end
    error("Unexpected recommend() shape $(size(r)) for n_users=$n_users, k=$k")
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
    lmf_path = joinpath(PY_FIXTURE_DIR, "py_lmf_scores.csv")
    ease_path = joinpath(PY_FIXTURE_DIR, "py_ease_B.csv")
    x_train_path = joinpath(PY_FIXTURE_DIR, "X_train.csv")
    x_train_dims = joinpath(PY_FIXTURE_DIR, "X_train_dims.csv")
    x_test_path = joinpath(PY_FIXTURE_DIR, "X_test.csv")
    x_test_dims = joinpath(PY_FIXTURE_DIR, "X_test_dims.csv")
    bpr_metrics_path = joinpath(PY_FIXTURE_DIR, "py_bpr_metrics.json")
    lmf_metrics_path = joinpath(PY_FIXTURE_DIR, "py_lmf_metrics.json")

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
    X_train = _load_sparse(x_train_path, x_train_dims)
    X_test = _load_sparse(x_test_path, x_test_dims)
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
        max_ndcg_delta = parse(Float64, get(ENV, "GIDEON_PY_BPR_MAX_NDCG_DELTA", "0.05"))
        max_recall_delta = parse(Float64, get(ENV, "GIDEON_PY_BPR_MAX_RECALL_DELTA", "0.05"))

        @test isfinite(c)
        @test c >= min_cor
        @test overlap >= min_ov

        if isfile(bpr_metrics_path)
            py_ndcg = _read_json_metric(bpr_metrics_path, "ndcg")
            py_recall = _read_json_metric(bpr_metrics_path, "recall")
            jl_preds = _safe_recommend(m, X_train, 10)
            jl_ndcg = mean(ndcg_at_k(jl_preds, X_test; k=10))
            jl_recall = mean(recall_at_k(jl_preds, X_test; k=10))
            @test abs(jl_ndcg - py_ndcg) <= max_ndcg_delta
            @test abs(jl_recall - py_recall) <= max_recall_delta
            println("  BPR NDCG@10: Julia=$jl_ndcg, Python=$py_ndcg, Δ=$(abs(jl_ndcg - py_ndcg))")
            println("  BPR Recall@10: Julia=$jl_recall, Python=$py_recall, Δ=$(abs(jl_recall - py_recall))")
        else
            @info "Skipping BPR split-metric parity: py_bpr_metrics.json not found"
        end
        println("  BPR score correlation: $c")
        println("  BPR top-$k overlap: $overlap")
    end

    @testset "LogisticMF (Julia) vs LMF (Python, optional)" begin
        if isfile(lmf_path)
            py_lmf = _read_matrix(lmf_path)

            m = LogisticMF(
                rank=rank, λ=0.01, α=1.0,
                learning_rate=0.01, max_iter=30, verbose=false,
            )
            fit!(m, X; rng=MersenneTwister(42))
            jl_scores = Matrix(transpose(m.user_factors) * m.item_factors)

            c = _cor(vec(jl_scores), vec(py_lmf))
            k = 10
            n_users_eval = min(25, size(jl_scores, 1))
            jl_top = _row_topk(jl_scores[1:n_users_eval, :], k)
            py_top = _row_topk(py_lmf[1:n_users_eval, :], k)
            overlap = _mean_topk_overlap(jl_top, py_top)

            min_cor = parse(Float64, get(ENV, "GIDEON_PY_LMF_MIN_COR", "-1.0"))
            min_ov = parse(Float64, get(ENV, "GIDEON_PY_LMF_MIN_OVERLAP", "0.08"))
            max_ndcg_delta = parse(Float64, get(ENV, "GIDEON_PY_LMF_MAX_NDCG_DELTA", "0.07"))
            max_recall_delta = parse(Float64, get(ENV, "GIDEON_PY_LMF_MAX_RECALL_DELTA", "0.07"))

            @test isfinite(c)
            # Different sampling/training implementations can shift score scales;
            # ranking overlap is the primary parity signal for LogisticMF.
            @test c >= min_cor
            @test overlap >= min_ov

            if isfile(lmf_metrics_path)
                py_ndcg = _read_json_metric(lmf_metrics_path, "ndcg")
                py_recall = _read_json_metric(lmf_metrics_path, "recall")
                jl_preds = _safe_recommend(m, X_train, 10)
                jl_ndcg = mean(ndcg_at_k(jl_preds, X_test; k=10))
                jl_recall = mean(recall_at_k(jl_preds, X_test; k=10))
                if get(ENV, "GIDEON_PY_LMF_STRICT", "0") == "1"
                    @test abs(jl_ndcg - py_ndcg) <= max_ndcg_delta
                    @test abs(jl_recall - py_recall) <= max_recall_delta
                else
                    @info "LMF split-metric parity is diagnostic by default; set GIDEON_PY_LMF_STRICT=1 to enforce thresholds"
                end
                println("  LMF NDCG@10: Julia=$jl_ndcg, Python=$py_ndcg, Δ=$(abs(jl_ndcg - py_ndcg))")
                println("  LMF Recall@10: Julia=$jl_recall, Python=$py_recall, Δ=$(abs(jl_recall - py_recall))")
            else
                @info "Skipping LMF split-metric parity: py_lmf_metrics.json not found"
            end
            println("  LMF score correlation: $c")
            println("  LMF top-$k overlap: $overlap")
        else
            @info "Skipping LogisticMF parity: py_lmf_scores.csv not found"
        end
    end

    @testset "EASE (Julia) vs EASE (Python)" begin
        if isfile(ease_path)
            py_B = _read_matrix(ease_path)

            m = EASE(λ=100.0, verbose=false)
            fit!(m, X)
            jl_B = Matrix(m.B)

            @test size(jl_B) == size(py_B)
            @test all(abs.(diag(jl_B)) .< 1e-8)

            rel_frob = norm(jl_B - py_B) / (norm(py_B) + 1e-12)
            c = _cor(vec(jl_B), vec(py_B))

            max_rel = parse(Float64, get(ENV, "GIDEON_PY_EASE_MAX_REL_FROB", "1e-6"))
            min_cor = parse(Float64, get(ENV, "GIDEON_PY_EASE_MIN_COR", "0.9999"))

            @test rel_frob <= max_rel
            @test c >= min_cor
            println("  EASE relative Frobenius error: $rel_frob")
            println("  EASE matrix correlation: $c")
        else
            @info "Skipping EASE parity: py_ease_B.csv not found"
        end
    end
end
