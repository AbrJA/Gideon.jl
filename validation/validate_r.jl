# validation/validate_r.jl
# Reference validation against R implementations (optional, run locally as needed)
# ─────────────────────────────────────────────────────────────────────────────
# This script validates Gideon.jl algorithms against R reference implementations.
# Run this locally when validating changes and performance.
#
# Prerequisites:
#   1. Generate fixtures: Rscript validation/fixtures_r.R
#   2. Run this script: julia --project=. validation/validate_r.jl
# ─────────────────────────────────────────────────────────────────────────────

using Gideon, SparseArrays, LinearAlgebra, Random
using Test

const FIXTURE_DIR = get(ENV, "GIDEON_R_FIXTURE_DIR", "/tmp/gideon_fixtures")

# ── Helpers for R fixture loading ─────────────────────────────────────────────

function _read_scalar(path::String)
    parse(Float64, strip(read(path, String)))
end

function _read_col(path::String, col::String)
    lines = readlines(path)
    header = [strip(h, '"') for h in split(lines[1], ',')]
    idx = findfirst(==(col), header)
    isnothing(idx) &&
        error("Column '$col' not in $(basename(path)). Found: $(join(header, ", "))")
    [parse(Float64, strip(split(lines[k], ',')[idx], '"'))
     for k in 2:length(lines) if !isempty(strip(lines[k]))]
end

function _read_matrix(path::String)
    lines = readlines(path)
    rows = [parse.(Float64, split(l, ','))
            for l in lines[2:end] if !isempty(strip(l))]
    isempty(rows) && return Matrix{Float64}(undef, 0, 0)
    reduce(vcat, [r' for r in rows])
end

function _load_sparse(triplet::String, dims::String)
    lines = readlines(triplet)
    rv = Int[]; cv = Int[]; vv = Float64[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        p = split(line, ',')
        push!(rv, parse(Int,p[1])); push!(cv, parse(Int,p[2])); push!(vv, parse(Float64,p[3]))
    end
    dlines = readlines(dims)
    hdr = [strip(h, '"') for h in split(dlines[1], ',')]
    dvals = split(dlines[2], ',')
    nr = parse(Int, dvals[findfirst(==("nr"), hdr)])
    nc = parse(Int, dvals[findfirst(==("nc"), hdr)])
    sparse(rv, cv, vv, nr, nc)
end

function _cor(x::Vector{<:AbstractFloat}, y::Vector{<:AbstractFloat})
    mx = sum(x) / length(x); my = sum(y) / length(y)
    dx = x .- mx; dy = y .- my
    dot(dx, dy) / (sqrt(dot(dx, dx) * dot(dy, dy)) + 1e-15)
end

function _wrmf_loss_ref(U::Matrix{<:AbstractFloat}, V::Matrix{<:AbstractFloat},
                        X::SparseMatrixCSC, λ::Float64, α::Float64)
    rv = rowvals(X); nz = nonzeros(X); loss = 0.0
    for j in axes(X, 2), idx in nzrange(X, j)
        i = rv[idx]; r = nz[idx]
        pred = dot(@view(U[:, i]), @view(V[:, j]))
        loss += (1.0 + α * r) * (1.0 - pred)^2
    end
    loss + λ * (sum(abs2, U) + sum(abs2, V))
end

_all_files_exist(paths::Vector{String}) = all(isfile, paths)

function _print_rsparse_capabilities_if_present()
    cap_path = joinpath(FIXTURE_DIR, "rsparse_capabilities.csv")
    isfile(cap_path) || return
    lines = readlines(cap_path)
    length(lines) <= 1 && return
    println("  rsparse capabilities (from fixtures):")
    for line in lines[2:end]
        isempty(strip(line)) && continue
        cols = split(strip(line), ',')
        # Supports both [model,available] and [rowid,model,available] CSV layouts.
        if length(cols) >= 3
            model = strip(cols[end - 1], '"')
            available = strip(cols[end], '"') == "1" ? "yes" : "no"
        elseif length(cols) == 2
            model = strip(cols[1], '"')
            available = strip(cols[2], '"') == "1" ? "yes" : "no"
        else
            continue
        end
        println("    - ", model, ": ", available)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 4 — R Reference Validation
# ═══════════════════════════════════════════════════════════════════════════════

@testset "R Reference Comparison" begin

    if !isdir(FIXTURE_DIR)
        @warn """
        Fixtures not found at: $FIXTURE_DIR
        To generate fixtures:
                    Rscript validation/fixtures_r.R
        """
        return
    end

    if !isfile(joinpath(FIXTURE_DIR, "wrmf_chol_loss.txt"))
        @warn """
        Fixture files incomplete. Expected to find:
          - wrmf_chol_loss.txt
          - wrmf_chol_user.csv, wrmf_chol_item.csv
          - wrmf_cg_loss.txt
          - X_small.csv, X_small_dims.csv
          - ftrl_*.csv, y_ftrl.csv
          - glove_*.csv, glove_final_cost.txt
          - metrics_ref.csv
        """
        return
    end

    _print_rsparse_capabilities_if_present()

    X_ref = _load_sparse(joinpath(FIXTURE_DIR, "X_small.csv"),
                         joinpath(FIXTURE_DIR, "X_small_dims.csv"))
    RANK = 5; λ_r = 0.1; α_r = 1.0

    @testset "WMF CholeskySolver: loss ≤ R × 1.05" begin
        r_loss = _read_scalar(joinpath(FIXTURE_DIR, "wrmf_chol_loss.txt"))
        m = WMF(rank=RANK, λ=λ_r, α=α_r, max_iter=50,
                 solver=CholeskySolver(), feedback=IMPLICIT, convergence_tol=1e-6, verbose=false)
        fit!(m, X_ref; rng=MersenneTwister(42))
        jl_loss = _wrmf_loss_ref(m.user_factors, m.item_factors, X_ref, λ_r, α_r)
        @test isfinite(jl_loss)
        @test jl_loss <= r_loss * 1.05
        println("  WMF CholeskySolver: Julia=$jl_loss, R=$r_loss, ratio=$(jl_loss/r_loss)")
    end

    @testset "WMF CholeskySolver: warm-start does not increase loss" begin
        U_raw = _read_matrix(joinpath(FIXTURE_DIR, "wrmf_chol_user.csv"))
        V_raw = _read_matrix(joinpath(FIXTURE_DIR, "wrmf_chol_item.csv"))
        U_r = Matrix{Float32}(U_raw')
        V_r = Matrix{Float32}(V_raw)
        r_loss = _read_scalar(joinpath(FIXTURE_DIR, "wrmf_chol_loss.txt"))

        m_ws = WMF(rank=RANK, λ=λ_r, α=α_r, max_iter=1,
                    solver=CholeskySolver(), feedback=IMPLICIT, convergence_tol=-1.0, verbose=false)
        fit!(m_ws, X_ref; rng=MersenneTwister(1), U_init=U_r, V_init=V_r)
        jl_loss_ws = _wrmf_loss_ref(m_ws.user_factors, m_ws.item_factors, X_ref, λ_r, α_r)
        @test jl_loss_ws <= r_loss * 1.05
        println("  WMF warm-start: Julia=$jl_loss_ws, R=$r_loss, ratio=$(jl_loss_ws/r_loss)")
    end

    @testset "WMF CG: loss ≤ R × 1.05" begin
        r_loss_cg = _read_scalar(joinpath(FIXTURE_DIR, "wrmf_cg_loss.txt"))
        m_cg = WMF(rank=RANK, λ=λ_r, α=α_r, max_iter=50,
                    solver=ConjugateGradient(), cg_steps=10,
                    convergence_tol=1e-6, verbose=false)
        fit!(m_cg, X_ref; rng=MersenneTwister(42))
        jl_loss_cg = _wrmf_loss_ref(m_cg.user_factors, m_cg.item_factors, X_ref, λ_r, α_r)
        @test isfinite(jl_loss_cg)
        @test jl_loss_cg <= r_loss_cg * 1.05
        println("  WMF CG: Julia=$jl_loss_cg, R=$r_loss_cg, ratio=$(jl_loss_cg/r_loss_cg)")
    end

    @testset "FTRL: weights and predictions match R (tight)" begin
        ftrl_files = [
            joinpath(FIXTURE_DIR, "X_ftrl.csv"),
            joinpath(FIXTURE_DIR, "X_ftrl_dims.csv"),
            joinpath(FIXTURE_DIR, "y_ftrl.csv"),
            joinpath(FIXTURE_DIR, "ftrl_weights.csv"),
            joinpath(FIXTURE_DIR, "ftrl_preds.csv"),
        ]
        if _all_files_exist(ftrl_files)
            X_ftrl = _load_sparse(joinpath(FIXTURE_DIR, "X_ftrl.csv"),
                                  joinpath(FIXTURE_DIR, "X_ftrl_dims.csv"))
            y_ftrl = _read_col(joinpath(FIXTURE_DIR, "y_ftrl.csv"), "y")
            r_w = _read_col(joinpath(FIXTURE_DIR, "ftrl_weights.csv"), "w")
            r_preds = _read_col(joinpath(FIXTURE_DIR, "ftrl_preds.csv"), "p")

            m_ftrl = FTRL(learning_rate=0.1, learning_rate_decay=0.5,
                          λ=0.01, l1_ratio=0.5, verbose=false)
            for _ in 1:5
                update!(m_ftrl, X_ftrl, y_ftrl; rng=MersenneTwister(42))
            end
            jl_w = coef(m_ftrl)
            jl_p = predict(m_ftrl, X_ftrl)

            cor_w = _cor(jl_w, r_w)
            cor_p = _cor(jl_p, r_preds)
            @test cor_w >= 0.9995
            @test cor_p >= 0.9995
            println("  FTRL weights correlation: $cor_w")
            println("  FTRL predictions correlation: $cor_p")
        else
            @info "Skipping FTRL comparison: one or more fixture files are missing"
        end
    end

    @testset "FM XOR: Julia matches R (optional)" begin
        fm_path = joinpath(FIXTURE_DIR, "fm_xor_preds.csv")
        if isfile(fm_path)
            r_preds_fm = _read_col(fm_path, "p")
            x_xor = sparse([0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0])
            y_xor = [0.0, 1.0, 1.0, 0.0]
            agreements = 0
            r_correct = r_preds_fm[1] < 0.3 && r_preds_fm[4] < 0.3 &&
                        r_preds_fm[2] > 0.7 && r_preds_fm[3] > 0.7
            for seed in 1:5
                m = FM(
                    learning_rate_w=10.0, rank=2, max_iter=200,
                    λ_w=0.0, λ_v=0.0, family=Binomial(), intercept=true, verbose=false)
                fit!(m, x_xor, y_xor; rng=MersenneTwister(seed))
                p = predict(m, x_xor)
                j_correct = p[1] < 0.3 && p[4] < 0.3 && p[2] > 0.7 && p[3] > 0.7
                agreements += (j_correct && r_correct) || (!j_correct && !r_correct)
            end
            @test agreements >= 4
            println("  FM XOR: $agreements/5 seeds agree with R")
        else
            @info "Skipping FM XOR comparison: fixture missing (rsparse::FM may be unavailable)"
        end
    end

    @testset "GloVe: Julia cost ≤ R × 2.1" begin
        glove_files = [
            joinpath(FIXTURE_DIR, "glove_final_cost.txt"),
            joinpath(FIXTURE_DIR, "glove_X.csv"),
            joinpath(FIXTURE_DIR, "glove_dims.csv"),
        ]
        if _all_files_exist(glove_files)
            r_cost = _read_scalar(joinpath(FIXTURE_DIR, "glove_final_cost.txt"))
            X_glove = _load_sparse(joinpath(FIXTURE_DIR, "glove_X.csv"),
                                   joinpath(FIXTURE_DIR, "glove_dims.csv"))
            m_glove = GloVe(rank=5, x_max=10.0, learning_rate=0.15, max_iter=30, verbose=false)
            fit!(m_glove, X_glove; rng=MersenneTwister(42))
            jl_cost = last(m_glove.loss_history)
            @test isfinite(jl_cost)
            @test jl_cost <= r_cost * 2.1
            println("  GloVe cost: Julia=$jl_cost, R=$r_cost, ratio=$(jl_cost/r_cost)")
        else
            @info "Skipping GloVe comparison: one or more fixture files are missing"
        end
    end

    @testset "Metrics: exact match with R" begin
        metrics_path = joinpath(FIXTURE_DIR, "metrics_ref.csv")
        if isfile(metrics_path)
            ref = (ap   = _read_col(metrics_path, "ap")[1],
                   ndcg = _read_col(metrics_path, "ndcg")[1])
            actual = sparse([1,1,1], [5,7,9], ones(3), 1, 10)
            preds = [5 7 9 2]
            ap = ap_at_k(preds, actual; k=4)[1]
            ndcg = ndcg_at_k(preds, actual; k=4)[1]
            @test ap ≈ ref.ap atol=1e-6
            @test ndcg ≈ ref.ndcg atol=1e-6
            println("  Metrics AP: Julia=$ap, R=$(ref.ap)")
            println("  Metrics NDCG: Julia=$ndcg, R=$(ref.ndcg)")
        else
            @info "Skipping metrics comparison: metrics_ref.csv is missing"
        end
    end

    @testset "SoftImpute: singular values and reconstruction match R" begin
        si_d_path = joinpath(FIXTURE_DIR, "softimpute_si_d.csv")
        si_obs_path = joinpath(FIXTURE_DIR, "softimpute_si_obs_preds.csv")
        si_frob_path = joinpath(FIXTURE_DIR, "softimpute_si_frob.txt")
        if _all_files_exist([si_d_path, si_obs_path, si_frob_path])
            r_d = _read_col(si_d_path, "d")
            r_obs = _read_col(si_obs_path, "pred")
            r_frob = _read_scalar(si_frob_path)

            m_si = SoftImpute(rank=5, λ=1.0, max_iter=100,
                              convergence_tol=1e-6,
                              final_svd=true, verbose=false)
            fit!(m_si, X_ref; rng=MersenneTwister(42))

            jl_d = m_si.d
            jl_frob = sum(abs2, jl_d)

            # Reconstruction at observed positions
            recon = m_si.U * Diagonal(m_si.d) * m_si.V'
            rv = rowvals(X_ref); nz = nonzeros(X_ref)
            jl_obs = Float64[]
            for j in axes(X_ref, 2), idx in nzrange(X_ref, j)
                push!(jl_obs, recon[rv[idx], j])
            end

            # Singular values: with rank-constrained SoftImpute, different random
            # inits can converge to different local optima (non-convex when rank < true rank).
            # We validate: same total variance, reasonable SV agreement, high reconstruction.
            r_d_sorted = sort(r_d, rev=true)
            jl_d_sorted = sort(jl_d, rev=true)
            n_compare = min(length(r_d_sorted), length(jl_d_sorted))
            sv_rel_err = norm(r_d_sorted[1:n_compare] .- jl_d_sorted[1:n_compare]) /
                         (norm(r_d_sorted[1:n_compare]) + 1e-15)
            @test sv_rel_err < 0.25
            println("  SoftImpute SV relative error: $sv_rel_err")

            # Frobenius norm (total variance): should be very close
            frob_rel = abs(jl_frob - r_frob) / (r_frob + 1e-15)
            @test frob_rel < 0.05
            println("  SoftImpute Frob norm: Julia=$jl_frob, R=$r_frob, rel=$frob_rel")

            # Reconstruction correlation at observed entries
            cor_obs = _cor(jl_obs, r_obs)
            @test cor_obs >= 0.97
            println("  SoftImpute obs reconstruction correlation: $cor_obs")
        else
            @info "Skipping SoftImpute comparison: fixture files missing"
        end
    end

    @testset "SoftSVD: singular values and reconstruction match R" begin
        svd_d_path = joinpath(FIXTURE_DIR, "softimpute_svd_d.csv")
        svd_obs_path = joinpath(FIXTURE_DIR, "softimpute_svd_obs_preds.csv")
        svd_frob_path = joinpath(FIXTURE_DIR, "softimpute_svd_frob.txt")
        if _all_files_exist([svd_d_path, svd_obs_path, svd_frob_path])
            r_d = _read_col(svd_d_path, "d")
            r_obs = _read_col(svd_obs_path, "pred")
            r_frob = _read_scalar(svd_frob_path)

            m_svd = SoftSVD(rank=5, λ=1.0, max_iter=100,
                            convergence_tol=1e-6,
                            final_svd=true, verbose=false)
            fit!(m_svd, X_ref; rng=MersenneTwister(42))

            jl_d = m_svd.d
            jl_frob = sum(abs2, jl_d)

            recon = m_svd.U * Diagonal(m_svd.d) * m_svd.V'
            rv = rowvals(X_ref); nz = nonzeros(X_ref)
            jl_obs = Float64[]
            for j in axes(X_ref, 2), idx in nzrange(X_ref, j)
                push!(jl_obs, recon[rv[idx], j])
            end

            r_d_sorted = sort(r_d, rev=true)
            jl_d_sorted = sort(jl_d, rev=true)
            n_compare = min(length(r_d_sorted), length(jl_d_sorted))
            sv_rel_err = norm(r_d_sorted[1:n_compare] .- jl_d_sorted[1:n_compare]) /
                         (norm(r_d_sorted[1:n_compare]) + 1e-15)
            @test sv_rel_err < 0.05
            println("  SoftSVD SV relative error: $sv_rel_err")

            frob_rel = abs(jl_frob - r_frob) / (r_frob + 1e-15)
            @test frob_rel < 0.10
            println("  SoftSVD Frob norm: Julia=$jl_frob, R=$r_frob, rel=$frob_rel")

            cor_obs = _cor(jl_obs, r_obs)
            @test cor_obs >= 0.99
            println("  SoftSVD obs reconstruction correlation: $cor_obs")
        else
            @info "Skipping SoftSVD comparison: fixture files missing"
        end
    end

end
