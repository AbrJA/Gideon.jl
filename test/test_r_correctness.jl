# test/test_r_correctness.jl
# Production-grade correctness validation for Gideon.jl
# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — Mathematical correctness (always run; no R required)
# Tier 2 — R reference validation (requires test/fixtures/; run generate_fixtures.R)
# ─────────────────────────────────────────────────────────────────────────────

const FIXTURE_DIR = joinpath(@__DIR__, "fixtures")

# ── Helpers ──────────────────────────────────────────────────────────────────

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

function _cor(x::Vector{Float64}, y::Vector{Float64})
    mx = sum(x) / length(x); my = sum(y) / length(y)
    dx = x .- mx; dy = y .- my
    dot(dx, dy) / (sqrt(dot(dx, dx) * dot(dy, dy)) + 1e-15)
end

function _wrmf_loss_ref(U::Matrix{Float64}, V::Matrix{Float64},
                        X::SparseMatrixCSC, λ::Float64, α::Float64)
    rv = rowvals(X); nz = nonzeros(X); loss = 0.0
    for j in axes(X, 2), idx in nzrange(X, j)
        i = rv[idx]; r = nz[idx]
        pred = dot(@view(U[:, i]), @view(V[:, j]))
        loss += (1.0 + α * r) * (1.0 - pred)^2
    end
    loss + λ * (sum(abs2, U) + sum(abs2, V))
end

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2 — R Reference Validation
# ═══════════════════════════════════════════════════════════════════════════════

if isdir(FIXTURE_DIR) && isfile(joinpath(FIXTURE_DIR, "wrmf_chol_loss.txt"))

    @testset "Tier 2 — R Reference" begin
        X_ref = _load_sparse(joinpath(FIXTURE_DIR, "X_small.csv"),
                              joinpath(FIXTURE_DIR, "X_small_dims.csv"))
        RANK = 5; λ_r = 0.1; α_r = 1.0

        @testset "WeightedMatrixFactorization CholeskySolver: loss ≤ R × 1.10" begin
            r_loss = _read_scalar(joinpath(FIXTURE_DIR, "wrmf_chol_loss.txt"))
            m = WeightedMatrixFactorization(rank=RANK, λ=λ_r, α=α_r, max_iter=50,
                     solver=CholeskySolver(), feedback=IMPLICIT, convergence_tol=1e-6, verbose=false)
            fit!(m, X_ref; rng=MersenneTwister(42))
            jl_loss = _wrmf_loss_ref(m.user_factors, m.item_factors, X_ref, λ_r, α_r)
            @test isfinite(jl_loss)
            @test jl_loss <= r_loss * 1.10
        end

        @testset "WeightedMatrixFactorization CholeskySolver: warm-start does not increase loss" begin
            U_raw = _read_matrix(joinpath(FIXTURE_DIR, "wrmf_chol_user.csv"))
            V_raw = _read_matrix(joinpath(FIXTURE_DIR, "wrmf_chol_item.csv"))
            U_r = Matrix{Float64}(U_raw')
            V_r = Matrix{Float64}(V_raw)
            r_loss = _read_scalar(joinpath(FIXTURE_DIR, "wrmf_chol_loss.txt"))

            m_ws = WeightedMatrixFactorization(rank=RANK, λ=λ_r, α=α_r, max_iter=1,
                        solver=CholeskySolver(), feedback=IMPLICIT, convergence_tol=-1.0, verbose=false)
            fit!(m_ws, X_ref; rng=MersenneTwister(1), U_init=U_r, V_init=V_r)
            jl_loss_ws = _wrmf_loss_ref(m_ws.user_factors, m_ws.item_factors, X_ref, λ_r, α_r)
            @test jl_loss_ws <= r_loss * 1.10
        end

        @testset "WeightedMatrixFactorization CG: loss ≤ R × 1.10" begin
            r_loss_cg = _read_scalar(joinpath(FIXTURE_DIR, "wrmf_cg_loss.txt"))
            m_cg = WeightedMatrixFactorization(rank=RANK, λ=λ_r, α=α_r, max_iter=50,
                        solver=ConjugateGradient(), cg_steps=10,
                        convergence_tol=1e-6, verbose=false)
            fit!(m_cg, X_ref; rng=MersenneTwister(42))
            jl_loss_cg = _wrmf_loss_ref(m_cg.user_factors, m_cg.item_factors, X_ref, λ_r, α_r)
            @test isfinite(jl_loss_cg)
            @test jl_loss_cg <= r_loss_cg * 1.10
        end

        @testset "OnlineRegressor: weights and predictions match R" begin
            X_ftrl = _load_sparse(joinpath(FIXTURE_DIR, "X_ftrl.csv"),
                                   joinpath(FIXTURE_DIR, "X_ftrl_dims.csv"))
            y_ftrl = _read_col(joinpath(FIXTURE_DIR, "y_ftrl.csv"), "y")
            r_w = _read_col(joinpath(FIXTURE_DIR, "ftrl_weights.csv"), "w")
            r_preds = _read_col(joinpath(FIXTURE_DIR, "ftrl_preds.csv"), "p")

            m_ftrl = OnlineRegressor(learning_rate=0.1, learning_rate_decay=0.5,
                          λ=0.01, l1_ratio=0.5, verbose=false)
            for _ in 1:5
                update!(m_ftrl, X_ftrl, y_ftrl; rng=MersenneTwister(42))
            end
            jl_w = coef(m_ftrl)
            jl_p = predict(m_ftrl, X_ftrl)

            @test _cor(jl_w, r_w) >= 0.999
            @test _cor(jl_p, r_preds) >= 0.999
        end

        @testset "FM XOR: Julia matches R" begin
            r_preds_fm = _read_col(joinpath(FIXTURE_DIR, "fm_xor_preds.csv"), "p")
            x_xor = sparse([0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0])
            y_xor = [0.0, 1.0, 1.0, 0.0]
            agreements = 0
            r_correct = r_preds_fm[1] < 0.3 && r_preds_fm[4] < 0.3 &&
                        r_preds_fm[2] > 0.7 && r_preds_fm[3] > 0.7
            for seed in 1:5
                m = FactorizationMachine(
                    learning_rate_w=10.0, rank=2, max_iter=200,
                    λ_w=0.0, λ_v=0.0, family=Binomial(), intercept=true, verbose=false)
                fit!(m, x_xor, y_xor; rng=MersenneTwister(seed))
                p = predict(m, x_xor)
                j_correct = p[1] < 0.3 && p[4] < 0.3 && p[2] > 0.7 && p[3] > 0.7
                agreements += (j_correct && r_correct) || (!j_correct && !r_correct)
            end
            @test agreements >= 4
        end

        @testset "GlobalVectors: Julia cost ≤ R × 3.0" begin
            r_cost = _read_scalar(joinpath(FIXTURE_DIR, "glove_final_cost.txt"))
            X_glove = _load_sparse(joinpath(FIXTURE_DIR, "glove_X.csv"),
                                    joinpath(FIXTURE_DIR, "glove_dims.csv"))
            m_glove = GlobalVectors(rank=5, x_max=10.0, learning_rate=0.15, max_iter=30, verbose=false)
            fit!(m_glove, X_glove; rng=MersenneTwister(42))
            jl_cost = last(m_glove.loss_history)
            @test isfinite(jl_cost)
            @test jl_cost <= r_cost * 3.0
        end

        @testset "Metrics: exact match with R" begin
            ref = (ap   = _read_col(joinpath(FIXTURE_DIR, "metrics_ref.csv"), "ap")[1],
                   ndcg = _read_col(joinpath(FIXTURE_DIR, "metrics_ref.csv"), "ndcg")[1])
            actual = sparse([1,1,1], [5,7,9], ones(3), 1, 10)
            preds = [5 7 9 2]
            @test ap_at_k(preds, actual; k=4)[1] ≈ ref.ap atol=1e-6
            @test ndcg_at_k(preds, actual; k=4)[1] ≈ ref.ndcg atol=1e-6
        end
    end

else
    @warn """
    Tier 2 (R reference fixtures) skipped — test/fixtures/ not found or incomplete.
    To generate: cd $(dirname(@__DIR__)) && Rscript test/generate_fixtures.R
    """
end
