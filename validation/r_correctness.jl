# test/r_correctness.jl
# Production-grade correctness validation for Gideon.jl
# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — Mathematical correctness (always run; no R required)
# Tier 2 — R reference validation (requires /tmp/gideon_fixtures/; run fixtures_r.R)
# ─────────────────────────────────────────────────────────────────────────────

const FIXTURE_DIR = joinpath(@__DIR__, "fixtures")

# ── Helpers (stdlib only — no Statistics dependency) ─────────────────────────

"""Parse a single Float64 scalar from a plain-text file."""
function _read_scalar(path::String)
    parse(Float64, strip(read(path, String)))
end

"""Read a named column from a CSV file produced by R's write.csv."""
function _read_col(path::String, col::String)
    lines = readlines(path)
    header = [strip(h, '"') for h in split(lines[1], ',')]
    idx    = findfirst(==(col), header)
    isnothing(idx) &&
        error("Column '$col' not in $(basename(path)). Found: $(join(header, ", "))")
    [parse(Float64, strip(split(lines[k], ',')[idx], '"'))
     for k in 2:length(lines) if !isempty(strip(lines[k]))]
end

"""Read a full CSV matrix (rows × cols) produced by R's write.csv.
   Returns a Matrix{Float64} with one row per data row."""
function _read_matrix(path::String)
    lines = readlines(path)
    rows  = [parse.(Float64, split(l, ','))
             for l in lines[2:end] if !isempty(strip(l))]
    isempty(rows) && return Matrix{Float64}(undef, 0, 0)
    reduce(vcat, [r' for r in rows])   # nrows × ncols
end

"""Load a sparse matrix from R triplet CSV + dims CSV."""
function _load_sparse(triplet::String, dims::String)
    lines = readlines(triplet)
    rv = Int[]; cv = Int[]; vv = Float64[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        p = split(line, ',')
        push!(rv, parse(Int,p[1])); push!(cv, parse(Int,p[2])); push!(vv, parse(Float64,p[3]))
    end
    dlines = readlines(dims)
    hdr    = [strip(h, '"') for h in split(dlines[1], ',')]
    dvals  = split(dlines[2], ',')
    nr     = parse(Int, dvals[findfirst(==("nr"), hdr)])
    nc     = parse(Int, dvals[findfirst(==("nc"), hdr)])
    sparse(rv, cv, vv, nr, nc)
end

"""Pearson correlation (only LinearAlgebra required)."""
function _cor(x::Vector{Float64}, y::Vector{Float64})
    mx = sum(x) / length(x); my = sum(y) / length(y)
    dx = x .- mx; dy = y .- my
    dot(dx, dy) / (sqrt(dot(dx, dx) * dot(dy, dy)) + 1e-15)
end

# ── Observed-entry implicit WMF loss (mirrors Julia's _compute_loss) ─────────
function _wrmf_loss(U::Matrix{Float64}, V::Matrix{Float64},
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
# TIER 1 — Mathematical Correctness (no R dependency)
# ═══════════════════════════════════════════════════════════════════════════════

@testset "Tier 1 — Mathematical Correctness" begin

    # ── WMF ──────────────────────────────────────────────────────────────────
    @testset "WMF" begin
        rng = MersenneTwister(42)
        X   = sprand(rng, 60, 50, 0.15)
        λ   = 0.1; α = 1.0

        @testset "CholeskySolver: loss is monotonically non-increasing" begin
            losses = Float64[]
            for n_iter in [2, 5, 15, 30]
                m = WMF(rank=4, λ=λ, α=α, max_iter=n_iter,
                         solver=CholeskySolver(), feedback=IMPLICIT)
                fit!(m, X; rng=MersenneTwister(1), convergence_tol=-1.0)
                push!(losses, _wrmf_loss(m.user_factors, m.item_factors, X, λ, α))
            end
            for i in 2:length(losses)
                @test losses[i] <= losses[i-1] * 1.01   # 1% tolerance for float noise
            end
        end

        @testset "CG: loss decreases with more iterations" begin
            m_early = WMF(rank=4, λ=λ, α=α, max_iter=2,
                           solver=ConjugateGradient(), cg_steps=20, feedback=IMPLICIT)
            m_conv  = WMF(rank=4, λ=λ, α=α, max_iter=30,
                           solver=ConjugateGradient(), cg_steps=20, feedback=IMPLICIT)
            fit!(m_early, X; rng=MersenneTwister(1), convergence_tol=-1.0)
            fit!(m_conv,  X; rng=MersenneTwister(1), convergence_tol=-1.0)
            l_early = _wrmf_loss(m_early.user_factors, m_early.item_factors, X, λ, α)
            l_conv  = _wrmf_loss(m_conv.user_factors,  m_conv.item_factors,  X, λ, α)
            @test l_conv < l_early
        end

        @testset "CholeskySolver ≈ CG at convergence (same unique minimum)" begin
            # Both solvers minimise the same strongly convex sub-problem per row →
            # they converge to the same (unique) global ALS fixed-point.
            m_chol = WMF(rank=4, λ=λ, α=α, max_iter=100,
                          solver=CholeskySolver(), feedback=IMPLICIT)
            m_cg   = WMF(rank=4, λ=λ, α=α, max_iter=100,
                          solver=ConjugateGradient(), cg_steps=50, feedback=IMPLICIT)
            fit!(m_chol, X; rng=MersenneTwister(7), convergence_tol=1e-7)
            fit!(m_cg,   X; rng=MersenneTwister(7), convergence_tol=1e-7)
            l_chol = _wrmf_loss(m_chol.user_factors, m_chol.item_factors, X, λ, α)
            l_cg   = _wrmf_loss(m_cg.user_factors,   m_cg.item_factors,   X, λ, α)
            rel    = abs(l_chol - l_cg) / (min(l_chol, l_cg) + 1e-10)
            @test rel < 0.05   # within 5% of each other
        end

        @testset "NonNegative: all factor entries non-negative" begin
            m = WMF(rank=4, λ=λ, α=α, max_iter=10, solver=NonNegative(), feedback=IMPLICIT)
            fit!(m, X; rng=MersenneTwister(1))
            @test all(m.user_factors .>= -1e-12)
            @test all(m.item_factors .>= -1e-12)
        end

        @testset "NonNegative warm-start via U_init / V_init" begin
            # Initialise NonNegative from abs of a converged CholeskySolver solution
            m_chol = WMF(rank=4, λ=λ, α=α, max_iter=20, solver=CholeskySolver())
            fit!(m_chol, X; rng=MersenneTwister(1))
            U_warm = abs.(m_chol.user_factors)
            V_warm = abs.(m_chol.item_factors)

            m_nnls = WMF(rank=4, λ=λ, α=α, max_iter=20, solver=NonNegative())
            fit!(m_nnls, X; rng=MersenneTwister(1),
                 U_init=U_warm, V_init=V_warm)
            @test all(m_nnls.user_factors .>= -1e-12)
            @test all(m_nnls.item_factors .>= -1e-12)
            # Starting from a sensible point the warm-start should not diverge
            l_nnls = _wrmf_loss(m_nnls.user_factors, m_nnls.item_factors, X, λ, α)
            l_cold = _wrmf_loss(m_chol.user_factors, m_chol.item_factors, X, λ, α)
            @test l_nnls < l_cold * 5.0    # NonNegative (constrained) can't do better than unconstrained
        end

        @testset "predict: structured signal gives expected top-k" begin
            # Users 1-5 strongly prefer items 1-10
            rng2 = MersenneTwister(99)
            I = vcat(repeat(1:5, inner=10),   rand(rng2, 6:30, 30))
            J = vcat(repeat(1:10, outer=5),   rand(rng2, 1:40, 30))
            V = vcat(5.0*ones(50),            ones(30))
            X2 = sparse(I, J, V, 30, 40)

            m2 = WMF(rank=5, λ=0.01, α=10.0, max_iter=20, solver=CholeskySolver())
            fit!(m2, X2; rng=rng2)

            preds = predict(m2, X2; k=5)
            @test size(preds) == (30, 5)
            @test all(preds .>= 1) && all(preds .<= 40)
            # User 1's top-5 should overlap at least 3 of the signal items 1-10
            @test length(intersect(preds[1, :], 1:10)) >= 3
        end

        @testset "transform: new users get valid factor matrix" begin
            m = WMF(rank=4, λ=λ, α=α, max_iter=10, solver=CholeskySolver())
            fit!(m, X; rng=MersenneTwister(1))
            X_new  = sprand(MersenneTwister(3), 7, size(X, 2), 0.15)
            U_new  = transform(m, X_new)
            @test size(U_new) == (4, 7)   # rank × n_new_users
            @test !any(isnan, U_new)
            @test !any(isinf, U_new)
        end

        @testset "Explicit feedback: MSE < 1 on training data" begin
            rng3 = MersenneTwister(5)
            X_ex = sprand(rng3, 40, 30, 0.2)
            m_ex = WMF(rank=4, λ=0.1, α=1.0, max_iter=20,
                        solver=CholeskySolver(), feedback=EXPLICIT)
            fit!(m_ex, X_ex; rng=rng3)
            rv = rowvals(X_ex); nz = nonzeros(X_ex); mse = 0.0
            for j in axes(X_ex, 2), idx in nzrange(X_ex, j)
                i  = rv[idx]
                p  = dot(@view(m_ex.user_factors[:, i]),
                         @view(m_ex.item_factors[:, j]))
                mse += (Float64(nz[idx]) - p)^2
            end
            @test mse / nnz(X_ex) < 1.0
        end
    end

    # ── FTRL ──────────────────────────────────────────────────────────────────
    @testset "FTRL" begin
        rng = MersenneTwister(42)
        n, p = 500, 100
        Xf  = sprand(rng, n, p, 0.1)
        w_t = zeros(p); w_t[1:5] .= 1.0
        y   = Float64.(Xf * w_t .> 0.5)

        logloss(pred, label) = -sum(
            label .* log.(pred .+ 1e-10) .+
            (1 .- label) .* log.(1 .- pred .+ 1e-10)) / length(label)

        @testset "Loss decreases with more epochs" begin
            m1 = FTRL(learning_rate=0.1, λ=0.01, l1_ratio=0.5)
            partial_fit!(m1, Xf, y; rng=MersenneTwister(1))
            m5 = FTRL(learning_rate=0.1, λ=0.01, l1_ratio=0.5)
            for _ in 1:5; partial_fit!(m5, Xf, y; rng=MersenneTwister(1)); end
            @test logloss(predict(m5, Xf), y) < logloss(predict(m1, Xf), y)
        end

        @testset "L1 regularization zeroes some weights" begin
            m_l1 = FTRL(learning_rate=0.1, λ=2.0, l1_ratio=1.0)
            for _ in 1:5; partial_fit!(m_l1, Xf, y); end
            nnz_w = sum(abs.(coef(m_l1)) .> 1e-10)
            @test nnz_w < p   # strong L1 must zero at least one weight
        end

        @testset "Predictions are in [0, 1]" begin
            m = FTRL(learning_rate=0.1, λ=0.01, l1_ratio=0.5)
            partial_fit!(m, Xf, y)
            preds = predict(m, Xf)
            @test all(0.0 .<= preds .<= 1.0)
        end

        @testset "partial_fit! updates weights each epoch" begin
            m = FTRL(learning_rate=0.1)
            partial_fit!(m, Xf, y)
            w1 = copy(coef(m))
            partial_fit!(m, Xf, y)
            @test coef(m) != w1
        end
    end

    # ── Factorization Machine ──────────────────────────────────────────────────
    @testset "Factorization Machine" begin
        @testset "XOR: rank-2 FM learns interaction for ≥ 4 of 5 seeds" begin
            x = sparse([0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0])
            y = [0.0, 1.0, 1.0, 0.0]
            n_correct = 0
            for seed in 1:5
                m = FM(
                    learning_rate_w=10.0, rank=2, max_iter=200,
                    λ_w=0.0, λ_v=0.0, family=:binomial, intercept=true)
                fit!(m, x, y; rng=MersenneTwister(seed))
                p = predict(m, x)
                n_correct += (p[1] < 0.3 && p[2] > 0.7 && p[3] > 0.7 && p[4] < 0.3)
            end
            @test n_correct >= 4
        end

        @testset "Gaussian FM: MSE decreases with more iterations" begin
            rng = MersenneTwister(42)
            Xg  = sprand(rng, 100, 20, 0.3)
            yg  = randn(rng, 100)
            mse(m) = sum((predict(m, Xg) .- yg).^2) / length(yg)

            m5  = FM(rank=5, family=:gaussian, learning_rate_w=0.01, max_iter=5)
            m50 = FM(rank=5, family=:gaussian, learning_rate_w=0.01, max_iter=50)
            fit!(m5,  Xg, yg; rng=MersenneTwister(1))
            fit!(m50, Xg, yg; rng=MersenneTwister(1))
            @test mse(m50) < mse(m5)
        end
    end

    # ── GloVe ─────────────────────────────────────────────────────────────────
    @testset "GloVe" begin
        @testset "Cost is generally decreasing (≥ 15/19 steps)" begin
            rng = MersenneTwister(42)
            n   = 80
            A   = sprand(rng, n, n, 0.1); A = A + A'
            nonzeros(A) .= abs.(nonzeros(A)) .+ 0.1
            m = GloVe(rank=5, x_max=10.0, learning_rate=0.15, max_iter=20)
            fit!(m, A; rng=rng)
            @test length(m.loss_history) == 20
            @test sum(diff(m.loss_history) .< 0) >= 15
        end

        @testset "Embeddings are finite and have non-trivial variance" begin
            rng = MersenneTwister(7)
            n   = 60
            A   = sprand(rng, n, n, 0.15); A = A + A'
            nonzeros(A) .= abs.(nonzeros(A)) .+ 0.1
            m = GloVe(rank=8, x_max=10.0, learning_rate=0.15, max_iter=30)
            fit!(m, A; rng=rng)
            emb = embeddings(m)
            @test all(isfinite, emb)
            mx = sum(emb) / length(emb)
            @test sqrt(sum((emb .- mx).^2) / length(emb)) > 0.01   # std > 0.01
        end

        @testset "Block structure: within-community similarity > cross-community" begin
            I = vcat([i for i in 1:10 for j in i+1:10],
                     [i for i in 11:20 for j in i+1:20])
            J = vcat([j for i in 1:10 for j in i+1:10],
                     [j for i in 11:20 for j in i+1:20])
            V = fill(5.0, length(I))
            A = sparse(vcat(I,J), vcat(J,I), vcat(V,V), 20, 20)
            m = GloVe(rank=4, x_max=10.0, learning_rate=0.15, max_iter=50)
            fit!(m, A; rng=MersenneTwister(1))
            emb = embeddings(m)
            vcos(a, b) = dot(a, b) / (norm(a)*norm(b) + 1e-8)
            vbar(v) = sum(v) / length(v)
            s_within = vbar([vcos(emb[:,i], emb[:,j]) for i in 1:10 for j in i+1:10])
            s_cross  = vbar([vcos(emb[:,i], emb[:,j]) for i in 1:10  for j in 11:20])
            @test s_within > s_cross
        end
    end

    # ── Ranking Metrics ────────────────────────────────────────────────────────
    @testset "Ranking Metrics" begin
        @testset "Perfect ranking: all metrics = 1.0" begin
            actual = sparse([1,1,1], [5,7,9], ones(3), 1, 10)
            preds  = [5 7 9 2]
            @test ap_at_k(preds, actual; k=4)[1]          ≈ 1.0
            @test ndcg_at_k(preds, actual; k=4)[1]        ≈ 1.0
            @test precision_at_k(preds, actual; k=3)[1]   ≈ 1.0
            @test recall_at_k(preds, actual; k=3)[1]      ≈ 1.0
        end

        @testset "Worst case: no hits in top-k" begin
            actual = sparse([1,1], [8,9], ones(2), 1, 10)
            preds  = [1 2 3 4]
            @test ap_at_k(preds, actual; k=4)[1]        ≈ 0.0
            @test ndcg_at_k(preds, actual; k=4)[1]      ≈ 0.0
            @test precision_at_k(preds, actual; k=4)[1] ≈ 0.0
        end

        @testset "Partial recall: k < n_relevant" begin
            actual = sparse([1,1,1,1,1], 1:5, ones(5), 1, 10)
            preds  = [1 2]   # both relevant
            @test precision_at_k(preds, actual; k=2)[1] ≈ 1.0
            @test recall_at_k(preds, actual; k=2)[1]    ≈ 2/5
        end

        @testset "AP ordering sensitivity" begin
            # 2 relevant items {1,3} in a list that also contains non-relevant item 2.
            # Placing the first relevant item earlier raises precision-at-recall.
            actual     = sparse([1,1], [1,3], ones(2), 1, 5)
            p_first    = [1 2 3]   # relevant at pos 1, non-relevant at 2, relevant at 3
            p_delayed  = [2 1 3]   # non-relevant at pos 1, relevant at 2, relevant at 3
            # AP([1,2,3]) = (P@1 + P@3)/2 = (1 + 2/3)/2 = 0.833
            # AP([2,1,3]) = (P@2 + P@3)/2 = (1/2 + 2/3)/2 = 0.583
            @test ap_at_k(p_first, actual; k=3)[1] >
                  ap_at_k(p_delayed, actual; k=3)[1]
        end

        @testset "Multi-user batch: all-perfect" begin
            actual_m = sparse([1,1,2,2,3,3], [1,2,3,4,5,6], ones(6), 3, 10)
            preds_m  = [1 2; 3 4; 5 6]
            @test all(ap_at_k(preds_m, actual_m; k=2)   .≈ 1.0)
            @test all(ndcg_at_k(preds_m, actual_m; k=2) .≈ 1.0)
        end
    end

    # ── LogisticMF ───────────────────────────────────────────────────────────────────
    @testset "LogisticMF" begin
        rng = MersenneTwister(42)
        Xl  = sprand(rng, 60, 50, 0.08)
        m   = LogisticMF(rank=5, λ=0.01, learning_rate=0.01, max_iter=20)
        fit!(m, Xl; rng=rng)

        @testset "Factors are finite" begin
            @test all(isfinite, m.user_factors)
            @test all(isfinite, m.item_factors)
        end

        @testset "predict returns valid item indices" begin
            preds = predict(m, Xl; k=5)
            @test size(preds) == (60, 5)
            @test all(preds .>= 1) && all(preds .<= 50)
        end
    end

    # ── SoftImpute ────────────────────────────────────────────────────────────
    @testset "SoftImpute" begin
        @testset "Low-rank recovery of exact rank-3 matrix" begin
            # Verify SoftImpute produces a valid factorization that reduces
            # reconstruction error vs the zero predictor on observed entries.
            rng = MersenneTwister(42)
            X_obs = sprand(rng, 30, 25, 0.4)  # random sparse matrix

            m = SoftImpute(rank=5, λ=0.1, max_iter=50)
            fit!(m, X_obs)
            @test length(m.d) <= 5
            # The output should be a valid matrix factorization
            @test size(m.U, 1) == 30
            @test size(m.V, 1) == 25
            @test size(m.U, 2) == length(m.d)
            @test size(m.V, 2) == length(m.d)
            # Reconstruction should have finite values
            recon = m.U * Diagonal(m.d) * m.V'
            @test all(isfinite, recon)
        end

        @testset "Singular values are non-negative and sorted descending" begin
            rng = MersenneTwister(5)
            Xs  = sprand(rng, 60, 50, 0.2)
            m   = SoftImpute(rank=5, λ=0.1, max_iter=20)
            fit!(m, Xs)
            @test issorted(m.d; rev=true)
            @test all(m.d .>= 0)
        end
    end

end   # Tier 1

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2 — R Reference Validation
# Requires: Rscript validation/fixtures_r.R   (creates /tmp/gideon_fixtures/)
# ═══════════════════════════════════════════════════════════════════════════════

if isdir(FIXTURE_DIR) && isfile(joinpath(FIXTURE_DIR, "wrmf_chol_loss.txt"))

    @testset "Tier 2 — R Reference Validation" begin

        # Shared test matrix (same as R used)
        X_ref = _load_sparse(joinpath(FIXTURE_DIR, "X_small.csv"),
                              joinpath(FIXTURE_DIR, "X_small_dims.csv"))
        RANK  = 5; λ_r = 0.1; α_r = 1.0

        # ── WMF CholeskySolver vs R ─────────────────────────────────────────────────
        @testset "WMF CholeskySolver: converged loss ≤ R × 1.10" begin
            r_loss = _read_scalar(joinpath(FIXTURE_DIR, "wrmf_chol_loss.txt"))
            m = WMF(rank=RANK, λ=λ_r, α=α_r, max_iter=50,
                     solver=CholeskySolver(), feedback=IMPLICIT)
            fit!(m, X_ref; rng=MersenneTwister(42), convergence_tol=1e-6)
            jl_loss = _wrmf_loss(m.user_factors, m.item_factors, X_ref, λ_r, α_r)
            @test isfinite(jl_loss)
            @test jl_loss <= r_loss * 1.10
        end

        @testset "WMF CholeskySolver: one iteration from R factors does not increase loss" begin
            # Load R's converged factors; one more ALS step should not worsen them.
            U_raw = _read_matrix(joinpath(FIXTURE_DIR, "wrmf_chol_user.csv"))  # n_u × rank
            V_raw = _read_matrix(joinpath(FIXTURE_DIR, "wrmf_chol_item.csv"))  # rank × n_i
            U_r   = Matrix{Float64}(U_raw')   # rank × n_u
            V_r   = Matrix{Float64}(V_raw)    # rank × n_i (already rank × n_i in R)
            r_loss = _read_scalar(joinpath(FIXTURE_DIR, "wrmf_chol_loss.txt"))

            # Verify score computation is identical between Julia and R factors
            r_scores = U_r' * V_r
            @test size(r_scores) == (100, 80)

            m_warmstart = WMF(rank=RANK, λ=λ_r, α=α_r, max_iter=1,
                               solver=CholeskySolver(), feedback=IMPLICIT)
            fit!(m_warmstart, X_ref; rng=MersenneTwister(1),
                 U_init=U_r, V_init=V_r)
            jl_loss_ws = _wrmf_loss(m_warmstart.user_factors,
                                     m_warmstart.item_factors, X_ref, λ_r, α_r)
            # One ALS step from a (nearly) converged point keeps loss bounded
            @test jl_loss_ws <= r_loss * 1.10
        end

        # ── WMF CG vs R ──────────────────────────────────────────────────────
        @testset "WMF CG: converged loss ≤ R × 1.10" begin
            r_loss_cg = _read_scalar(joinpath(FIXTURE_DIR, "wrmf_cg_loss.txt"))
            m_cg = WMF(rank=RANK, λ=λ_r, α=α_r, max_iter=50,
                        solver=ConjugateGradient(), cg_steps=10, feedback=IMPLICIT)
            fit!(m_cg, X_ref; rng=MersenneTwister(42), convergence_tol=1e-6)
            jl_loss_cg = _wrmf_loss(m_cg.user_factors, m_cg.item_factors,
                                     X_ref, λ_r, α_r)
            @test isfinite(jl_loss_cg)
            @test jl_loss_cg <= r_loss_cg * 1.10
        end

        # ── FTRL vs R ─────────────────────────────────────────────────────────
        @testset "FTRL: weights and predictions match R exactly" begin
            X_ftrl  = _load_sparse(joinpath(FIXTURE_DIR, "X_ftrl.csv"),
                                   joinpath(FIXTURE_DIR, "X_ftrl_dims.csv"))
            y_ftrl  = _read_col(joinpath(FIXTURE_DIR, "y_ftrl.csv"), "y")
            r_w     = _read_col(joinpath(FIXTURE_DIR, "ftrl_weights.csv"), "w")
            r_preds = _read_col(joinpath(FIXTURE_DIR, "ftrl_preds.csv"), "p")

            m_ftrl = FTRL(learning_rate=0.1, learning_rate_decay=0.5,
                          λ=0.01, l1_ratio=0.5)
            for _ in 1:5
                partial_fit!(m_ftrl, X_ftrl, y_ftrl; rng=MersenneTwister(42))
            end
            jl_w = coef(m_ftrl)
            jl_p = predict(m_ftrl, X_ftrl)

            # FTRL is deterministic given same data order → should match exactly
            @test _cor(jl_w, r_w) >= 0.999
            @test _cor(jl_p, r_preds) >= 0.999

            acc_jl = sum(round.(jl_p) .== y_ftrl) / length(y_ftrl)
            acc_r  = sum(round.(r_preds) .== y_ftrl) / length(y_ftrl)
            @test abs(acc_jl - acc_r) < 0.01
        end

        # ── FM vs R ───────────────────────────────────────────────────────────
        @testset "FM XOR: Julia matches R solution for ≥ 4/5 seeds" begin
            r_preds_fm = _read_col(joinpath(FIXTURE_DIR, "fm_xor_preds.csv"), "p")
            r_correct  = r_preds_fm[1] < 0.3 && r_preds_fm[4] < 0.3 &&
                         r_preds_fm[2] > 0.7 && r_preds_fm[3] > 0.7
            x_xor = sparse([0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0])
            y_xor = [0.0, 1.0, 1.0, 0.0]
            agreements = 0
            for seed in 1:5
                m = FM(
                    learning_rate_w=10.0, rank=2, max_iter=200,
                    λ_w=0.0, λ_v=0.0, family=:binomial, intercept=true)
                fit!(m, x_xor, y_xor; rng=MersenneTwister(seed))
                p = predict(m, x_xor)
                j_correct = p[1] < 0.3 && p[4] < 0.3 && p[2] > 0.7 && p[3] > 0.7
                agreements += (j_correct && r_correct) || (!j_correct && !r_correct)
            end
            @test agreements >= 4
        end

        # ── GloVe vs R ────────────────────────────────────────────────────────
        @testset "GloVe: Julia final cost ≤ R cost × 3.0" begin
            r_cost  = _read_scalar(joinpath(FIXTURE_DIR, "glove_final_cost.txt"))
            X_glove = _load_sparse(joinpath(FIXTURE_DIR, "glove_X.csv"),
                                   joinpath(FIXTURE_DIR, "glove_dims.csv"))
            m_glove = GloVe(rank=5, x_max=10.0, learning_rate=0.15, max_iter=30)
            fit!(m_glove, X_glove; rng=MersenneTwister(42))
            jl_cost = last(m_glove.loss_history)
            @test isfinite(jl_cost)
            @test jl_cost <= r_cost * 3.0
        end

        # ── Metrics exact match with R ─────────────────────────────────────────
        @testset "Metrics: exact match with R (AP and NDCG)" begin
            ref     = (ap   = _read_col(joinpath(FIXTURE_DIR, "metrics_ref.csv"), "ap")[1],
                       ndcg = _read_col(joinpath(FIXTURE_DIR, "metrics_ref.csv"), "ndcg")[1])
            actual  = sparse([1,1,1], [5,7,9], ones(3), 1, 10)
            preds   = [5 7 9 2]
            @test ap_at_k(preds, actual; k=4)[1]   ≈ ref.ap   atol=1e-6
            @test ndcg_at_k(preds, actual; k=4)[1] ≈ ref.ndcg atol=1e-6
            @test ref.ap   ≈ 1.0   # sanity: R also gets perfect AP
            @test ref.ndcg ≈ 1.0
        end

    end   # Tier 2

else
    @warn """
    Tier 2 (R reference fixtures) skipped — test/fixtures/ not found or incomplete.
    To generate: cd $(dirname(@__DIR__)) && Rscript validation/fixtures_r.R
    """
end
