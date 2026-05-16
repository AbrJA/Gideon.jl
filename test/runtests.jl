using Test
using Gideon
using SparseArrays
using LinearAlgebra
using Random
using Aqua
using JET

@testset "Gideon.jl" begin

    # ──────────────────────────────────────────────────────────
    # Aqua.jl — automated quality assurance
    # ──────────────────────────────────────────────────────────
    @testset "Aqua" begin
        Aqua.test_all(Gideon; ambiguities=false)
    end

    # ──────────────────────────────────────────────────────────
    # JET.jl — static analysis
    # ──────────────────────────────────────────────────────────
    @testset "JET" begin
        JET.test_package(Gideon; target_modules=(Gideon,))
    end

    # ──────────────────────────────────────────────────────────
    # Type hierarchy
    # ──────────────────────────────────────────────────────────
    @testset "Type Hierarchy" begin
        @test WRMF <: AbstractMatrixFactorization
        @test WRMF <: AbstractSparseModel
        @test GloVe <: AbstractMatrixFactorization
        @test LMF <: AbstractMatrixFactorization
        @test FTRL <: AbstractSparseRegression
        @test FTRL <: AbstractSparseModel
        @test FactorizationMachine <: AbstractSparseRegression
    end

    # ──────────────────────────────────────────────────────────
    # Utility functions
    # ──────────────────────────────────────────────────────────
    @testset "Utilities" begin
        @test sigmoid(0.0) ≈ 0.5
        @test sigmoid(100.0) ≈ 1.0 atol=1e-10
        @test sigmoid(-100.0) ≈ 0.0 atol=1e-10
        @test sigmoid(1.0) ≈ 1 / (1 + exp(-1.0))

        rng = MersenneTwister(42)
        F = init_factors(rng, 5, 10)
        @test size(F) == (5, 10)
        @test all(isfinite, F)
    end

    # ──────────────────────────────────────────────────────────
    # Sparse utilities
    # ──────────────────────────────────────────────────────────
    @testset "Sparse Utils" begin
        rng = MersenneTwister(42)
        A = sprand(rng, 50, 30, 0.1)

        norms = sparse_row_norms(A, 2)
        @test length(norms) == 50
        @test all(norms .>= 0)

        cnnz = sparse_col_nnz(A)
        @test length(cnnz) == 30
        @test sum(cnnz) == nnz(A)

        rnnz = sparse_row_nnz(A)
        @test length(rnnz) == 50
        @test sum(rnnz) == nnz(A)

        A_csc, At = dual_representation(A)
        @test size(At) == (30, 50)
        @test A_csc === A
    end

    # ──────────────────────────────────────────────────────────
    # WRMF — convergence & correctness
    # ──────────────────────────────────────────────────────────
    @testset "WRMF" begin
        rng = MersenneTwister(42)
        X = sprand(rng, 100, 80, 0.05)

        @testset "Implicit Cholesky" begin
            model = WRMF(rank=5, λ=0.1, α=1.0, max_iter=5,
                         solver=CHOLESKY, feedback=IMPLICIT)
            fit!(model, X; rng=rng)

            @test model.is_fitted
            @test size(model.user_factors) == (5, 100)
            @test size(model.item_factors) == (5, 80)
            @test !any(isnan, model.user_factors)
            @test !any(isnan, model.item_factors)
        end

        @testset "Implicit CG" begin
            model = WRMF(rank=5, λ=0.1, α=1.0, max_iter=5,
                         solver=CONJUGATE_GRADIENT, feedback=IMPLICIT)
            fit!(model, X; rng=rng)

            @test model.is_fitted
            @test size(model.user_factors) == (5, 100)
            @test !any(isnan, model.user_factors)
        end

        @testset "Explicit" begin
            model = WRMF(rank=5, λ=0.1, α=1.0, max_iter=5,
                         solver=CHOLESKY, feedback=EXPLICIT)
            fit!(model, X; rng=rng)
            @test model.is_fitted
        end

        @testset "NNLS" begin
            model = WRMF(rank=5, λ=0.1, α=1.0, max_iter=3,
                         solver=NNLS, feedback=IMPLICIT)
            fit!(model, X; rng=rng)
            @test model.is_fitted
            @test all(model.user_factors .>= 0)
            @test all(model.item_factors .>= 0)
        end

        @testset "predict" begin
            model = WRMF(rank=5, λ=0.1, α=1.0, max_iter=3)
            fit!(model, X; rng=rng)
            preds = predict(model, X; k=5)
            @test size(preds) == (100, 5)
            @test all(preds .>= 1)
            @test all(preds .<= 80)
        end

        @testset "Empty sparse matrix" begin
            X_empty = sparse(Int[], Int[], Float64[], 10, 10)
            model = WRMF(rank=3, max_iter=2)
            fit!(model, X_empty; rng=rng)
            @test model.is_fitted
        end
    end

    # ──────────────────────────────────────────────────────────
    # FTRL
    # ──────────────────────────────────────────────────────────
    @testset "FTRL" begin
        rng = MersenneTwister(42)
        n, p = 500, 100
        X = sprand(rng, n, p, 0.1)
        # Generate simple linear signal
        w_true = zeros(p)
        w_true[1:5] .= 1.0
        y = Float64.(vec(X * w_true) .> 0.5)

        model = FTRL(learning_rate=0.1, λ=0.01, l1_ratio=0.5)
        fit!(model, X, y; n_iter=5)

        @test model.is_initialized
        @test model.n_features == p

        w = coef(model)
        @test length(w) == p
        @test !any(isnan, w)

        preds = predict(model, X)
        @test length(preds) == n
        @test all(0 .<= preds .<= 1)

        @testset "partial_fit! updates" begin
            model2 = FTRL(learning_rate=0.1)
            partial_fit!(model2, X, y)
            w1 = copy(coef(model2))
            partial_fit!(model2, X, y)
            w2 = coef(model2)
            @test w1 != w2  # weights should change after another epoch
        end
    end

    # ──────────────────────────────────────────────────────────
    # Factorization Machines
    # ──────────────────────────────────────────────────────────
    @testset "Factorization Machines" begin
        # Classic XOR test — FM should learn XOR
        x = sparse([0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0])
        y_xor = [0.0, 1.0, 1.0, 0.0]

        rng = MersenneTwister(42)
        fm = FactorizationMachine(
            learning_rate_w=10.0, rank=2,
            λ_w=0.0, λ_v=0.0, family=:binomial, intercept=true,
        )
        fit!(fm, x, y_xor; n_iter=200, rng=rng)

        preds = predict(fm, x)
        @test length(preds) == 4
        @test all(isfinite, preds)
        # FM should approximate XOR reasonably well
        @test preds[1] < 0.3  # (0,0) → 0
        @test preds[4] < 0.3  # (1,1) → 0
        @test preds[2] > 0.7  # (0,1) → 1
        @test preds[3] > 0.7  # (1,0) → 1

        @testset "Gaussian family" begin
            rng2 = MersenneTwister(42)
            fm_g = FactorizationMachine(rank=3, family=:gaussian, learning_rate_w=0.01)
            X = sprand(rng2, 50, 10, 0.3)
            y = randn(rng2, 50)
            fit!(fm_g, X, y; n_iter=5, rng=rng2)
            preds_g = predict(fm_g, X)
            @test length(preds_g) == 50
            @test all(isfinite, preds_g)
        end
    end

    # ──────────────────────────────────────────────────────────
    # GloVe
    # ──────────────────────────────────────────────────────────
    @testset "GloVe" begin
        rng = MersenneTwister(42)
        # Create a symmetric positive co-occurrence matrix
        n = 50
        A = sprand(rng, n, n, 0.1)
        A = A + A'   # make symmetric
        # Ensure all values are positive
        nz = nonzeros(A)
        nz .= abs.(nz) .+ 0.1

        model = GloVe(rank=10, x_max=10.0, learning_rate=0.15)
        fit!(model, A; n_iter=5, rng=rng)

        @test model.is_fitted
        @test size(model.W_main) == (10, n)
        @test size(model.W_ctx) == (10, n)
        @test length(model.cost_history) >= 1
        @test all(isfinite, model.cost_history)

        emb = get_embeddings(model)
        @test size(emb) == (10, n)
        @test all(isfinite, emb)
    end

    # ──────────────────────────────────────────────────────────
    # LMF
    # ──────────────────────────────────────────────────────────
    @testset "LMF" begin
        rng = MersenneTwister(42)
        X = sprand(rng, 80, 60, 0.05)

        model = LMF(rank=5, λ=0.01, α=1.0, learning_rate=0.01, max_iter=5)
        fit!(model, X; rng=rng)

        @test model.is_fitted
        @test size(model.user_factors) == (5, 80)
        @test size(model.item_factors) == (5, 60)
        @test !any(isnan, model.user_factors)

        preds = predict(model, X; k=5)
        @test size(preds) == (80, 5)
    end

    # ──────────────────────────────────────────────────────────
    # SoftImpute / SoftSVD
    # ──────────────────────────────────────────────────────────
    @testset "SoftImpute" begin
        rng = MersenneTwister(42)
        X = sprand(rng, 50, 40, 0.2)

        result = soft_impute(X; rank=5, λ=0.1, n_iter=20)
        @test result isa SoftImputeResult
        @test size(result.U, 2) <= 5
        @test length(result.d) <= 5
        @test all(result.d .>= 0)
        @test all(isfinite, result.U)
        @test all(isfinite, result.V)

        @testset "SoftSVD" begin
            result_svd = soft_svd(X; rank=5, λ=0.0, n_iter=20)
            @test result_svd isa SoftImputeResult
            @test all(result_svd.d .>= 0)
        end
    end

    # ──────────────────────────────────────────────────────────
    # Metrics
    # ──────────────────────────────────────────────────────────
    @testset "Metrics" begin
        # Simple known test case:
        # User 1: relevant items = {5, 7, 9}
        # Predictions: [5, 7, 9, 2]  → perfect ranking
        actual = sparse([1, 1, 1], [5, 7, 9], [1.0, 1.0, 1.0], 1, 10)
        predictions = [5 7 9 2]

        @testset "AP@K" begin
            ap = ap_at_k(predictions, actual; k=4)
            @test length(ap) == 1
            @test ap[1] ≈ 1.0
        end

        @testset "MAP@K" begin
            m = map_at_k(predictions, actual; k=4)
            @test m ≈ 1.0
        end

        @testset "NDCG@K" begin
            ndcg = ndcg_at_k(predictions, actual; k=4)
            @test length(ndcg) == 1
            @test ndcg[1] ≈ 1.0
        end

        @testset "Precision@K" begin
            prec = precision_at_k(predictions, actual; k=4)
            @test prec[1] ≈ 0.75   # 3 out of 4 are relevant
        end

        @testset "Recall@K" begin
            rec = recall_at_k(predictions, actual; k=4)
            @test rec[1] ≈ 1.0     # all 3 relevant items found
        end

        @testset "Imperfect ranking" begin
            # Predictions: [2, 5, 3, 7] → hits at pos 2 and 4
            preds_bad = [2 5 3 7]
            ap_bad = ap_at_k(preds_bad, actual; k=4)
            @test ap_bad[1] < 1.0
            @test ap_bad[1] > 0.0

            prec_bad = precision_at_k(preds_bad, actual; k=4)
            @test prec_bad[1] ≈ 0.5   # 2 out of 4
        end

        @testset "Multiple users" begin
            actual_multi = sparse(
                [1, 1, 2, 2],
                [1, 3, 2, 4],
                [1.0, 1.0, 1.0, 1.0],
                2, 5
            )
            preds_multi = [1 3; 2 4]
            ap_multi = ap_at_k(preds_multi, actual_multi; k=2)
            @test length(ap_multi) == 2
            @test all(ap_multi .≈ 1.0)
        end

        @testset "No relevant items" begin
            actual_empty = sparse(Int[], Int[], Float64[], 1, 10)
            preds_empty = [1 2 3]
            ap_empty = ap_at_k(preds_empty, actual_empty; k=3)
            @test ap_empty[1] ≈ 0.0
        end
    end

end
