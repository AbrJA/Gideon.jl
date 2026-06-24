# test/test_utils.jl — Type hierarchy, utilities, sparse utils

@testset "Type Hierarchy" begin
    @test WeightedMatrixFactorization <: AbstractMatrixFactorization
    @test WeightedMatrixFactorization <: AbstractRecommender
    @test WeightedMatrixFactorization <: AbstractSparseModel
    @test GlobalVectors <: AbstractMatrixFactorization
    @test GlobalVectors <: AbstractRecommender
    @test LogisticMatrixFactorization <: AbstractMatrixFactorization
    @test BayesianPersonalizedRanking <: AbstractMatrixFactorization
    @test ImplicitALS <: AbstractMatrixFactorization
    @test ElementwiseALS <: AbstractMatrixFactorization
    @test ShallowAutoencoder <: AbstractItemSimilarity
    @test ShallowAutoencoder <: AbstractRecommender
    @test SparseLinearModel <: AbstractItemSimilarity
    @test SparseLinearModel <: AbstractRecommender
    @test OnlineRegressor <: AbstractSparseRegression
    @test OnlineRegressor <: AbstractSparseModel
    @test FactorizationMachine <: AbstractSparseRegression
    # Verify AbstractRecommender is NOT a parent of regression models
    @test !(OnlineRegressor <: AbstractRecommender)
    @test !(FactorizationMachine <: AbstractRecommender)
end

@testset "Sigmoid" begin
    @test Gideon.sigmoid(0.0) ≈ 0.5
    @test Gideon.sigmoid(100.0) ≈ 1.0 atol=1e-10
    @test Gideon.sigmoid(-100.0) ≈ 0.0 atol=1e-10
    @test Gideon.sigmoid(1.0) ≈ 1 / (1 + exp(-1.0))
    # Numerical stability at extremes
    @test isfinite(Gideon.sigmoid(1000.0))
    @test isfinite(Gideon.sigmoid(-1000.0))
end

@testset "link_function" begin
    @test Gideon.link_function(BINOMIAL, 0.0) ≈ 0.5
    @test Gideon.link_function(GAUSSIAN, 1.5) ≈ 1.5
    @test Gideon.link_function(POISSON, 0.0) ≈ 1.0
    @test Gideon.link_function(POISSON, 1.0) ≈ exp(1.0)
end

@testset "init_factors" begin
    rng = MersenneTwister(42)
    F = init_factors(rng, 5, 10)
    @test size(F) == (5, 10)
    @test all(isfinite, F)
    @test maximum(abs, F) < 0.1  # scale=0.01
end

@testset "Sparse Utils" begin
    rng = MersenneTwister(42)
    A = sprand(rng, 50, 30, 0.1)

    @testset "row norms" begin
        norms = sparse_row_norms(A, 2)
        @test length(norms) == 50
        @test all(norms .>= 0)
        # Verify against dense computation
        A_dense = Matrix(A)
        for i in 1:50
            @test norms[i] ≈ norm(A_dense[i, :]) atol=1e-10
        end
    end

    @testset "L1 norms" begin
        norms1 = sparse_row_norms(A, 1)
        A_dense = Matrix(A)
        for i in 1:50
            @test norms1[i] ≈ sum(abs, A_dense[i, :]) atol=1e-10
        end
    end

    @testset "col nnz" begin
        cnnz = sparse_col_nnz(A)
        @test length(cnnz) == 30
        @test sum(cnnz) == nnz(A)
    end

    @testset "row nnz" begin
        rnnz = sparse_row_nnz(A)
        @test length(rnnz) == 50
        @test sum(rnnz) == nnz(A)
    end

    @testset "dual representation" begin
        A_csc, At = dual_representation(A)
        @test size(At) == (30, 50)
        @test A_csc === A
        @test Matrix(At) ≈ Matrix(A')
    end

    @testset "to_csr" begin
        csr = to_csr(A)
        @test size(csr) == size(A)
    end

    @testset "empty matrix" begin
        empty = sparse(Int[], Int[], Float64[], 10, 5)
        @test sparse_row_norms(empty) == zeros(10)
        @test sparse_col_nnz(empty) == zeros(Int, 5)
        @test sparse_row_nnz(empty) == zeros(Int, 10)
    end
end
