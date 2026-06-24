using Test
using Gideon
using SparseArrays
using LinearAlgebra
using Random
using Aqua
using JET
using Pkg

@testset "Gideon.jl" begin
    @testset "Quality" begin
        include("test_quality.jl")
    end
    @testset "Types & Utils" begin
        include("test_utils.jl")
    end
    @testset "WeightedMatrixFactorization" begin
        include("test_wrmf.jl")
    end
    @testset "iALS" begin
        include("test_ials.jl")
    end
    @testset "eALS" begin
        include("test_eals.jl")
    end
    @testset "OnlineRegressor" begin
        include("test_ftrl.jl")
    end
    @testset "FM" begin
        include("test_fm.jl")
    end
    @testset "GlobalVectors" begin
        include("test_glove.jl")
    end
    @testset "LogisticMatrixFactorization" begin
        include("test_lmf.jl")
    end
    @testset "BayesianPersonalizedRanking" begin
        include("test_bpr.jl")
    end
    @testset "ShallowAutoencoder" begin
        include("test_ease.jl")
    end
    @testset "SparseLinearModel" begin
        include("test_slim.jl")
    end
    @testset "SoftImpute" begin
        include("test_soft_impute.jl")
    end
    @testset "Metrics" begin
        include("test_metrics.jl")
    end
    @testset "Infrastructure" begin
        include("test_infrastructure.jl")
    end
    @testset "Tables" begin
        include("test_tables.jl")
    end
    @testset "GPU" begin
        include("test_gpu.jl")
    end
    @testset "Coverage" begin
        include("test_coverage.jl")
    end
    @testset "R Correctness" begin
        include("test_r_correctness.jl")
    end
end
