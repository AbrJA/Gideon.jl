using Test
using Gideon
using SparseArrays
using LinearAlgebra
using Random
using Aqua
using JET

@testset "Gideon.jl" begin
    @testset "Quality" begin
        include("test_quality.jl")
    end
    @testset "Types & Utils" begin
        include("test_utils.jl")
    end
    @testset "WRMF" begin
        include("test_wrmf.jl")
    end
    @testset "FTRL" begin
        include("test_ftrl.jl")
    end
    @testset "FM" begin
        include("test_fm.jl")
    end
    @testset "GloVe" begin
        include("test_glove.jl")
    end
    @testset "LMF" begin
        include("test_lmf.jl")
    end
    @testset "SoftImpute" begin
        include("test_soft_impute.jl")
    end
    @testset "Metrics" begin
        include("test_metrics.jl")
    end
    @testset "R Correctness" begin
        include("test_r_correctness.jl")
    end
end
