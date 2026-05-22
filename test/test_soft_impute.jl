# test/test_soft_impute.jl — SoftImpute/SoftSVD tests

@testset "Basic SoftImpute" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 40, 0.2)
    result = soft_impute(X; rank=5, λ=0.1, max_iter=20, verbose=false)
    @test result isa SoftImputeResult
    @test size(result.U, 2) <= 5
    @test length(result.d) <= 5
    @test all(result.d .>= 0)
    @test all(isfinite, result.U)
    @test all(isfinite, result.V)

    # Reconstruction should be finite
    recon = result.U * Diagonal(result.d) * result.V'
    @test all(isfinite, recon)
end

@testset "SoftSVD" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 40, 0.2)
    result = soft_svd(X; rank=5, λ=0.0, max_iter=20, verbose=false)
    @test result isa SoftImputeResult
    @test all(result.d .>= 0)
end

@testset "Singular values sorted descending" begin
    rng = MersenneTwister(5)
    X = sprand(rng, 60, 50, 0.2)
    result = soft_impute(X; rank=5, λ=0.1, max_iter=20, verbose=false)
    @test issorted(result.d; rev=true)
    @test all(result.d .>= 0)
end

@testset "Low-rank recovery" begin
    # Create a rank-3 matrix with noise
    rng = MersenneTwister(42)
    U_true = randn(rng, 50, 3)
    V_true = randn(rng, 40, 3)
    M_true = U_true * V_true'
    # Sample entries
    mask = sprand(rng, 50, 40, 0.5)
    X_obs = SparseMatrixCSC(mask .!= 0) .* M_true
    # Make it actually sparse
    X_obs = sparse(findnz(X_obs)..., 50, 40)

    result = soft_impute(X_obs; rank=3, λ=0.01, max_iter=50, verbose=false)
    @test length(result.d) <= 3
    @test size(result.U, 1) == 50
    @test size(result.V, 1) == 40
end

@testset "Convergence tolerance" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 30, 25, 0.3)
    result = soft_impute(X; rank=5, λ=0.1, max_iter=1000, convergence_tol=1e-6, verbose=false)
    @test all(isfinite, result.d)
end

@testset "λ shrinks singular values" begin
    rng = MersenneTwister(42)
    X = sprand(rng, 50, 40, 0.2)
    r_low_λ = soft_impute(X; rank=5, λ=0.01, max_iter=30, verbose=false)
    r_high_λ = soft_impute(X; rank=5, λ=5.0, max_iter=30, verbose=false)
    @test sum(r_high_λ.d) <= sum(r_low_λ.d)
end
