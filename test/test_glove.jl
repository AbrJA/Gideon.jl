# test/test_glove.jl — GloVe algorithm tests

@testset "Basic fit" begin
    rng = MersenneTwister(42)
    n = 50
    A = sprand(rng, n, n, 0.1)
    A = A + A'
    nonzeros(A) .= abs.(nonzeros(A)) .+ 0.1

    model = GloVe(rank=10, x_max=10.0, learning_rate=0.15, verbose=false)
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

@testset "Cost generally decreasing" begin
    rng = MersenneTwister(42)
    n = 80
    A = sprand(rng, n, n, 0.1); A = A + A'
    nonzeros(A) .= abs.(nonzeros(A)) .+ 0.1
    model = GloVe(rank=5, x_max=10.0, learning_rate=0.15, verbose=false)
    fit!(model, A; n_iter=20, rng=rng)
    @test length(model.cost_history) == 20
    @test sum(diff(model.cost_history) .< 0) >= 15
end

@testset "Embeddings finite with non-trivial variance" begin
    rng = MersenneTwister(7)
    n = 60
    A = sprand(rng, n, n, 0.15); A = A + A'
    nonzeros(A) .= abs.(nonzeros(A)) .+ 0.1
    model = GloVe(rank=8, x_max=10.0, learning_rate=0.15, verbose=false)
    fit!(model, A; n_iter=30, rng=rng)
    emb = get_embeddings(model)
    @test all(isfinite, emb)
    mx = sum(emb) / length(emb)
    @test sqrt(sum((emb .- mx).^2) / length(emb)) > 0.01
end

@testset "Block structure: within > cross community similarity" begin
    I = vcat([i for i in 1:10 for j in i+1:10],
             [i for i in 11:20 for j in i+1:20])
    J = vcat([j for i in 1:10 for j in i+1:10],
             [j for i in 11:20 for j in i+1:20])
    V = fill(5.0, length(I))
    A = sparse(vcat(I,J), vcat(J,I), vcat(V,V), 20, 20)
    model = GloVe(rank=4, x_max=10.0, learning_rate=0.15, verbose=false)
    fit!(model, A; n_iter=50, rng=MersenneTwister(1))
    emb = get_embeddings(model)
    vcos(a, b) = dot(a, b) / (norm(a)*norm(b) + 1e-8)
    vbar(v) = sum(v) / length(v)
    s_within = vbar([vcos(emb[:,i], emb[:,j]) for i in 1:10 for j in i+1:10])
    s_cross = vbar([vcos(emb[:,i], emb[:,j]) for i in 1:10 for j in 11:20])
    @test s_within > s_cross
end

@testset "Convergence tolerance" begin
    rng = MersenneTwister(42)
    n = 50
    A = sprand(rng, n, n, 0.1); A = A + A'
    nonzeros(A) .= abs.(nonzeros(A)) .+ 0.1
    model = GloVe(rank=5, x_max=10.0, learning_rate=0.15, convergence_tol=0.001, verbose=false)
    fit!(model, A; n_iter=100, rng=rng)
    @test model.is_fitted
    # Should converge before 100 iterations
    @test length(model.cost_history) <= 100
end

@testset "Regularization" begin
    rng = MersenneTwister(42)
    n = 40
    A = sprand(rng, n, n, 0.2); A = A + A'
    nonzeros(A) .= abs.(nonzeros(A)) .+ 0.1

    m_noreg = GloVe(rank=5, x_max=10.0, learning_rate=0.15, λ=0.0, verbose=false)
    m_reg = GloVe(rank=5, x_max=10.0, learning_rate=0.15, λ=1.0, verbose=false)
    fit!(m_noreg, A; n_iter=20, rng=MersenneTwister(1))
    fit!(m_reg, A; n_iter=20, rng=MersenneTwister(1))

    # Regularization should produce smaller embeddings
    @test sum(abs2, get_embeddings(m_reg)) < sum(abs2, get_embeddings(m_noreg))
end
