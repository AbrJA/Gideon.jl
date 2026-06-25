# benchmark/benchmark_comparison.jl
# Julia vs Python comparison across 3 matrix sizes.
#
# Run Julia:
#   julia --project -t8 benchmark/benchmark_comparison.jl
#
# Run Python:
#   python benchmark/benchmark_comparison.py
#
# Both scripts output a CSV results file for comparison.

using Gideon, SparseArrays, Random, LinearAlgebra, Printf, Statistics

println("=" ^ 70)
println("Gideon.jl Performance Benchmark")
println("=" ^ 70)
println("  Julia version: $(VERSION)")
println("  Threads: $(Threads.nthreads())")
println("  BLAS threads: $(BLAS.get_num_threads())")
println()

# ─────────────────────────────────────────────
# Generate synthetic sparse matrices at 3 scales
# ─────────────────────────────────────────────
function generate_matrix(n_users::Int, n_items::Int, density::Float64; seed::Int=42)
    rng = MersenneTwister(seed)
    nnz_target = round(Int, n_users * n_items * density)
    rows = rand(rng, 1:n_users, nnz_target)
    cols = rand(rng, 1:n_items, nnz_target)
    vals = ones(Float64, nnz_target)
    X = sparse(rows, cols, vals, n_users, n_items)
    X
end

# Three scales: hundreds, thousands, millions of interactions
const MATRIX_CONFIGS = [
    (name="hundreds",   n_users=500,     n_items=300,     density=0.10),   # ~15K nnz
    (name="thousands",  n_users=5_000,   n_items=3_000,   density=0.02),   # ~300K nnz
    (name="millions",   n_users=100_000, n_items=50_000,  density=0.001),  # ~5M nnz
]

# Algorithms to benchmark with their configurations
function get_benchmarks(X)
    n_users, n_items = size(X)
    [
        (name="ALS", model=WMF(rank=64, λ=0.1, α=40.0, max_iter=10,
            solver=CholeskySolver(), verbose=false)),
        (name="ALS-CG", model=WMF(rank=64, λ=0.1, α=40.0, max_iter=10,
            solver=ConjugateGradient(), cg_steps=3, verbose=false)),
        (name="BPR", model=BPR(rank=64, λ_user=0.01, λ_pos=0.01, λ_neg=0.01,
            learning_rate=0.05, max_iter=10, verbose=false)),
        (name="LogisticMF", model=LogisticMF(rank=64, λ=0.6, learning_rate=1.0,
            max_iter=10, n_negative=30, verbose=false)),
    ]
end

# ─────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────
struct BenchResult
    matrix_name::String
    n_users::Int
    n_items::Int
    nnz::Int
    algorithm::String
    time_seconds::Float64
    n_iters::Int
end

function benchmark_algorithm(model, X; rng=MersenneTwister(42))
    # Warmup (needed for JIT on first call of each model type)
    X_tiny = X[1:min(10, size(X,1)), 1:min(10, size(X,2))]
    try
        m_warmup = deepcopy(model)
        fit!(m_warmup, X_tiny; rng=MersenneTwister(1))
    catch
        # Some models might fail on tiny matrices; that's fine
    end

    # Actual benchmark
    m = deepcopy(model)
    t0 = time_ns()
    fit!(m, X; rng=rng)
    elapsed = (time_ns() - t0) / 1e9
    return elapsed
end

function run_benchmarks()
    results = BenchResult[]

    for cfg in MATRIX_CONFIGS
        println("─" ^ 50)
        @printf("Matrix: %s (%d × %d, density=%.3f)\n",
                cfg.name, cfg.n_users, cfg.n_items, cfg.density)
        println("─" ^ 50)

        X = generate_matrix(cfg.n_users, cfg.n_items, cfg.density)
        @printf("  Generated: %d users × %d items, nnz=%d\n",
                size(X, 1), size(X, 2), nnz(X))

        benchmarks = get_benchmarks(X)

        for b in benchmarks
            # Skip EASE for large matrix (O(n_items³) is too slow)
            if b.name == "EASE" && cfg.n_items > 5_000
                @printf("  %-15s  SKIPPED (n_items=%d too large for O(n³))\n",
                        b.name, cfg.n_items)
                continue
            end
            # Skip SLIM for large matrix (per-item regression)
            if b.name == "SLIM" && cfg.n_items > 5_000
                @printf("  %-15s  SKIPPED (n_items=%d too large)\n",
                        b.name, cfg.n_items)
                continue
            end

            GC.gc()
            elapsed = benchmark_algorithm(b.model, X)
            @printf("  %-15s  %8.3f s\n", b.name, elapsed)

            push!(results, BenchResult(
                cfg.name, cfg.n_users, cfg.n_items, nnz(X),
                b.name, elapsed, 10
            ))
        end
        println()
    end

    return results
end

# ─────────────────────────────────────────────
# Run and save results
# ─────────────────────────────────────────────
results = run_benchmarks()

# Write CSV output
outpath = joinpath(@__DIR__, "results_julia.csv")
open(outpath, "w") do io
    println(io, "matrix,n_users,n_items,nnz,algorithm,time_seconds,n_iters")
    for r in results
        @printf(io, "%s,%d,%d,%d,%s,%.4f,%d\n",
                r.matrix_name, r.n_users, r.n_items, r.nnz,
                r.algorithm, r.time_seconds, r.n_iters)
    end
end

println("=" ^ 70)
println("Results saved to: $outpath")
println("=" ^ 70)

# Summary table
println("\nSummary:")
println("─" ^ 60)
@printf("%-8s %-15s %10s\n", "Matrix", "Algorithm", "Time (s)")
println("─" ^ 60)
for r in results
    @printf("%-8s %-15s %10.3f\n", r.matrix_name, r.algorithm, r.time_seconds)
end
println("─" ^ 60)
