# benchmark/julia_huge_benchmark.jl
# Aggressive huge-matrix benchmark for Gideon.jl WRMF
# Compares Julia (multi-threaded) vs R rsparse across 5 scale levels
#
# Run from project root:
#   julia --project=. --threads=4,2 benchmark/julia_huge_benchmark.jl

using Gideon, SparseArrays, LinearAlgebra, Random, Printf

# ── Helpers ────────────────────────────────────────────────────────────────────

fmt_nnz(n::Int) = n >= 1_000_000 ? @sprintf("%.1fM", n/1e6) :
                  n >= 1_000     ? @sprintf("%.1fK", n/1e3)  : "$n"

function make_sparse_matrix(n_users::Int, n_items::Int, density::Float64; seed::Int=42)
    rng  = MersenneTwister(seed)
    nnz  = max(1, round(Int, density * n_users * n_items))
    rows = rand(rng, 1:n_users, nnz)
    cols = rand(rng, 1:n_items, nnz)
    vals = Float32.(rand(rng, 1:5, nnz))
    sparse(rows, cols, vals, n_users, n_items)
end

function load_triplet(path::String, dims_path::String)
    lines  = readlines(path)
    n      = length(lines) - 1
    rv     = Vector{Int}(undef, n)
    cv     = Vector{Int}(undef, n)
    vv     = Vector{Float32}(undef, n)
    for (i, line) in enumerate(lines[2:end])
        p = split(line, ',')
        rv[i] = parse(Int,     p[1])
        cv[i] = parse(Int,     p[2])
        vv[i] = parse(Float32, p[3])
    end
    dlines = readlines(dims_path)
    nr, nc = parse(Int, dlines[2]), parse(Int, dlines[3])
    sparse(rv, cv, vv, nr, nc)
end

function bench_wrmf(X::SparseMatrixCSC, solver::ALSSolver;
                    n_runs::Int=1, rank::Int=10,
                    lambda=0.1, alpha=1.0, n_iter::Int=10)
    minimum(
        @elapsed(fit!(WRMF(rank=rank, λ=lambda, α=alpha, max_iter=n_iter,
                           solver=solver, feedback=IMPLICIT),
                      X; rng=MersenneTwister(42)))
        for _ in 1:n_runs
    )
end

function load_r_timing(path::String)
    isfile(path) || return nothing
    lines = readlines(path)
    length(lines) < 2 && return nothing
    chol_t, cg_t = Float64[], Float64[]
    for line in lines[2:end]
        p = split(line, ',')
        length(p) < 5 && continue
        push!(chol_t, parse(Float64, p[4]))
        push!(cg_t,   parse(Float64, p[5]))
    end
    (chol=chol_t, cg=cg_t)
end

# ── Main ────────────────────────────────────────────────────────────────────────

println("=== Julia Gideon Huge-Matrix Benchmark ===")
println("Julia threads: $(Threads.nthreads()) default + $(Threads.nthreads(:interactive)) interactive")
println()

print("Warming up JIT... ")
let X_w = make_sparse_matrix(300, 200, 0.1; seed=1)
    for s in (CHOLESKY, CONJUGATE_GRADIENT)
        fit!(WRMF(rank=10, λ=0.1, α=1.0, max_iter=3, solver=s, feedback=IMPLICIT),
             X_w; rng=MersenneTwister(1))
    end
end
println("done.\n")

# Scale 1 — XLarge: 10K × 5K, density=1%
println("─── XLarge (10K × 5K, density=1%) ───")
X_xl = try
    X = load_triplet("/tmp/X_xlarge.csv", "/tmp/X_xlarge_dims.csv")
    println("  Loaded from R   nnz=$(fmt_nnz(nnz(X)))  size=$(size(X))")
    X
catch
    X = make_sparse_matrix(10_000, 5_000, 0.01)
    println("  Generated       nnz=$(fmt_nnz(nnz(X)))  size=$(size(X))")
    X
end
t_xl_chol = bench_wrmf(X_xl, CHOLESKY;           n_runs=3)
t_xl_cg   = bench_wrmf(X_xl, CONJUGATE_GRADIENT; n_runs=3)
println(@sprintf("  Cholesky: %.3f s   CG: %.3f s\n", t_xl_chol, t_xl_cg))

# Scale 2 — XXLarge: 50K × 10K, density=0.5%
println("─── XXLarge (50K × 10K, density=0.5%) ───")
X_xxl = try
    X = load_triplet("/tmp/X_xxlarge.csv", "/tmp/X_xxlarge_dims.csv")
    println("  Loaded from R   nnz=$(fmt_nnz(nnz(X)))  size=$(size(X))")
    X
catch
    X = make_sparse_matrix(50_000, 10_000, 0.005)
    println("  Generated       nnz=$(fmt_nnz(nnz(X)))  size=$(size(X))")
    X
end
t_xxl_chol = bench_wrmf(X_xxl, CHOLESKY;           n_runs=2)
t_xxl_cg   = bench_wrmf(X_xxl, CONJUGATE_GRADIENT; n_runs=2)
println(@sprintf("  Cholesky: %.3f s   CG: %.3f s\n", t_xxl_chol, t_xxl_cg))

# Scale 3 — Large3: 200K × 20K, density=0.1%
println("─── Large3 (200K × 20K, density=0.1%) ───")
X_l3 = make_sparse_matrix(200_000, 20_000, 0.001)
println("  Generated       nnz=$(fmt_nnz(nnz(X_l3)))  size=$(size(X_l3))")
t_l3_chol = bench_wrmf(X_l3, CHOLESKY;           n_runs=1)
t_l3_cg   = bench_wrmf(X_l3, CONJUGATE_GRADIENT; n_runs=1)
println(@sprintf("  Cholesky: %.3f s   CG: %.3f s\n", t_l3_chol, t_l3_cg))

# Scale 4 — Huge: 500K × 50K, density=0.05%
println("─── Huge (500K × 50K, density=0.05%) ───")
X_h = make_sparse_matrix(500_000, 50_000, 0.0005)
println("  Generated       nnz=$(fmt_nnz(nnz(X_h)))  size=$(size(X_h))")
t_h_chol = bench_wrmf(X_h, CHOLESKY;           n_runs=1)
t_h_cg   = bench_wrmf(X_h, CONJUGATE_GRADIENT; n_runs=1)
println(@sprintf("  Cholesky: %.3f s   CG: %.3f s\n", t_h_chol, t_h_cg))

# Scale 5 — MEGA: 1M × 100K, density=0.01%
println("─── MEGA (1M × 100K, density=0.01%) ───")
X_mega = make_sparse_matrix(1_000_000, 100_000, 0.0001)
println("  Generated       nnz=$(fmt_nnz(nnz(X_mega)))  size=$(size(X_mega))")
t_mega_chol = bench_wrmf(X_mega, CHOLESKY;           n_runs=1)
t_mega_cg   = bench_wrmf(X_mega, CONJUGATE_GRADIENT; n_runs=1)
println(@sprintf("  Cholesky: %.3f s   CG: %.3f s\n", t_mega_chol, t_mega_cg))

# ── Summary table ──────────────────────────────────────────────────────────────
labels     = ["XLarge  (10K×5K)",   "XXLarge (50K×10K)",
              "Large3  (200K×20K)", "Huge    (500K×50K)",  "MEGA    (1M×100K)"]
jl_chol    = [t_xl_chol, t_xxl_chol, t_l3_chol, t_h_chol, t_mega_chol]
jl_cg      = [t_xl_cg,   t_xxl_cg,   t_l3_cg,   t_h_cg,   t_mega_cg]
nnz_counts = [nnz(X_xl), nnz(X_xxl), nnz(X_l3), nnz(X_h), nnz(X_mega)]
r_timing   = load_r_timing("/tmp/r_huge_timing.csv")

println("=" ^ 82)
println("COMBINED TABLE — Gideon.jl vs R rsparse  ($(Threads.nthreads()) threads each)")
println("=" ^ 82)

for (solver_name, jl_t, r_key) in (
        ("CHOLESKY",          jl_chol, :chol),
        ("CONJUGATE GRADIENT", jl_cg,  :cg))
    r_t_vec = r_timing !== nothing ? getfield(r_timing, r_key) : fill(NaN, 5)
    println("\n  Solver: $solver_name")
    println(@sprintf("  %-22s  %9s  %8s  %8s  %8s",
                     "Scale", "Julia (s)", "R (s)", "Speedup", "nnz"))
    println("  " * "─"^62)
    for i in 1:5
        r_v = i <= length(r_t_vec) ? r_t_vec[i] : NaN
        spd = isnan(r_v) ? "   N/A  " : @sprintf("%6.1fx", r_v / jl_t[i])
        r_s = isnan(r_v) ? "   N/A  " : @sprintf("%8.3f", r_v)
        println(@sprintf("  %-22s  %9.3f  %s  %s  %8s",
                         labels[i], jl_t[i], r_s, spd, fmt_nnz(nnz_counts[i])))
    end
end
println()

# Save results
open("/tmp/julia_huge_timing.csv", "w") do f
    println(f, "size,nnz,jl_chol_s,jl_cg_s,n_threads")
    for i in 1:5
        println(f, "\"$(labels[i])\",$(nnz_counts[i]),$(jl_chol[i]),$(jl_cg[i]),$(Threads.nthreads())")
    end
end
println("Timings saved → /tmp/julia_huge_timing.csv")
println("=== Julia huge-matrix benchmark complete ===")
