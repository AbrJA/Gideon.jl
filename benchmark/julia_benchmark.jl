# benchmark/julia_benchmark.jl
# Julia Gideon validation + benchmarking vs R rsparse output
# Run from project root: julia --project=. benchmark/julia_benchmark.jl

using Gideon, SparseArrays, LinearAlgebra, Random, CSV, DataFrames, Statistics
using Printf

println("=== Julia Gideon Validation & Benchmark ===")

# ─────────────────────────────────────────────
# Helper: load triplet CSV → SparseMatrixCSC
# ─────────────────────────────────────────────
function load_sparse(path_triplet, path_dims)
    df   = CSV.read(path_triplet, DataFrame)
    dims = CSV.read(path_dims, DataFrame)
    n_rows = dims[1,1]
    n_cols = dims[2,1]
    sparse(df.row, df.col, Float64.(df.val), n_rows, n_cols)
end

function load_sparse_nosize(path_triplet)
    df = CSV.read(path_triplet, DataFrame)
    # infer size from max indices
    sparse(df.row, df.col, Float64.(df.val), maximum(df.row), maximum(df.col))
end

# ─────────────────────────────────────────────
# 1. Load shared matrices from R
# ─────────────────────────────────────────────
println("\nLoading R-generated matrices...")
X_small  = load_sparse("/tmp/X_small.csv",  "/tmp/X_small_dims.csv")
X_medium = load_sparse("/tmp/X_medium.csv", "/tmp/X_medium_dims.csv")
X_large  = load_sparse("/tmp/X_large.csv",  "/tmp/X_large_dims.csv")
println("  X_small:  $(size(X_small))")
println("  X_medium: $(size(X_medium))")
println("  X_large:  $(size(X_large))")

RANK = 10; LAMBDA = 0.1; ALPHA = 1.0; N_ITER = 10

# ─────────────────────────────────────────────
# 2. WMF — correctness vs R
# ─────────────────────────────────────────────
println("\n--- WMF Correctness vs R (small, CholeskySolver, 10 iter) ---")

# Load R reference factors
r_user_raw = CSV.read("/tmp/r_user_emb_small.csv", DataFrame)
r_item_raw = CSV.read("/tmp/r_item_emb_small.csv", DataFrame)
# R stores user embeddings as n_users × rank → transpose to rank × n_users
# R stores item embeddings (components) as rank × n_items → no transpose needed
R_user = Matrix{Float64}(r_user_raw)'  # rank × n_users
R_item = Matrix{Float64}(r_item_raw)   # rank × n_items

println("  R user factors shape (rank×n):  $(size(R_user))")
println("  R item factors shape (rank×n):  $(size(R_item))")
println("  R user F-norm: $(@sprintf("%.6f", norm(R_user)))")
println("  R item F-norm: $(@sprintf("%.6f", norm(R_item)))")

rng = MersenneTwister(42)
model_jl = WMF(rank=RANK, λ=LAMBDA, α=ALPHA, max_iter=N_ITER,
                solver=CholeskySolver(), feedback=IMPLICIT)
t_small_chol = @elapsed fit!(model_jl, X_small; rng=rng)

println("  Julia user factors shape: $(size(model_jl.user_factors))")
println("  Julia item factors shape: $(size(model_jl.item_factors))")
println("  Julia user F-norm: $(@sprintf("%.6f", norm(model_jl.user_factors)))")
println("  Julia item F-norm: $(@sprintf("%.6f", norm(model_jl.item_factors)))")
println("  Julia time (first run, no JIT): $(@sprintf("%.3f", t_small_chol)) s")

# Compare reconstructions — both should have similar R² on the non-zero entries
function reconstruction_r2(U, V, X)
    rv = rowvals(X); nz = nonzeros(X)
    ss_res = 0.0; ss_tot = 0.0
    mu = mean(nz)
    for j in axes(X, 2), idx in nzrange(X, j)
        i = rv[idx]
        pred = dot(@view(U[:, i]), @view(V[:, j]))
        ss_res += (nz[idx] - pred)^2
        ss_tot += (nz[idx] - mu)^2
    end
    1.0 - ss_res / ss_tot
end

r2_julia = reconstruction_r2(model_jl.user_factors, model_jl.item_factors, X_small)
r2_r     = reconstruction_r2(R_user, R_item, X_small)
println("  R² (Julia):  $(@sprintf("%.6f", r2_julia))")
println("  R² (R):      $(@sprintf("%.6f", r2_r))")
println("  R² diff:     $(@sprintf("%.6f", abs(r2_julia - r2_r)))")

# ─────────────────────────────────────────────
# 3. WMF Benchmarks — BenchmarkTools
# ─────────────────────────────────────────────
println("\n--- WMF Benchmarks ---")

# ─────────────────────────────────────────────
# 3. WMF Benchmarks — 3-run average
# ─────────────────────────────────────────────
println("\n--- WMF Benchmarks ---")

function bench_wrmf_elapsed(X, solver, n_iter=N_ITER, n_runs=3)
    times = Float64[]
    for _ in 1:n_runs
        t = @elapsed begin
            m = WMF(rank=RANK, λ=LAMBDA, α=ALPHA, max_iter=n_iter,
                     solver=solver, feedback=IMPLICIT)
            fit!(m, X; rng=MersenneTwister(42))
        end
        push!(times, t)
    end
    minimum(times)
end

t_jl_small_chol  = bench_wrmf_elapsed(X_small,  CholeskySolver())
t_jl_medium_chol = bench_wrmf_elapsed(X_medium, CholeskySolver())
t_jl_large_chol  = bench_wrmf_elapsed(X_large,  CholeskySolver(), N_ITER, 1)
t_jl_medium_cg   = bench_wrmf_elapsed(X_medium, ConjugateGradient())

println("  Small  ($(size(X_small)))   CholeskySolver: $(@sprintf("%.3f", t_jl_small_chol)) s")
println("  Medium ($(size(X_medium))) CholeskySolver: $(@sprintf("%.3f", t_jl_medium_chol)) s")
println("  Large  ($(size(X_large)))  CholeskySolver: $(@sprintf("%.3f", t_jl_large_chol)) s")
println("  Medium ($(size(X_medium))) CG:       $(@sprintf("%.3f", t_jl_medium_cg)) s")

# ─────────────────────────────────────────────
# 4. FTRL — correctness vs R
# ─────────────────────────────────────────────
println("\n--- FTRL Correctness vs R ---")
X_ftrl_df   = CSV.read("/tmp/X_ftrl.csv", DataFrame)
y_ftrl_df   = CSV.read("/tmp/y_ftrl.csv", DataFrame)
dims_ftrl   = CSV.read("/tmp/X_ftrl_dims.csv", DataFrame)
n_f, p_f    = dims_ftrl[1,1], dims_ftrl[2,1]
X_ftrl      = sparse(X_ftrl_df.row, X_ftrl_df.col, Float64.(X_ftrl_df.val), n_f, p_f)
y_ftrl      = Float64.(y_ftrl_df.y)

r_weights   = CSV.read("/tmp/r_ftrl_weights.csv", DataFrame).w
r_preds_ftrl = CSV.read("/tmp/r_ftrl_preds.csv", DataFrame).p

rng2 = MersenneTwister(42)
ftrl_jl = FTRL(learning_rate=0.1, learning_rate_decay=0.5, λ=0.01, l1_ratio=0.5)
t_ftrl_jl = @elapsed for _ in 1:5
    partial_fit!(ftrl_jl, X_ftrl, y_ftrl; rng=rng2)
end

w_jl   = coef(ftrl_jl)
p_jl   = predict(ftrl_jl, X_ftrl)
acc_jl = mean(round.(p_jl) .== y_ftrl)
acc_r  = mean(round.(r_preds_ftrl) .== y_ftrl)

println("  Julia time (5 epochs, $(size(X_ftrl))): $(@sprintf("%.3f", t_ftrl_jl)) s")
println("  Julia NNZ weights: $(sum(abs.(w_jl) .> 1e-10)) / $p_f")
println("  Julia Accuracy: $(@sprintf("%.4f", acc_jl))")
println("  R     Accuracy: $(@sprintf("%.4f", acc_r))")
println("  Weight correlation (Julia vs R): $(@sprintf("%.6f", cor(w_jl, r_weights)))")
println("  Prediction correlation: $(@sprintf("%.6f", cor(p_jl, r_preds_ftrl)))")

t_ftrl_bench = let ts = Float64[]
    for _ in 1:3
        push!(ts, @elapsed begin
            m = FTRL(learning_rate=0.1, learning_rate_decay=0.5, λ=0.01, l1_ratio=0.5)
            for _ in 1:5; partial_fit!(m, X_ftrl, y_ftrl; rng=MersenneTwister(42)); end
        end)
    end
    minimum(ts)
end
let s = @sprintf("%.3f", t_ftrl_bench); println("  Julia min-3 time: $s s"); end

# ─────────────────────────────────────────────
# 5. FM — XOR
# ─────────────────────────────────────────────
println("\n--- FM (XOR) ---")
x_xor = sparse([0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0])
y_xor = [0.0, 1.0, 1.0, 0.0]

rng3 = MersenneTwister(42)
fm_jl = FM(learning_rate_w=10.0, rank=2, λ_w=0.0, λ_v=0.0,
                             family=:binomial, intercept=true, learning_rate_v=10.0)
t_fm_jl = @elapsed fit!(fm_jl, x_xor, y_xor; n_iter=200, rng=rng3)
p_fm_jl = predict(fm_jl, x_xor)
xor_correct = p_fm_jl[1] < 0.3 && p_fm_jl[4] < 0.3 && p_fm_jl[2] > 0.7 && p_fm_jl[3] > 0.7
println("  Time (200 iter): $(@sprintf("%.3f", t_fm_jl)) s")
println("  Predictions: $(@sprintf("%.4f  %.4f  %.4f  %.4f", p_fm_jl[1], p_fm_jl[2], p_fm_jl[3], p_fm_jl[4]))")
println("  XOR correct: $xor_correct")

t_fm_bench = let ts = Float64[]
    for _ in 1:5
        push!(ts, @elapsed begin
            m = FM(learning_rate_w=10.0, rank=2, λ_w=0.0, λ_v=0.0,
                                     family=:binomial, intercept=true, learning_rate_v=10.0)
            fit!(m, x_xor, y_xor; n_iter=200, rng=MersenneTwister(42))
        end)
    end
    minimum(ts)
end
let s = @sprintf("%.3f", t_fm_bench); println("  Min-5 time: $s s"); end

# ─────────────────────────────────────────────
# 6. GloVe — timing
# ─────────────────────────────────────────────
println("\n--- GloVe ---")
rng4 = MersenneTwister(42)
A = sprand(rng4, 200, 200, 0.1)
A = A + A'
nz = nonzeros(A); nz .= abs.(nz) .+ 0.1

rng5 = MersenneTwister(42)
glove_jl = GloVe(rank=20, x_max=10.0, learning_rate=0.15)
t_glove_jl = @elapsed fit!(glove_jl, A; n_iter=10, rng=rng5)
println("  Time (10 iter, 200×200): $(@sprintf("%.3f", t_glove_jl)) s")
println("  Cost history length: $(length(glove_jl.loss_history))")
println("  Final cost: $(@sprintf("%.6f", glove_jl.loss_history[end]))")

t_glove_bench = let ts = Float64[]
    for _ in 1:3
        push!(ts, @elapsed begin
            m = GloVe(rank=20, x_max=10.0, learning_rate=0.15)
            fit!(m, A; n_iter=10, rng=MersenneTwister(42))
        end)
    end
    minimum(ts)
end
let s = @sprintf("%.3f", t_glove_bench); println("  Min-3 time: $s s"); end

# ─────────────────────────────────────────────
# 7. SoftImpute
# ─────────────────────────────────────────────
println("\n--- SoftImpute ---")
rng6 = MersenneTwister(42)
X_si = sprand(rng6, 200, 150, 0.1)
t_si = @elapsed soft_impute(X_si; rank=10, λ=0.1, n_iter=20)
t_si_bench = minimum(@elapsed(soft_impute(X_si; rank=10, λ=0.1, n_iter=20)) for _ ∈ 1:3)
println("  Time (200×150, rank=10, 20 iter): $(@sprintf("%.3f", t_si)) s")
println("  Min-3 time: $(@sprintf("%.3f", t_si_bench)) s")

# ─────────────────────────────────────────────
# 8. Metrics
# ─────────────────────────────────────────────
println("\n--- Metrics Correctness ---")
actual_met = sparse([1,1,1], [5,7,9], [1.0,1.0,1.0], 1, 10)
preds_met  = [5 7 9 2]
ap_val   = ap_at_k(preds_met, actual_met; k=4)[1]
ndcg_val = ndcg_at_k(preds_met, actual_met; k=4)[1]
prec_val = precision_at_k(preds_met, actual_met; k=4)[1]
rec_val  = recall_at_k(preds_met, actual_met; k=4)[1]
println("  AP@4 (perfect):  $(@sprintf("%.6f", ap_val))   (expected 1.0)")
println("  NDCG@4 (perfect): $(@sprintf("%.6f", ndcg_val))  (expected 1.0)")
println("  Prec@4:  $(@sprintf("%.6f", prec_val))   (expected 0.75)")
println("  Recall@4: $(@sprintf("%.6f", rec_val))    (expected 1.0)")

# ─────────────────────────────────────────────
# 9. Save Julia timing summary
# ─────────────────────────────────────────────
r_timing = CSV.read("/tmp/r_timing.csv", DataFrame)

julia_times = [t_jl_small_chol, t_jl_medium_chol, t_jl_large_chol,
               t_jl_medium_cg, t_ftrl_bench, t_fm_bench, t_glove_bench]
r_times    = r_timing.r_time_s

df_timing = DataFrame(
    algorithm   = r_timing.algorithm,
    size        = r_timing.size,
    r_time_s    = r_times,
    julia_time_s = julia_times,
    speedup     = r_times ./ julia_times
)

CSV.write("/tmp/julia_timing.csv", df_timing)
println("\n\nTiming Summary:")
println(df_timing)
println("\n=== Julia benchmark complete ===")
