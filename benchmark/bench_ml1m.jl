# benchmark/bench_ml1m.jl
# Quick Julia vs Python comparison on ML-1M
# Run: julia --project -t 16 benchmark/bench_ml1m.jl

using Gideon, SparseArrays, Random, LinearAlgebra, Printf, DelimitedFiles

println("=" ^ 70)
println("Julia (Gideon) Benchmark — MovieLens 1M")
println("=" ^ 70)
println("  Julia threads: $(Threads.nthreads())")
println("  BLAS threads:  $(BLAS.get_num_threads())")
println()

# ─────────────────────────────────────────────
# Load ML-1M
# ─────────────────────────────────────────────
function load_ml1m(path)
    println("[Loading ML-1M...]")
    t0 = time()
    lines = readlines(path)
    n = length(lines)
    users = Vector{Int}(undef, n)
    items = Vector{Int}(undef, n)
    for (idx, line) in enumerate(lines)
        parts = split(line, "::")
        users[idx] = parse(Int, parts[1])
        items[idx] = parse(Int, parts[2])
    end
    # Re-index to contiguous IDs
    u_map = Dict{Int,Int}()
    i_map = Dict{Int,Int}()
    for u in users
        haskey(u_map, u) || (u_map[u] = length(u_map) + 1)
    end
    for i in items
        haskey(i_map, i) || (i_map[i] = length(i_map) + 1)
    end
    rows = [u_map[u] for u in users]
    cols = [i_map[i] for i in items]
    X = sparse(rows, cols, ones(Float64, n), length(u_map), length(i_map))
    println("  Loaded in $(@sprintf("%.2f", time()-t0)) s")
    println("  Matrix: $(size(X, 1)) users × $(size(X, 2)) items, nnz=$(nnz(X))")
    X
end

X = load_ml1m("usage/ml-1m/ratings.dat")

# Train/test split: 80/20 per user
function train_test_split(X; test_frac=0.2, seed=42)
    rng = MersenneTwister(seed)
    m, n = size(X)
    train_rows, train_cols, train_vals = Int[], Int[], Float64[]
    test_rows, test_cols, test_vals = Int[], Int[], Float64[]
    for u in 1:m
        items_u = findnz(X[u, :])[1]
        n_test = max(1, round(Int, length(items_u) * test_frac))
        perm = shuffle(rng, items_u)
        test_items = Set(perm[1:n_test])
        for i in items_u
            if i in test_items
                push!(test_rows, u); push!(test_cols, i); push!(test_vals, 1.0)
            else
                push!(train_rows, u); push!(train_cols, i); push!(train_vals, 1.0)
            end
        end
    end
    train = sparse(train_rows, train_cols, train_vals, m, n)
    test = sparse(test_rows, test_cols, test_vals, m, n)
    train, test
end

println("[Splitting train/test...]")
X_train, X_test = train_test_split(X)
println("  Train: nnz=$(nnz(X_train)), Test: nnz=$(nnz(X_test))")
println()

# ─────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────
function ndcg_at_k(predictions::Matrix{Int}, test::SparseMatrixCSC, k::Int)
    n_users = size(predictions, 1)
    ndcgs = zeros(n_users)
    for u in 1:n_users
        relevant = Set(findnz(test[u, :])[1])
        isempty(relevant) && continue
        dcg = 0.0
        for rank in 1:min(k, size(predictions, 2))
            if predictions[u, rank] in relevant
                dcg += 1.0 / log2(rank + 1)
            end
        end
        ideal_hits = min(length(relevant), k)
        idcg = sum(1.0 / log2(i + 1) for i in 1:ideal_hits)
        ndcgs[u] = idcg > 0 ? dcg / idcg : 0.0
    end
    mean(ndcgs)
end

function recall_at_k(predictions::Matrix{Int}, test::SparseMatrixCSC, k::Int)
    n_users = size(predictions, 1)
    recalls = zeros(n_users)
    for u in 1:n_users
        relevant = Set(findnz(test[u, :])[1])
        isempty(relevant) && continue
        hits = count(predictions[u, r] in relevant for r in 1:min(k, size(predictions, 2)))
        recalls[u] = hits / length(relevant)
    end
    mean(recalls)
end

using Statistics: mean

# ─────────────────────────────────────────────
# Benchmark function
# ─────────────────────────────────────────────
function benchmark_bpr(X_train, X_test; rank=64, n_iter=50, lr=0.05, λ=0.01)
    println("─" ^ 70)
    println("BPR — rank=$rank, iters=$n_iter, lr=$lr")
    println("─" ^ 70)

    model = BPR(rank=rank, learning_rate=lr, max_iter=n_iter,
                λ_user=λ, λ_pos=λ, λ_neg=λ, verbose=true)

    # Warmup (1 iter)
    warmup = BPR(rank=rank, learning_rate=lr, max_iter=1,
                 λ_user=λ, λ_pos=λ, λ_neg=λ, verbose=false)
    fit!(warmup, X_train; rng=MersenneTwister(1))
    predict(warmup, X_train; k=10)
    GC.gc()

    # Training
    t_train = @elapsed fit!(model, X_train; rng=MersenneTwister(42))
    println("  Train time: $(@sprintf("%.2f", t_train)) s ($(@sprintf("%.3f", t_train/n_iter)) s/iter)")

    # Prediction
    t_pred = @elapsed preds = predict(model, X_train; k=10)
    println("  Predict time: $(@sprintf("%.2f", t_pred)) s")

    # Metrics
    ndcg = ndcg_at_k(preds, X_test, 10)
    rec = recall_at_k(preds, X_test, 10)
    println("  NDCG@10:   $(@sprintf("%.4f", ndcg))")
    println("  Recall@10: $(@sprintf("%.4f", rec))")
    println()

    return (train_time=t_train, predict_time=t_pred, ndcg=ndcg, recall=rec)
end

function benchmark_ials(X_train, X_test; rank=64, n_iter=15, α=40.0, λ=0.1, solver=ConjugateGradient())
    println("─" ^ 70)
    println("IALS — rank=$rank, iters=$n_iter, α=$α, solver=$solver")
    println("─" ^ 70)

    model = IALS(rank=rank, α=α, λ=λ, max_iter=n_iter, solver=solver, verbose=true)

    # Warmup
    warmup = IALS(rank=rank, α=α, λ=λ, max_iter=1, solver=solver, verbose=false)
    fit!(warmup, X_train; rng=MersenneTwister(1))
    predict(warmup, X_train; k=10)
    GC.gc()

    # Training
    t_train = @elapsed fit!(model, X_train; rng=MersenneTwister(42))
    println("  Train time: $(@sprintf("%.2f", t_train)) s ($(@sprintf("%.3f", t_train/n_iter)) s/iter)")

    # Prediction
    t_pred = @elapsed preds = predict(model, X_train; k=10)
    println("  Predict time: $(@sprintf("%.2f", t_pred)) s")

    # Metrics
    ndcg = ndcg_at_k(preds, X_test, 10)
    rec = recall_at_k(preds, X_test, 10)
    println("  NDCG@10:   $(@sprintf("%.4f", ndcg))")
    println("  Recall@10: $(@sprintf("%.4f", rec))")
    println()

    return (train_time=t_train, predict_time=t_pred, ndcg=ndcg, recall=rec)
end

# ─────────────────────────────────────────────
# Run benchmarks
# ─────────────────────────────────────────────
println("=" ^ 70)
println("BENCHMARKS")
println("=" ^ 70)

results = Dict{String, Any}()

for rank in [32, 64]
    results["BPR_r$rank"] = benchmark_bpr(X_train, X_test; rank=rank, n_iter=50)
end

for rank in [32, 64]
    results["iALS_CG_r$rank"] = benchmark_ials(X_train, X_test; rank=rank, solver=ConjugateGradient())
end

# Summary table
println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)
@printf("%-20s %8s %8s %8s %8s\n", "Algorithm", "Train(s)", "Pred(s)", "NDCG@10", "Recall@10")
@printf("%-20s %8s %8s %8s %8s\n", "-"^20, "-"^8, "-"^8, "-"^8, "-"^8)
for (name, r) in sort(collect(results))
    @printf("%-20s %8.2f %8.2f %8.4f %8.4f\n", name, r.train_time, r.predict_time, r.ndcg, r.recall)
end
