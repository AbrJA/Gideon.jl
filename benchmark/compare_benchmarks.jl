# benchmark/compare_benchmarks.jl
# Compare Julia vs Python benchmark results.
#
# Usage: julia benchmark/compare_benchmarks.jl
#
# Expects results_julia.csv and results_python.csv in the benchmark/ directory.

using Printf, DelimitedFiles

const BENCH_DIR = @__DIR__

function load_csv(path::String)
    if !isfile(path)
        error("File not found: $path\nRun the benchmark first.")
    end
    lines = readlines(path)
    header = split(lines[1], ',')
    rows = []
    for line in lines[2:end]
        isempty(strip(line)) && continue
        parts = split(line, ',')
        push!(rows, Dict(zip(header, parts)))
    end
    rows
end

function main()
    jl_path = joinpath(BENCH_DIR, "results_julia.csv")
    py_path = joinpath(BENCH_DIR, "results_python.csv")

    jl_results = load_csv(jl_path)
    py_results = load_csv(py_path)

    # Build lookup: (matrix, algo_family) -> time
    # Map Python names to Julia equivalents
    algo_map = Dict(
        "ALS" => "ALS",
        "ALS-CG" => "ALS-CG",
        "BPR" => "BPR",
        "LogisticMF" => "LogisticMF",
    )

    py_lookup = Dict{Tuple{String,String}, Float64}()
    for r in py_results
        key = (r["matrix"], r["algorithm"])
        py_lookup[key] = parse(Float64, r["time_seconds"])
    end

    println("=" ^ 75)
    println("        Gideon.jl vs Python (implicit) Benchmark Comparison")
    println("=" ^ 75)
    println()
    @printf("%-8s %-15s %10s %10s %10s\n",
            "Matrix", "Algorithm", "Julia (s)", "Python (s)", "Speedup")
    println("─" ^ 75)

    for r in jl_results
        matrix = r["matrix"]
        jl_algo = r["algorithm"]
        jl_time = parse(Float64, r["time_seconds"])

        # Find matching Python result
        py_time = nothing
        for (py_algo, mapped) in algo_map
            if mapped == jl_algo
                key = (matrix, py_algo)
                if haskey(py_lookup, key)
                    py_time = py_lookup[key]
                    break
                end
            end
        end

        if py_time !== nothing
            speedup = py_time / jl_time
            color = speedup >= 1.0 ? "✓" : "✗"
            @printf("%-8s %-15s %10.3f %10.3f %9.2fx %s\n",
                    matrix, jl_algo, jl_time, py_time, speedup, color)
        else
            @printf("%-8s %-15s %10.3f %10s %10s\n",
                    matrix, jl_algo, jl_time, "—", "—")
        end
    end
    println("─" ^ 75)
    println()
    println("✓ = Julia faster, ✗ = Python faster")
end

main()
