# GloVe on text8 — Gideon (Julia)
# Run as script:  julia --project=. download.jl
# Or in REPL:     include("download.jl")  then call main()

using Downloads, SparseArrays, Random, LinearAlgebra, Printf
using Gideon

function ensure_text8(path)
    if !isfile(path)
        println("Downloading text8...")
        Downloads.download("http://mattmahoney.net/dc/text8.zip", path * ".zip")
        run(`unzip -o $(path * ".zip") -d $(dirname(path))`)
    end
end

function build_vocab(tokens; min_count = 5)
    counts = Dict{String, Int}()
    for t in tokens
        counts[t] = get(counts, t, 0) + 1
    end
    vocab = sort([w for (w, c) in counts if c >= min_count])
    word_to_id = Dict(w => i for (i, w) in enumerate(vocab))
    return vocab, word_to_id
end

function build_tcm(tokens, word_to_id; window = 5)
    ids = [get(word_to_id, t, 0) for t in tokens]
    V   = length(word_to_id)
    I = Int[]; J = Int[]; W = Float64[]
    for c in eachindex(ids)
        i = ids[c]; i == 0 && continue
        for d in 1:window
            c + d > length(ids) && break
            j = ids[c + d]; j == 0 && continue
            push!(I, i, j); push!(J, j, i); push!(W, 1/d, 1/d)
        end
    end
    return sparse(I, J, W, V, V)
end

function nearest(word, E, word_to_id, id_to_word; k = 5)
    id = get(word_to_id, word, 0)
    id == 0 && return String[]
    v    = E[:, id]
    sims = [dot(E[:, j], v) / (norm(E[:, j]) * norm(v) + 1e-12) for j in axes(E, 2)]
    sims[id] = -Inf
    return id_to_word[sortperm(sims, rev = true)[1:k]]
end

# Julia REPL — after include("download.jl"); res = main()
function analogy(a, b, c, E, word_to_id, id_to_word; k = 5)
    ids = [get(word_to_id, w, 0) for w in (a, b, c)]
    any(==(0), ids) && return String[]
    v = E[:, ids[2]] - E[:, ids[1]] + E[:, ids[3]]
    sims = [dot(E[:, j], v) / (norm(E[:, j]) * norm(v) + 1e-12) for j in axes(E, 2)]
    foreach(i -> sims[i] = -Inf, ids)
    id_to_word[sortperm(sims, rev = true)[1:k]]
end

# Julia
function wordsim_spearman(pairs, human_scores, E, word_to_id)
    model_sims = Float64[]
    valid_human = Float64[]
    for (i, (w1, w2)) in enumerate(pairs)
        i1, i2 = get(word_to_id, w1, 0), get(word_to_id, w2, 0)
        (i1 == 0 || i2 == 0) && continue
        v1, v2 = E[:, i1], E[:, i2]
        push!(model_sims, dot(v1, v2) / (norm(v1) * norm(v2) + 1e-12))
        push!(valid_human, human_scores[i])
    end
    # Spearman = Pearson on ranks
    r1 = invperm(sortperm(model_sims)) .|> Float64
    r2 = invperm(sortperm(valid_human)) .|> Float64
    dot(r1 .- mean(r1), r2 .- mean(r2)) / (std(r1) * std(r2) * (length(r1) - 1))
end

function main()
    text8 = expanduser("~/text8")
    ensure_text8(text8)

    wiki = open(readline, text8)
    tokens = split(wiki)

    id_to_word, word_to_id = build_vocab(tokens)
    C = build_tcm(tokens, word_to_id)
    @printf "vocab: %d words  |  co-occ nnz: %d\n" length(word_to_id) nnz(C)

    model = GloVe(rank = 50, x_max = 10.0, learning_rate = 0.15, verbose = true)
    fit!(model, C; n_iter = 20, rng = MersenneTwister(42))
    E = get_embeddings(model)   # rank × vocab

    println("\n── Nearest neighbors ──────────────────────────────────────────")
    for w in ("paris", "london", "king", "computer")
        haskey(word_to_id, w) || continue
        @printf "%-10s  %s\n" w join(nearest(w, E, word_to_id, id_to_word), "  ")
    end

    println("\n── Analogies ────────────────────────────────────────────────")
    analogy("man", "king", "woman", E, word_to_id, id_to_word)   # expect: queen
    analogy("paris", "france", "berlin", E, word_to_id, id_to_word)  # expect: germany

    return (; model, E, word_to_id, id_to_word, C)
end

main()
