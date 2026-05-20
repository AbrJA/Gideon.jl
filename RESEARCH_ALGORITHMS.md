# Research Algorithms Roadmap: VAE-CF, LightGCN, Contrastive Learning, Causal Recommendation

## Overview

This document outlines how to integrate four research-grade recommendation algorithms into Gideon.jl. Each algorithm follows Gideon's architecture: a `mutable struct <: AbstractSparseModel` with `fit!()`, `predict()`, and optionally `predict_scores()`.

**Dependencies needed:**
- [Flux.jl](https://github.com/FluxML/Flux.jl) — neural network framework (for VAE-CF, LightGCN, Contrastive)
- [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl) — bipartite graph ops (for LightGCN)
- [Zygote.jl](https://github.com/FluxML/Zygote.jl) — autodiff (comes with Flux)

All should be weak dependencies (extensions), similar to `GideonCUDAExt.jl`.

---

## 1. VAE-CF (Variational Autoencoder for Collaborative Filtering)

**Paper:** Liang et al., "Variational Autoencoders for Collaborative Filtering" (WWW 2018)

### Architecture

```
Input: x_u ∈ ℝⁿ (user's interaction row, n = n_items)
       ↓
  Encoder: x → μ, log(σ²)   [MLP: n → 600 → 2×k]
       ↓
  z ~ N(μ, σ²I)             [Reparameterization trick]
       ↓
  Decoder: z → x̂_u          [MLP: k → 600 → n, multinomial likelihood]
       ↓
  Loss: -E_q[log p(x|z)] + β·KL(q(z|x) || p(z))
```

### Gideon Integration

```julia
# src/algorithms/vae_cf.jl

mutable struct VAECF{T<:AbstractFloat} <: AbstractMatrixFactorization
    # Hyperparameters
    latent_dim::Int          # k (latent dimension, e.g., 200)
    hidden_dim::Int          # encoder/decoder hidden layer (e.g., 600)
    β::T                     # KL annealing weight (0→1 over training)
    dropout::T               # input dropout rate (e.g., 0.5)
    learning_rate::T
    max_iter::Int            # epochs
    batch_size::Int
    convergence_tol::T
    verbose::Bool

    # Learned parameters (Flux models stored after training)
    encoder_μ::Any           # Flux Chain
    encoder_logvar::Any      # Flux Chain
    decoder::Any             # Flux Chain
    is_fitted::Bool
end
```

### Key Implementation Notes

1. **Input**: Each user row `x_u` of the sparse matrix → dense vector (only during batch loading)
2. **Multinomial likelihood**: Use log-softmax + elementwise multiply by x_u
3. **KL annealing**: Start β=0, linearly increase to 1 over first N epochs
4. **Dropout on input**: Critical for performance (prevents copying)
5. **Predict**: Decode from z=μ (no sampling at test time), mask seen items

### Benchmark Comparison

VAE-CF typically outperforms ALS/BPR on sparse implicit feedback datasets (MovieLens, Netflix) by 3-8% NDCG@100. Main competitor: EASE (which is much simpler but surprisingly competitive).

---

## 2. LightGCN (Light Graph Convolution Network)

**Paper:** He et al., "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation" (SIGIR 2020)

### Architecture

```
User-Item Bipartite Graph:
  Users: u₁, u₂, ..., uₘ  (embeddings E_u ∈ ℝᵏ)
  Items: i₁, i₂, ..., iₙ  (embeddings E_i ∈ ℝᵏ)

Layer propagation (no feature transform, no activation):
  e_u^(l+1) = Σ_{i∈N(u)} (1/√|N(u)|·√|N(i)|) · e_i^(l)
  e_i^(l+1) = Σ_{u∈N(i)} (1/√|N(i)|·√|N(u)|) · e_u^(l)

Final embedding (layer combination):
  e_u = (1/(L+1)) · Σ_{l=0}^{L} e_u^(l)    [mean pooling]
  e_i = (1/(L+1)) · Σ_{l=0}^{L} e_i^(l)

Score: ŷ_ui = e_u · e_i
Loss: BPR loss = -Σ log σ(ŷ_ui - ŷ_uj)  [j = negative sample]
```

### Gideon Integration

```julia
# src/algorithms/lightgcn.jl

mutable struct LightGCN{T<:AbstractFloat} <: AbstractMatrixFactorization
    # Hyperparameters
    rank::Int                # embedding dimension k
    n_layers::Int            # L (typically 3-4)
    learning_rate::T
    λ::T                     # L2 regularization
    max_iter::Int            # epochs
    batch_size::Int
    n_negative::Int          # negatives per positive
    convergence_tol::T
    verbose::Bool

    # Learned parameters
    user_factors::Matrix{T}  # (k, n_users) — e_u^(0)
    item_factors::Matrix{T}  # (k, n_items) — e_i^(0)

    # Precomputed graph structure
    norm_adj::Any            # Normalized adjacency matrix (sparse)
    is_fitted::Bool
end
```

### Key Implementation Notes

1. **No learnable weights per layer** — only the initial embeddings are parameters
2. **Normalized adjacency**: Precompute `D^{-1/2} A D^{-1/2}` where A is the bipartite adj matrix
3. **Layer propagation** = sparse matrix multiplication (very efficient in Julia!)
4. **Message passing**: `E^(l+1) = norm_adj * E^(l)` — just SpMM
5. **Training**: Mini-batch BPR with uniform negative sampling
6. **Predict**: `user_factors` and `item_factors` are the final (aggregated) embeddings

### Why Julia Excels Here

- Sparse matrix × Dense matrix (SpMM) is what Julia does best
- No framework overhead — Flux.jl or even manual gradient for BPR
- The message passing is literally `Y = A * X` — a single `mul!` call
- Can easily parallelize negative sampling with `Threads.@threads`

### Benchmark Comparison

LightGCN typically beats BPR by 10-15% NDCG and matches or slightly beats VAE-CF while being much more scalable.

---

## 3. Contrastive Learning (SGL / SimGCL)

**Papers:**
- Wu et al., "Self-supervised Graph Learning for Recommendation" (SIGIR 2021) — SGL
- Yu et al., "Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation" (SIGIR 2022) — SimGCL

### Architecture (SimGCL — simpler, often better)

```
Standard GCN forward pass (like LightGCN):
  E = propagate(E_0, norm_adj, L layers)

Contrastive views via noise perturbation (no graph augmentation!):
  E'_u = E_u + Δ_u,  where Δ ~ Uniform, ‖Δ‖₂ = ε
  E''_u = E_u + Δ'_u

InfoNCE contrastive loss:
  L_cl = -Σ_u log[ exp(sim(e'_u, e''_u)/τ) / Σ_v exp(sim(e'_u, e''_v)/τ) ]

Total loss:
  L = L_BPR + λ_cl · L_cl
```

### Gideon Integration

```julia
# src/algorithms/simgcl.jl

mutable struct SimGCL{T<:AbstractFloat} <: AbstractMatrixFactorization
    # Hyperparameters (inherits LightGCN structure)
    rank::Int
    n_layers::Int
    learning_rate::T
    λ::T                     # L2 reg
    λ_cl::T                  # contrastive loss weight (e.g., 0.1)
    ε::T                     # noise magnitude (e.g., 0.1)
    τ::T                     # temperature (e.g., 0.2)
    max_iter::Int
    batch_size::Int
    n_negative::Int
    convergence_tol::T
    verbose::Bool

    # Learned
    user_factors::Matrix{T}
    item_factors::Matrix{T}
    norm_adj::Any
    is_fitted::Bool
end
```

### Key Implementation Notes

1. **SimGCL is simpler than SGL** — no graph augmentation (edge dropping, node dropping), just noise
2. **Noise perturbation**: Add uniform random vectors normalized to ε to embeddings
3. **InfoNCE**: Compute pairwise cosine similarity within batch — GPU-friendly matmul
4. **Builds on LightGCN**: Same message passing, same BPR loss, just adds contrastive term
5. **Can share code** with LightGCN implementation (just override the loss)

### Benchmark Comparison

SimGCL improves over LightGCN by 3-7% NDCG on most benchmarks, with minimal additional cost.

---

## 4. Causal Recommendation (IPS / Doubly Robust)

**Papers:**
- Schnabel et al., "Recommendations as Treatments" (ICML 2016) — IPS
- Wang et al., "Doubly Robust Joint Learning" (KDD 2019) — DR

### Architecture

```
Problem: Observed interactions are biased (popular items shown more)

Propensity Score estimation:
  p(o_ui = 1) = P(item i shown to user u)
  Typically: p_i ∝ (popularity_i)^α   [naive popularity model]
  Or: logistic regression on user/item features

IPS-weighted loss:
  L_IPS = Σ_{(u,i) observed} (1/p_ui) · loss(ŷ_ui, y_ui)

Doubly Robust estimator:
  L_DR = Σ_{(u,i)∈Obs} (δ_ui - ê_ui)/p_ui + Σ_{all u,i} ê_ui
  where ê_ui = imputed error, δ_ui = actual error

SNIPS (Self-Normalized IPS) for variance reduction:
  L_SNIPS = Σ (1/p_ui)·loss_ui / Σ (1/p_ui)
```

### Gideon Integration

```julia
# src/algorithms/causal_mf.jl

@enum PropensityEstimator begin
    POPULARITY_PROPENSITY    # p_i ∝ pop^α
    UNIFORM_PROPENSITY       # p = constant (baseline, no debiasing)
    LEARNED_PROPENSITY       # logistic model from features
end

@enum DebiasMethod begin
    IPS          # Inverse propensity scoring
    SNIPS        # Self-normalized IPS
    DOUBLY_ROBUST
end

mutable struct CausalMF{T<:AbstractFloat} <: AbstractMatrixFactorization
    # Base MF hyperparameters
    rank::Int
    learning_rate::T
    λ::T
    max_iter::Int
    batch_size::Int

    # Causal parameters
    debias::DebiasMethod
    propensity::PropensityEstimator
    propensity_cap::T        # max propensity weight (clip for variance reduction, e.g., 50)
    α::T                     # popularity exponent for propensity (e.g., 0.5)
    convergence_tol::T
    verbose::Bool

    # Learned
    user_factors::Matrix{T}
    item_factors::Matrix{T}
    propensity_scores::Vector{T}  # per-item propensity
    is_fitted::Bool
end
```

### Key Implementation Notes

1. **Propensity estimation**: Easiest = popularity-based: `p_i = (count_i / max_count)^α`
2. **Weight clipping**: Essential — clip IPS weights to `[1/cap, cap]` for stability
3. **No neural network needed** — this wraps any MF model with debiased training
4. **Can be composed**: `CausalMF` trains a standard MF model but with IPS-weighted loss
5. **Predict**: Same as standard MF (U'V), the debiasing is in training only

### Why This Fits Gideon Well

- No external dependencies (pure Julia implementation)
- Small modification to existing BPR/LMF training loops
- Can be implemented as a wrapper or trait:
  ```julia
  # Option A: Wrapper model
  fit!(model::CausalMF, X) # internally trains BPR/MF with weighted loss

  # Option B: Keyword argument to existing models
  fit!(model::BPR, X; propensity_weights=propensity_scores)
  ```

### Benchmark Comparison

IPS-MF typically improves NDCG@10 by 5-15% on biased datasets (where popular items dominate). On uniformly random datasets, no improvement (as expected).

---

## Implementation Priority & Effort

| Algorithm | Effort | Dependencies | Expected Gain | Priority |
|-----------|--------|-------------|---------------|----------|
| **CausalMF** | 2-3 days | None | +5-15% on biased data | ★★★★★ |
| **LightGCN** | 3-5 days | Flux.jl (weak dep) | +10-15% vs BPR | ★★★★☆ |
| **SimGCL** | 1-2 days (after LightGCN) | Flux.jl | +3-7% vs LightGCN | ★★★☆☆ |
| **VAE-CF** | 3-5 days | Flux.jl | +3-8% vs EASE | ★★★☆☆ |

### Recommended Order

1. **CausalMF first** — no dependencies, wraps existing code, immediately useful
2. **LightGCN** — most impactful research algorithm, great Julia fit (SpMM)
3. **SimGCL** — trivial extension of LightGCN
4. **VAE-CF** — most complex, Flux dependency, competitive with EASE which is simpler

---

## Extension Architecture

```
ext/
  GideonCUDAExt.jl          # existing
  GideonFluxExt.jl          # NEW: VAE-CF, LightGCN, SimGCL (requires Flux.jl)

src/algorithms/
  causal_mf.jl              # NEW: no external deps
  lightgcn.jl               # NEW: struct only, fit! in ext
  simgcl.jl                 # NEW: struct only, fit! in ext
  vae_cf.jl                 # NEW: struct only, fit! in ext
```

Project.toml additions:
```toml
[weakdeps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"

[extensions]
GideonCUDAExt = "CUDA"
GideonFluxExt = "Flux"
```

---

## Benchmarking Plan

For fair comparison with existing Gideon algorithms:

| Dataset | Algorithms | Metrics |
|---------|-----------|---------|
| MovieLens-20M | All 4 + EASE + iALS + BPR | NDCG@10, MAP@10, Recall@20 |
| Amazon-Books | LightGCN + SimGCL + BPR + iALS | Same |
| Yahoo! R3 (unbiased test) | CausalMF + iALS + BPR | Same (unbiased eval) |

This validates:
- CausalMF improves on biased training with unbiased test
- LightGCN/SimGCL improve over BPR/iALS
- VAE-CF competitive with EASE
