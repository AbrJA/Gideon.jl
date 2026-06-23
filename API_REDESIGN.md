# Gideon.jl API Redesign Plan

## Problem Statement

The current API has semantic overloading and inconsistencies:

| Issue | Description |
|-------|-------------|
| `predict` means two things | MF models: `predict(m, X; k=10) → Matrix{Int}` (top-k) vs FTRL/FM: `predict(m, X) → Vector{T}` (scores) |
| `predict_scores` is fragmented | WRMF has both variants; IALS/BPR only pairwise; EALS/LMF/EASE only full-matrix; GloVe/FTRL/FM none |
| GloVe has no predict | Only exports `get_embeddings` |
| SoftImpute isn't a model | Function returning a plain struct; can't participate in the API |
| EASE/SLIM have no shared type | Item-similarity models typed directly as `AbstractSparseModel` |
| Cross-validation assumes top-k | `cv_evaluate` calls `predict(model, X; k=k)` — excludes FTRL/FM/GloVe |

## Design Principles (from cross-ecosystem research)

1. **Domain-appropriate naming**: `recommend` for "top-k items" (RecSys), `predict` for regression (ML)
2. **Consistency**: Same contract for same type — every `AbstractRecommender` has `recommend` + `score`
3. **Separation of concerns**: Different operations get different names, not overloaded semantics
4. **Progressive disclosure**: Simple things simple, complex things possible
5. **Composability**: Models work with eval tools/crossval/pipelines without special-casing

## New API Surface

### Type Hierarchy

```
AbstractSparseModel
├── AbstractRecommender                 # NEW — has recommend() + score()
│   ├── AbstractMatrixFactorization     # has get_embeddings, similar_items/similar_users
│   │   ├── WRMF, IALS, EALS, BPR, LMF, GloVe
│   └── AbstractItemSimilarity          # NEW — item-item models
│       ├── EASE, SLIM
└── AbstractSparseRegression            # has predict()
    ├── FTRL, FactorizationMachine
```

### Function Contracts

| Function | Available on | Signature | Returns |
|----------|-------------|-----------|---------|
| `fit!` | All models | `fit!(model, X; kwargs...)` | model (mutated) |
| `recommend` | AbstractRecommender | `recommend(model, X; k=10)` | `Matrix{Int}` (top-k item indices) |
| `score` | AbstractRecommender | `score(model, X)` | `Matrix{T}` (full score matrix) |
| `score` | AbstractRecommender | `score(model, user_ids, item_ids)` | `Vector{T}` (pairwise) |
| `predict` | AbstractSparseRegression | `predict(model, X)` | `Vector{T}` |
| `similar_items` | AbstractMatrixFactorization | `similar_items(model, item_id; k=10)` | `(ids, scores)` |
| `similar_users` | AbstractMatrixFactorization | `similar_users(model, user_id; k=10)` | `(ids, scores)` |
| `get_embeddings` | AbstractMatrixFactorization | `get_embeddings(model)` | `Matrix{T}` |
| `coef` | AbstractSparseRegression | `coef(model)` | model coefficients |

### Backward Compatibility

Old functions (`predict` for recommenders, `predict_scores`) will be kept as deprecated thin wrappers
that call the new functions and emit a deprecation warning. This allows gradual migration.

### Callback Enhancement

Add `on_train_begin` and `on_train_end` hooks for setup/teardown (e.g., progress bars, logging).

## Implementation Order

1. **Add new abstract types** (`AbstractRecommender`, `AbstractItemSimilarity`) to types.jl
2. **Rename top-k `predict` → `recommend`** across all recommender models
3. **Rename `predict_scores` → `score`** and unify signatures
4. **Give GloVe `recommend`/`score` methods**
5. **Add `similar_items`/`similar_users`** for MF models
6. **Update cross-validation** to use `recommend` for recommenders
7. **Add `on_train_begin`/`on_train_end`** callbacks
8. **Update all tests** for new API names
9. **Add deprecation wrappers** for old function names

Each step will be validated with the full test suite before proceeding to the next.
