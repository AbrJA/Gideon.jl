Blueprint: Replicating & Enhancing rsparse in Julia

# 1. Executive Summary & Goal

The objective of this project is to port the R package rsparse (Statistical Learning on Sparse Matrices) into a 100% pure, high-performance, production-ready Julia framework.

While the original R library relies on C++ and OpenMP backends to handle scale, this Julia implementation will leverage Multiple Dispatch, Native Multithreading, and SIMD Vectorization to match or exceed the original performance while maintaining an elegant, extensible, and maintainable codebase.

The R code is now in my local in this path /home/ajaimes/Documents/GitHub/R/rsparse or can be found on https://github.com/dselivanov/rsparse. The R, src and tests folders are the most important to look for undestand the implementation

# 2. Core Architectural Design

## 2.1 Type Hierarchy & Multiple Dispatch

To ensure the library is easy to extend with new algorithms, we will avoid monolithic functions and instead use a trait-based, abstract type architecture.

```{julia}
abstract type AbstractSparseModel end
abstract type AbstractMatrixFactorization <: AbstractSparseModel end
abstract type AbstractSparseRegression <: AbstractSparseModel end

# Concrete Types (Examples)
struct WRMF <: AbstractMatrixFactorization
    factors::Int
    λ::Float64
    α::Float64
    max_iter::Int
end

struct FactorizationMachine <: AbstractSparseRegression
    # FM specific fields
end
```

## 2.2 The Storage Challenge: CSC vs. CSR

Julia’s standard SparseArrays.jl uses Compressed Sparse Column (CSC) format. Recommender systems require fast row-slicing (for users) and column-slicing (for items).

Strategy: We will utilize SparseMatricesCSR.jl for row-oriented operations, or maintain dual-representations (both a CSC matrix $A$ and its transposed CSC copy $A^T$ acting as CSR) to ensure cache-local, multithreaded operations during Alternating Least Squares (ALS) sweeps.

# 3. Algorithm Implementation Mapping

We will implement the core rsparse suite using the absolute best-in-class Julia performance practices.

| R Algorithm / Feature | Julia Strategy / Library | Target Performance Optimization |
| :--- | :--- | :--- |
| **WRMF / Implicit ALS** | Custom Solver + `LinearAlgebra` | Multithreaded Cholesky decomposition per user/item row. |
| **Logistic Matrix Fac. (LMF)** | Custom SGD/ALS Solver | Accelerated via `@turbo` / `LoopVectorization.jl` for loss sigmoid updates. |
| **Elastic Net (FTRL-SGD)** | Custom Streaming Iterators | Fast sparse-vector dot products via SIMD. |
| **Factorization Machines** | Custom SGD | Type-stable gradient updates with state tracking. |
| **Soft-Impute / SVD** | `Arpack.jl` or `TSVD.jl` base | Accelerated SVD iterations targeting top-$k$ singular values. |
| **GloVe Matrix Fac.** | Custom Async SGD | Lock-free Hogwild-style multithreading. |

Objective Function Example (WRMF)

For the explicit/implicit feedback loops, the solver will optimize the standard penalized weighted squared error:

$$L = \sum_{u,i} c_{ui} (p_{ui} - x_u^T y_i)^2 + \lambda \left(\sum_u ||x_u||^2 + \sum_i ||y_i||^2\right)$$

# 4. Performance & Vectorization Stack

To ensure this library is production-ready and outpaces C++, the code must adhere to the following strict performance rules:

Zero-Allocation Inner Loops: Pre-allocate all work arrays (e.g., the $A^T A$ matrices for ALS step) per thread. Never allocate memory inside the user/item loops.

Thread Management: Use Polyester.jl (@batch) for low-overhead threading on small-to-medium loops, and standard Threads.@threads for massive outer loops.

SIMD and Loop Unrolling: Use LoopVectorization.jl (@turbo) or native @simd annotations on all element-wise sparse operations and dot products.

Cache Locality: Ensure that when iterating over users, we read sequential memory blocks.

# 5. Testing & Verification Framework

To ensure the library is battle-tested and production-ready, the testing suite will be divided into three layers:

## 5.1 Unit and Numerical Correctness Testing

Every algorithm must be tested against known baseline inputs.

Precision Verification: Compare factorization outputs directly against rsparse R outputs using pre-exported .csv or .rds datasets.

Edge Cases: Test with empty sparse matrices, completely dense matrices, matrices with missing rows/columns, and duplicate entries.

```
using Test
using SparseArrays

@testset "WRMF Convergence & Correctness" begin
    X = sprand(100, 100, 0.05)
    model = WRMF(factors=10, λ=0.1, α=1.0, max_iter=5)
    fit!(model, X)

    @test size(model.user_factors) == (10, 100)
    @test !any(isnan, model.user_factors)
end
```

## 5.2 Observability & Logging

Use Julia's native Logging standard library.

Implement highly informative @debug logs inside optimization loops (e.g., tracking per-iteration loss reductions).

Avoid printing directly to stdout (println) inside core algorithms to keep production logs clean.

## 5.3 Automated Benchmarking

A benchmark/ directory will be established using BenchmarkTools.jl to prevent performance regressions during development.

# 6. Phase-by-Phase Roadmap

Phase 1: Core Data Structures. Set up the project structure, dependencies (SparseArrays, SparseMatricesCSR, LinearAlgebra), and establish the abstract type interfaces.

Phase 2: The ALS Engine. Implement the implicit WRMF/ALS algorithm first, as it is the most critical benchmark for rsparse. Focus heavily on multi-threaded performance.

Phase 3: Remaining Algorithms. Implement FTRL, FMs, LMF, and GloVe sequentially.

Phase 5: Evaluation Metrics. Implement high-performance Top-N recommendation evaluation metrics (MAP@K, NDCG@K, Precision@K).

Don't stop until you reach the goal. Check that the code is correct (validated vs R's) and benchmarked (comparated vs R's).
