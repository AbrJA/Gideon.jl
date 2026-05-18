# Gideon.jl — Future Steps & Technical Report

## Current State (v1.0.0)

### Completed
- **Dependency cleanup**: Removed LoopVectorization, StaticArrays, ZipFile, Dates. Package now uses only LinearAlgebra, SparseArrays, SparseMatricesCSR, Random, Logging, Printf.
- **Early stopping / convergence**: All iterative algorithms (WRMF, GloVe, LMF, SoftImpute, FM) support `convergence_tol` via a unified `ConvergenceMonitor`.
- **Multi-family FTRL**: Supports `BINOMIAL`, `GAUSSIAN`, `POISSON` via the `Family` enum with proper link functions (sigmoid, identity, exp).
- **Performance**: BLAS-level operations (syr!, axpy!, syrk!, potrf!/potrs!), per-thread pre-allocated buffers, @threads :static, manual SIMD-friendly loops for small nnz.
- **Progress logging**: Structured iteration logging with timing (time_ns based), loss tracking.
- **Restructured tests**: 10 focused test files, 281 tests covering all algorithms + Aqua/JET quality checks + R fixture validation.
- **Documentation**: Documenter.jl setup with index, algorithms, metrics, and API reference pages.

### Test Results
- 281 tests total: Quality (11), Types & Utils (133), WRMF (30), FTRL (21), FM (8), GloVe (15), LMF (13), SoftImpute (16), Metrics (22), R Correctness (12)
- All passing (including R fixture validation for WRMF, FTRL, FM, GloVe, and ranking metrics)

---

## Future Steps

### Priority 1 — Performance & Scalability

1. **Precompilation statements**: Add `@precompile_calls` blocks for common workflows to reduce TTFX (time-to-first-execution).

2. **GPU acceleration (CUDA.jl)**: WRMF's Cholesky solver and GloVe's AdaGrad updates are embarrassingly parallel. Add optional `device=:gpu` parameter that dispatches to CuSparse/cuBLAS when CUDA.jl is loaded (via package extensions).

3. **Out-of-core / chunked matrix support**: For matrices that don't fit in RAM (>100M interactions), implement a block-coordinate descent variant of WRMF that processes column blocks sequentially.

4. **CSR format throughout**: Currently WRMF transposes internally. Consider storing both CSC and CSR representations to avoid repeated transpositions on large matrices.

5. **SIMD-optimized dot products**: For the inner loops of CG solver and GloVe, use `@simd ivdep` with manual unrolling for rank <= 64.

### Priority 2 — Algorithm Enhancements

6. **iALS (implicit Alternating Least Squares)**: The modern reformulation from "Revisiting the Performance of iALS on Item Recommendation Benchmarks" (Rendle et al., 2022) that avoids forming the full Gramian.

7. **eALS**: Efficient ALS with element-wise weighting (He et al., 2016) — more memory-efficient than WRMF for non-uniform confidence.

8. **BPR (Bayesian Personalized Ranking)**: Popular pairwise learning-to-rank objective that pairs well with LMF's negative sampling.

9. **SLIM (Sparse Linear Methods)**: Item-item similarity via L1/L2-regularized regression — complements matrix factorization methods.

10. **Cross-validation utilities**: `cv_fit(model, X; n_folds=5, metric=map_at_k)` with proper temporal split support for recommendation data.

11. **Hyperparameter search**: Simple grid/random search with warm-starting (reuse previous factor matrices as initialization).

### Priority 3 — API & Usability

12. **Serialization**: `save_model(model, "path.jls")` / `load_model("path.jls")` with version checking.

13. **Predict API**: Add `predict_scores(model, user_indices, item_indices)` returning raw scores without top-k sorting — useful for evaluation pipelines.

14. **Warm-start API**: Formalize `fit!(model, X; U_init=..., V_init=...)` with validation and documentation.

15. **Callback system**: `fit!(model, X; callback=f)` where `f(iter, loss, model)` is called each iteration — enables custom logging, checkpointing, learning rate schedules.

16. **Type stability audit**: Run `@code_warntype` on all hot paths and eliminate any remaining type instabilities (especially in GloVe's shuffle path).

### Priority 4 — Ecosystem Integration

17. **MLJ.jl integration**: Implement the MLJ model interface (`fit`, `predict`, `transform`) so Gideon models can participate in MLJ pipelines.

18. **Tables.jl / DataFrames support**: Accept interaction data as `(user, item, value)` triplets from any Tables.jl-compatible source.

19. **RecSys benchmarks package**: Separate package with standard datasets (MovieLens 100K/1M/10M/20M, Amazon reviews) + evaluation protocols.

20. **CI/CD pipeline**: GitHub Actions with Julia 1.9/1.10/1.11/nightly matrix, coverage reporting (Codecov), and automatic documentation deployment.

### Priority 5 — Research & Validation

21. **Comprehensive R benchmark suite**: Systematic comparison on MovieLens 10M with wall-clock timing, memory profiling, and result quality (MAP@10, NDCG@10).

22. **Scalability study**: Profile on synthetic matrices from 1M to 1B interactions. Identify the crossover point where GPU becomes necessary.

23. **Numerical stability audit**: Test all algorithms with extreme values (very large/small entries, near-singular Gramians) and add safeguards.

24. **Reproducibility**: Pin RNG seeds in benchmarks, document hardware specs, publish reproducible Pluto notebooks.

---

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| No LoopVectorization | Heavy dependency, Julia's native SIMD + BLAS sufficient for our access patterns |
| `Family` enum over Symbol | Type safety, exhaustive dispatch, no string comparison overhead |
| `ConvergenceMonitor` pattern | Uniform early stopping across all algorithms without code duplication |
| `time_ns()` over Dates | Zero-allocation timing, no Dates dependency |
| Per-thread buffers | Avoid false sharing in multi-threaded ALS, predictable allocation |
| SparseMatricesCSR for row access | O(1) row slice vs O(nnz) for CSC; critical for user-factor updates |

---

## Known Limitations

1. **No online/incremental updates for WRMF**: Adding new users/items requires full refit.
2. **GloVe shuffle is allocation-heavy**: Creates new index permutation each iteration.
3. **FM limited to 2nd-order interactions**: Higher-order requires fundamentally different architecture.
4. **No multi-GPU support**: Single GPU via CUDA.jl extensions only.
5. **SoftImpute memory**: Stores full dense low-rank approximation — problematic for very large matrices.
