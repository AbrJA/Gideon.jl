# Gideon.jl — Future Steps & Technical Report

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

18. **Tables.jl / DataFrames support**: Accept interaction data as `(user, item, value)` triplets from any Tables.jl-compatible source.

19. **RecSys benchmarks scripts/package**: Separate scripts/package with standard datasets (MovieLens 100K/1M/10M/20M, Amazon reviews) + evaluation protocols.

20. **CI/CD pipeline**: GitHub Actions with Julia 1.9/1.10/1.11/nightly matrix, coverage reporting (Codecov), and automatic documentation deployment.

### Priority 5 — Research & Validation

21. **Comprehensive R benchmark suite**: Systematic comparison on MovieLens 10M with wall-clock timing, memory profiling, and result quality (MAP@10, NDCG@10).

22. **Scalability study**: Profile on synthetic matrices from 1M to 1B interactions. Identify the crossover point where GPU becomes necessary.

23. **Numerical stability audit**: Test all algorithms with extreme values (very large/small entries, near-singular Gramians) and add safeguards.

24. **Reproducibility**: Pin RNG seeds in benchmarks, document hardware specs, publish reproducible Pluto notebooks or Julia scripts.

## Known Limitations

1. **No online/incremental updates for WRMF**: Adding new users/items requires full refit.
2. **GloVe shuffle is allocation-heavy**: Creates new index permutation each iteration.
3. **FM limited to 2nd-order interactions**: Higher-order requires fundamentally different architecture.
4. **No multi-GPU support**: Single GPU via CUDA.jl extensions only.
5. **SoftImpute memory**: Stores full dense low-rank approximation — problematic for very large matrices.
