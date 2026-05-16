#!/usr/bin/env Rscript
# benchmark/r_huge_benchmark.R
# Aggressive huge-matrix benchmark for rsparse — scales up to 500K×50K
# Exports shared matrices (XLarge, XXLarge) as CSV triplets for Julia comparison
# Run: Rscript benchmark/r_huge_benchmark.R

suppressPackageStartupMessages({
  library(rsparse)
  library(Matrix)
  library(MatrixExtra)
})

# ── Use the same thread count as Julia (4 default threads) ──────────────────
N_THREADS <- 4L
if (requireNamespace("RcppParallel", quietly = TRUE)) {
  RcppParallel::setThreadOptions(numThreads = N_THREADS)
  cat(sprintf("R threads: %d (via RcppParallel)\n", N_THREADS))
} else {
  cat(sprintf("R threads: system default (RcppParallel not directly available)\n"))
}

RANK   <- 10L
LAMBDA <- 0.1
ALPHA  <- 1.0
N_ITER <- 10L

set.seed(42)
cat("=== R rsparse Huge-Matrix Benchmark ===\n\n")

# ── Helper: fast sparse matrix generation with Poisson ratings ───────────────
make_sparse <- function(n_users, n_items, density, seed = 42L) {
  set.seed(seed)
  nnz <- max(1L, as.integer(round(density * n_users * n_items)))
  ri  <- sample.int(n_users, nnz, replace = TRUE)
  ci  <- sample.int(n_items, nnz, replace = TRUE)
  xi  <- as.numeric(pmax(1L, rpois(nnz, 2L)))   # Poisson ratings ≥ 1
  sparseMatrix(i = ri, j = ci, x = xi, dims = c(n_users, n_items))
}

write_triplet <- function(X, path) {
  cx <- as(X, "TsparseMatrix")
  write.csv(
    data.frame(row = cx@i + 1L, col = cx@j + 1L, val = cx@x),
    path, row.names = FALSE
  )
}
write_dims <- function(nr, nc, path) {
  write.csv(data.frame(x = c(nr, nc)), path, row.names = FALSE)
}

# ── Benchmark helper ──────────────────────────────────────────────────────────
bench_wrmf <- function(X, solver = "cholesky", n_runs = 1L) {
  times <- numeric(n_runs)
  for (i in seq_len(n_runs)) {
    m  <- WRMF$new(rank = RANK, lambda = LAMBDA, alpha = ALPHA,
                   feedback = "implicit", solver = solver)
    t0 <- proc.time()
    m$fit_transform(X, n_iter = N_ITER, convergence_tol = -1)
    times[i] <- (proc.time() - t0)[["elapsed"]]
  }
  min(times)
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. XLarge — 10K users × 5K items, density=1%  (~500K nnz)
#    → exported to /tmp/ for Julia cross-validation
# ─────────────────────────────────────────────────────────────────────────────
cat("--- XLarge (10K × 5K, density=1%) ---\n")
X_xl <- make_sparse(10000L, 5000L, 0.01)
cat(sprintf("  nnz: %d  avg nnz/user: %.1f\n", nnzero(X_xl), nnzero(X_xl)/10000))

write_triplet(X_xl, "/tmp/X_xlarge.csv")
write_dims(10000L, 5000L, "/tmp/X_xlarge_dims.csv")

t_xl_chol <- bench_wrmf(X_xl, "cholesky", 2L)
t_xl_cg   <- bench_wrmf(X_xl, "conjugate_gradient", 2L)
cat(sprintf("  Cholesky: %.3f s\n", t_xl_chol))
cat(sprintf("  CG:       %.3f s\n\n", t_xl_cg))

# ─────────────────────────────────────────────────────────────────────────────
# 2. XXLarge — 50K users × 10K items, density=0.5%  (~2.5M nnz)
#    → exported to /tmp/ for Julia cross-validation
# ─────────────────────────────────────────────────────────────────────────────
cat("--- XXLarge (50K × 10K, density=0.5%) ---\n")
X_xxl <- make_sparse(50000L, 10000L, 0.005)
cat(sprintf("  nnz: %d  avg nnz/user: %.1f\n", nnzero(X_xxl), nnzero(X_xxl)/50000))

write_triplet(X_xxl, "/tmp/X_xxlarge.csv")
write_dims(50000L, 10000L, "/tmp/X_xxlarge_dims.csv")

t_xxl_chol <- bench_wrmf(X_xxl, "cholesky", 2L)
t_xxl_cg   <- bench_wrmf(X_xxl, "conjugate_gradient", 2L)
cat(sprintf("  Cholesky: %.3f s\n", t_xxl_chol))
cat(sprintf("  CG:       %.3f s\n\n", t_xxl_cg))

# ─────────────────────────────────────────────────────────────────────────────
# 3. Large3 — 200K users × 20K items, density=0.1%  (~4M nnz)
# ─────────────────────────────────────────────────────────────────────────────
cat("--- Large3 (200K × 20K, density=0.1%) ---\n")
X_l3 <- make_sparse(200000L, 20000L, 0.001)
cat(sprintf("  nnz: %d  avg nnz/user: %.1f\n", nnzero(X_l3), nnzero(X_l3)/200000))

# Only export dims (matrix too large to write fast via CSV)
write_dims(200000L, 20000L, "/tmp/X_large3_dims.csv")

t_l3_chol <- bench_wrmf(X_l3, "cholesky", 1L)
t_l3_cg   <- bench_wrmf(X_l3, "conjugate_gradient", 1L)
cat(sprintf("  Cholesky: %.3f s\n", t_l3_chol))
cat(sprintf("  CG:       %.3f s\n\n", t_l3_cg))

# ─────────────────────────────────────────────────────────────────────────────
# 4. Huge — 500K users × 50K items, density=0.05%  (~12.5M nnz)
# ─────────────────────────────────────────────────────────────────────────────
cat("--- Huge (500K × 50K, density=0.05%) ---\n")
X_h <- make_sparse(500000L, 50000L, 0.0005)
cat(sprintf("  nnz: %d  avg nnz/user: %.1f\n", nnzero(X_h), nnzero(X_h)/500000))

# CG only at this scale for R (Cholesky is still O(n×k²×nnz) but CG is better)
t_h_chol <- bench_wrmf(X_h, "cholesky", 1L)
t_h_cg   <- bench_wrmf(X_h, "conjugate_gradient", 1L)
cat(sprintf("  Cholesky: %.3f s\n", t_h_chol))
cat(sprintf("  CG:       %.3f s\n\n", t_h_cg))

# ─────────────────────────────────────────────────────────────────────────────
# 5. Mega — 1M users × 100K items, density=0.01%  (~10M nnz)
#    Julia-comparable scale — this is the real target
# ─────────────────────────────────────────────────────────────────────────────
cat("--- Mega (1M × 100K, density=0.01%) ---\n")
X_m <- make_sparse(1000000L, 100000L, 0.0001)
cat(sprintf("  nnz: %d  avg nnz/user: %.1f\n", nnzero(X_m), nnzero(X_m)/1000000))

t_m_chol <- bench_wrmf(X_m, "cholesky", 1L)
t_m_cg   <- bench_wrmf(X_m, "conjugate_gradient", 1L)
cat(sprintf("  Cholesky: %.3f s\n", t_m_chol))
cat(sprintf("  CG:       %.3f s\n\n", t_m_cg))

# ─────────────────────────────────────────────────────────────────────────────
# 6. Save timing summary
# ─────────────────────────────────────────────────────────────────────────────
timing_huge <- data.frame(
  size      = c("10K×5K",   "50K×10K",  "200K×20K", "500K×50K", "1M×100K"),
  nnz_approx= c("~500K",    "~2.5M",    "~4M",      "~12.5M",   "~10M"),
  density   = c("1.0%",     "0.5%",     "0.1%",     "0.05%",    "0.01%"),
  r_chol_s  = c(t_xl_chol,  t_xxl_chol, t_l3_chol,  t_h_chol,   t_m_chol),
  r_cg_s    = c(t_xl_cg,    t_xxl_cg,   t_l3_cg,    t_h_cg,     t_m_cg),
  n_threads  = rep(N_THREADS, 5)
)
write.csv(timing_huge, "/tmp/r_huge_timing.csv", row.names = FALSE)

cat("=== R huge benchmark complete ===\n")
cat("Results saved to /tmp/r_huge_timing.csv\n")
cat("Shared matrices (XLarge, XXLarge) exported to /tmp/X_xlarge*.csv, /tmp/X_xxlarge*.csv\n")
print(timing_huge[, c("size", "r_chol_s", "r_cg_s")])
