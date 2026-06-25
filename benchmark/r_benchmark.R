#!/usr/bin/env Rscript
# R benchmark / validation script for rsparse
# Outputs CSV results to /tmp/rsparse_results.csv

suppressPackageStartupMessages({
  library(rsparse)
  library(Matrix)
  library(MatrixExtra)
})

set.seed(42)
cat("=== R rsparse Validation & Benchmark ===\n")

# ─────────────────────────────────────────────
# 1. Generate shared test matrices
# ─────────────────────────────────────────────
# Small: correctness check
n_users_s <- 100L; n_items_s <- 80L
X_small <- rsparsematrix(n_users_s, n_items_s, density = 0.05, rand.x = function(n) sample(1:5, n, TRUE))
X_small <- as(X_small, "CsparseMatrix")

# Medium: speed benchmark
n_users_m <- 1000L; n_items_m <- 500L
X_medium <- rsparsematrix(n_users_m, n_items_m, density = 0.03, rand.x = function(n) sample(1:5, n, TRUE))
X_medium <- as(X_medium, "CsparseMatrix")

# Large: speed benchmark
n_users_l <- 5000L; n_items_l <- 2000L
X_large <- rsparsematrix(n_users_l, n_items_l, density = 0.01, rand.x = function(n) sample(1:5, n, TRUE))
X_large <- as(X_large, "CsparseMatrix")

# Export sparse matrices as CSV triplets for Julia
write_triplet <- function(X, path) {
  cx <- as(X, "TsparseMatrix")
  df <- data.frame(
    row = cx@i + 1L,
    col = cx@j + 1L,
    val = cx@x
  )
  write.csv(df, path, row.names = FALSE)
}
write_triplet(X_small,  "/tmp/X_small.csv")
write_triplet(X_medium, "/tmp/X_medium.csv")
write_triplet(X_large,  "/tmp/X_large.csv")
write.csv(c(n_users_s, n_items_s), "/tmp/X_small_dims.csv",  row.names=FALSE)
write.csv(c(n_users_m, n_items_m), "/tmp/X_medium_dims.csv", row.names=FALSE)
write.csv(c(n_users_l, n_items_l), "/tmp/X_large_dims.csv",  row.names=FALSE)
cat("Data exported.\n")

# ─────────────────────────────────────────────
# 2. WRMF — correctness (small) + timing (all sizes)
# ─────────────────────────────────────────────
cat("\n--- WRMF ---\n")
RANK <- 10L; LAMBDA <- 0.1; ALPHA <- 1.0; N_ITER <- 10L

# Correctness on small
model_s <- WRMF$new(rank = RANK, lambda = LAMBDA, feedback = "implicit",
                    solver = "cholesky")
t0 <- proc.time()
user_emb_s <- model_s$fit_transform(X_small, n_iter = N_ITER, convergence_tol = -1)
t_wrmf_small <- (proc.time() - t0)[["elapsed"]]
item_emb_s <- model_s$components

cat(sprintf("  Small (%dx%d): %.3f s\n", n_users_s, n_items_s, t_wrmf_small))
cat(sprintf("  user_factors shape: %d x %d\n", nrow(user_emb_s), ncol(user_emb_s)))
cat(sprintf("  item_factors shape: %d x %d\n", nrow(item_emb_s), ncol(item_emb_s)))
cat(sprintf("  user_factors norm (F): %.6f\n", norm(user_emb_s, "F")))
cat(sprintf("  item_factors norm (F): %.6f\n", norm(item_emb_s, "F")))
cat(sprintf("  any NaN user: %s\n", any(is.nan(user_emb_s))))
cat(sprintf("  any NaN item: %s\n", any(is.nan(item_emb_s))))

# Export factors for Julia comparison
write.csv(user_emb_s, "/tmp/r_user_emb_small.csv", row.names = FALSE)
write.csv(item_emb_s, "/tmp/r_item_emb_small.csv", row.names = FALSE)

# Timing on medium
model_m <- WRMF$new(rank = RANK, lambda = LAMBDA, feedback = "implicit", solver = "cholesky")
t0 <- proc.time()
model_m$fit_transform(X_medium, n_iter = N_ITER, convergence_tol = -1)
t_wrmf_medium <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("  Medium (%dx%d): %.3f s\n", n_users_m, n_items_m, t_wrmf_medium))

# Timing on large
model_l <- WRMF$new(rank = RANK, lambda = LAMBDA, feedback = "implicit", solver = "cholesky")
t0 <- proc.time()
model_l$fit_transform(X_large, n_iter = N_ITER, convergence_tol = -1)
t_wrmf_large <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("  Large (%dx%d): %.3f s\n", n_users_l, n_items_l, t_wrmf_large))

# CG solver timing (medium)
model_cg <- WRMF$new(rank = RANK, lambda = LAMBDA, feedback = "implicit", solver = "conjugate_gradient")
t0 <- proc.time()
model_cg$fit_transform(X_medium, n_iter = N_ITER, convergence_tol = -1)
t_wrmf_cg_medium <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("  Medium CG (%dx%d): %.3f s\n", n_users_m, n_items_m, t_wrmf_cg_medium))

# ─────────────────────────────────────────────
# 3. FTRL — correctness + timing
# ─────────────────────────────────────────────
cat("\n--- FTRL ---\n")
n_s <- 1000L; p <- 200L
set.seed(42)
X_ftrl <- rsparsematrix(n_s, p, density = 0.1)
X_ftrl <- as(X_ftrl, "RsparseMatrix")
w_true <- rep(0, p); w_true[1:5] <- 1
logit <- function(x) 1 / (1 + exp(-x))
y_ftrl <- as.numeric(logit(as.vector(X_ftrl %*% w_true)) > 0.5)
write_triplet(as(X_ftrl, "CsparseMatrix"), "/tmp/X_ftrl.csv")
write.csv(data.frame(y = y_ftrl), "/tmp/y_ftrl.csv", row.names = FALSE)
write.csv(c(n_s, p), "/tmp/X_ftrl_dims.csv", row.names = FALSE)

ftrl_model <- FTRL$new(learning_rate = 0.1, learning_rate_decay = 0.5,
                       lambda = 0.01, l1_ratio = 0.5, dropout = 0)
t0 <- proc.time()
for (i in 1:5) ftrl_model$partial_fit(X_ftrl, y_ftrl)
t_ftrl <- (proc.time() - t0)[["elapsed"]]
preds_ftrl <- ftrl_model$predict(X_ftrl)
w_ftrl <- ftrl_model$coef()
nnz_w <- sum(w_ftrl != 0)
acc <- mean(round(preds_ftrl) == y_ftrl)
cat(sprintf("  Fit 5 epochs (%dx%d): %.3f s\n", n_s, p, t_ftrl))
cat(sprintf("  NNZ weights: %d / %d\n", nnz_w, p))
cat(sprintf("  Accuracy: %.4f\n", acc))
write.csv(data.frame(w = w_ftrl), "/tmp/r_ftrl_weights.csv", row.names = FALSE)
write.csv(data.frame(p = preds_ftrl), "/tmp/r_ftrl_preds.csv", row.names = FALSE)

# ─────────────────────────────────────────────
# 4. Factorization Machines — XOR test
# ─────────────────────────────────────────────
cat("\n--- FM (XOR) ---\n")
x_xor <- matrix(c(0,0, 0,1, 1,0, 1,1), nrow=4, byrow=TRUE)
x_xor <- as(x_xor, "RsparseMatrix")
y_xor  <- c(0, 1, 1, 0)

fm <- FM$new(learning_rate_w=10, rank=2, lambda_w=0,
                               lambda_v=0, family='binomial', intercept=TRUE,
                               learning_rate_v=10)
t0 <- proc.time()
fm$fit(x_xor, y_xor, n_iter=200)
t_fm <- (proc.time() - t0)[["elapsed"]]
preds_fm <- fm$predict(x_xor)
cat(sprintf("  200 iterations: %.3f s\n", t_fm))
cat(sprintf("  Predictions: %.4f %.4f %.4f %.4f\n",
            preds_fm[1], preds_fm[2], preds_fm[3], preds_fm[4]))
cat(sprintf("  XOR correct (p<0.3 for 0, p>0.7 for 1): %s\n",
            all(preds_fm[c(1,4)] < 0.3) && all(preds_fm[c(2,3)] > 0.7)))

# ─────────────────────────────────────────────
# 5. GloVe — timing
# ─────────────────────────────────────────────
cat("\n--- GloVe ---\n")
co_mat <- rsparsematrix(200, 200, density=0.1, rand.x=function(n) runif(n, 0.1, 10))
co_mat <- as(co_mat + t(co_mat), "CsparseMatrix")  # symmetrize
co_mat@x <- abs(co_mat@x) + 0.1

glove_model <- GloVe$new(rank=20, x_max=10, learning_rate=0.15)
t0 <- proc.time()
glove_model$fit_transform(co_mat, n_iter=10, n_threads=1)
t_glove <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("  10 iter (200x200 co-occ): %.3f s\n", t_glove))
cat(sprintf("  cost history length: %d\n", length(glove_model$get_history()$cost_history)))

# ─────────────────────────────────────────────
# 6. Metrics — correctness
# ─────────────────────────────────────────────
cat("\n--- Metrics ---\n")
actual_m <- sparseMatrix(i=c(1,1,1), j=c(5,7,9), x=c(1,1,1), dims=c(1,10))
preds_m  <- matrix(c(5L,7L,9L,2L), nrow=1)
ap  <- rsparse::ap_k(preds_m, actual_m)
ndcg <- rsparse::ndcg_k(preds_m, actual_m)
cat(sprintf("  AP@4 (perfect ranking): %.6f  (expected 1.0)\n", ap))
cat(sprintf("  NDCG@4 (perfect ranking): %.6f  (expected 1.0)\n", ndcg))

# ─────────────────────────────────────────────
# 7. Save timing summary
# ─────────────────────────────────────────────
timing <- data.frame(
  algorithm = c("WRMF_cholesky_small", "WRMF_cholesky_medium", "WRMF_cholesky_large",
                "WRMF_cg_medium", "FTRL_5ep", "FM_xor_200iter", "GloVe_10iter"),
  size      = c(sprintf("%dx%d", n_users_s, n_items_s),
                sprintf("%dx%d", n_users_m, n_items_m),
                sprintf("%dx%d", n_users_l, n_items_l),
                sprintf("%dx%d", n_users_m, n_items_m),
                sprintf("%dx%d", n_s, p),
                "4x2", "200x200"),
  r_time_s  = c(t_wrmf_small, t_wrmf_medium, t_wrmf_large,
                t_wrmf_cg_medium, t_ftrl, t_fm, t_glove)
)
write.csv(timing, "/tmp/r_timing.csv", row.names = FALSE)
cat("\nR timing saved to /tmp/r_timing.csv\n")
cat("\n=== R benchmark complete ===\n")
