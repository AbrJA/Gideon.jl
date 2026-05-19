# GloVe on text8 — text2vec (R)
# Run as script:  Rscript download.r
# Or in REPL:     source("download.r")  then call main()

suppressPackageStartupMessages({
  library(text2vec)
  library(parallel)
})

ensure_text8 <- function(path) {
  if (!file.exists(path)) {
    message("Downloading text8...")
    download.file("http://mattmahoney.net/dc/text8.zip", paste0(path, ".zip"), mode = "wb")
    unzip(paste0(path, ".zip"), exdir = dirname(path))
  }
}

build_tcm <- function(tokens, min_count = 5L, window = 5L) {
  it    <- itoken(tokens, progressbar = FALSE)
  vocab <- prune_vocabulary(create_vocabulary(it), term_count_min = min_count)
  vect  <- vocab_vectorizer(vocab)
  tcm   <- create_tcm(it, vect, skip_grams_window = window)
  list(tcm = tcm, vocab = vocab)
}

train_glove <- function(tcm, rank = 50L, x_max = 10, learning_rate = 0.15,
                        n_iter = 20L, n_threads = detectCores()) {
  glove <- GlobalVectors$new(rank = rank, x_max = x_max, learning_rate = learning_rate)
  wv    <- glove$fit_transform(tcm, n_iter = n_iter, convergence_tol = 0.01,
                               n_threads = n_threads)
  E <- wv + t(glove$components)   # vocab x rank  (rows = words)
  list(glove = glove, E = E)
}

nearest <- function(word, E, k = 5L) {
  v    <- E[word, , drop = FALSE]
  sims <- sim2(E, v, method = "cosine", norm = "l2")[, 1]
  head(sort(sims[names(sims) != word], decreasing = TRUE), k)
}

analogy <- function(a, b, c, k = 5) {
  v    <- E[b, ] - E[a, ] + E[c, ]
  sims <- (E / sqrt(rowSums(E^2))) %*% (v / sqrt(sum(v^2)))
  top  <- order(sims, decreasing = TRUE)
  top  <- top[!rownames(E)[top] %in% c(a, b, c)][1:k]
  setNames(sims[top], rownames(E)[top])
}

main <- function() {
  text8 <- path.expand("~/text8")
  ensure_text8(text8)

  tokens <- space_tokenizer(readLines(text8, n = 1L, warn = FALSE))

  res   <- build_tcm(tokens)
  cat(sprintf("vocab: %d words  |  co-occ nnz: %d\n", nrow(res$vocab), Matrix::nnzero(res$tcm)))

  out   <- train_glove(res$tcm)
  E     <<- out$E
  glove <<- out$glove

  cat("\n── Nearest neighbors ──────────────────────────────────────────\n")
  for (w in c("paris", "london", "king", "computer")) {
    if (w %in% rownames(E))
      cat(sprintf("%-10s  %s\n", w, paste(names(nearest(w, E)), collapse = "  ")))
  }

  cat("\n── Analogies ────────────────────────────────────────────────\n")
  analogy("man", "king", "woman", 10)
  analogy("paris", "france", "berlin", 10)

  invisible(list(glove = glove, E = E, vocab = res$vocab, tcm = res$tcm))
}

main()
