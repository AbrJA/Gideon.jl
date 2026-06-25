# Validation

This directory contains optional reference-based validation scripts for Gideon.jl. These are **not part of the test suite**—they're tools for validating changes and performance against reference implementations.

By default, R fixtures are stored in `/tmp/gideon_fixtures`.
You can override this with `GIDEON_R_FIXTURE_DIR`.
By default, Python fixtures are stored in `/tmp/gideon_fixtures/python`.
You can override this with `GIDEON_PY_FIXTURE_DIR`.

## Quick Start (One Command)

Run everything (generate fixtures + compare against R and Python):

```bash
julia --project=. validation/run.jl --prepare --all
```

Run only one reference:

```bash
julia --project=. validation/run.jl --prepare --r
julia --project=. validation/run.jl --prepare --python
```

If dependencies are missing (`Rscript`, `python3`, or Python `implicit`), the runner prints warnings and skips that part.

## Structure

- **`validate_r.jl`** — Validates Gideon algorithms against R implementations (rsparse package)
- **`validate_py.jl`** — Validates Gideon algorithms against Python implementations (implicit package)
- **`run.jl`** — Single entrypoint for prepare+run workflow
- **`fixtures_r.R`** — R fixture generator
- **`fixtures_py.py`** — Python fixture generator

## Running R Reference Validation

### Step 1: Generate Fixtures

Fixtures are R reference outputs saved as CSV files. Generate them once:

```bash
Rscript validation/fixtures_r.R
```

This creates files in `/tmp/gideon_fixtures/` (or `GIDEON_R_FIXTURE_DIR` if set):
- `rsparse_capabilities.csv` (detected model/class availability in your R environment)
- `wrmf_chol_loss.txt`, `wrmf_chol_user.csv`, `wrmf_chol_item.csv`
- `wrmf_cg_loss.txt`, `wrmf_cg_user.csv`, `wrmf_cg_item.csv`
- `X_small.csv`, `X_small_dims.csv`
- `X_ftrl.csv`, `X_ftrl_dims.csv`, `y_ftrl.csv`, `ftrl_weights.csv`, `ftrl_preds.csv`
- `fm_xor_preds.csv` (generated with `rsparse::FactorizationMachine`)
- `glove_X.csv`, `glove_dims.csv`, `glove_final_cost.txt`
- `metrics_ref.csv`

**Requirements:**
- R with `rsparse` package
- May take 2-5 minutes depending on system

The FM fixture is generated with `rsparse::FactorizationMachine`.

### Step 2: Run Validation

```bash
julia --project=. validation/validate_r.jl
```

## Running Python Reference Validation

### Step 1: Generate Python Fixtures

```bash
python3 validation/fixtures_py.py
```

This creates files in `/tmp/gideon_fixtures/python/` (or `GIDEON_PY_FIXTURE_DIR` if set).

### Step 2: Run Validation

```bash
julia --project=. validation/validate_py.jl
```

This compares score behavior for:
- WMF (Julia) vs ALS (Python implicit)
- IALS (Julia) vs ALS-based Python reference
- EALS (Julia) vs ALS-surrogate Python reference
- BPR (Julia) vs BPR (Python implicit)
- LogisticMF (Julia) vs LogisticMatrixFactorization (Python implicit, optional)
- EASE (Julia) vs deterministic NumPy EASE implementation
- SLIM (Julia) vs scikit-learn ElasticNet reference (optional)
- SoftImpute (Julia) vs iterative soft-threshold SVD reference

The script reports score correlation and top-k overlap.
For EASE it also reports matrix-level relative Frobenius error.
For LogisticMF, top-k overlap is the primary pass criterion by default; score correlation is reported for diagnostics.
For BPR and LogisticMF, it additionally compares split-based ranking quality (NDCG@10 and Recall@10)
on the same train/test split generated from Python fixtures.
By default, LogisticMF split-metric deltas are diagnostic only; set `GIDEON_PY_LMF_STRICT=1`
to enforce threshold-based pass/fail for those deltas.
By default, SoftImpute parity is diagnostic only; set `GIDEON_PY_SOFT_STRICT=1`
to enforce reconstruction/singular-value thresholds.

Note: SLIM parity requires `scikit-learn` in the Python environment. If unavailable,
SLIM fixture generation/parity is skipped gracefully.

Output shows:
- Test pass/fail status
- Comparison metrics (Julia vs R):
  - Loss ratios for WMF/GloVe
  - Correlation for FTRL
  - Agreement rate for FM

### Example Output

```
WMF CholeskySolver: Julia=123.45, R=120.0, ratio=1.0288
FTRL weights correlation: 0.99951
GloVe cost: Julia=456.78, R=450.0, ratio=1.0151
```

## When to Run Validation

- **Before major releases**: Ensure no regression vs R reference
- **After algorithm changes**: Validate numerical correctness
- **Performance investigation**: Check implementation efficiency
- **CI/CD**: Add as optional step before tagging releases

## Reference Implementation Versions

**R package versions used:**
- `rsparse`: 0.5.0+
- See `validation/fixtures_r.R` for exact versions

## Notes

- Fixtures are **NOT** stored in git—generate locally only when validating
- R fixtures are placed in `/tmp/gideon_fixtures` by default
- Override path with `GIDEON_R_FIXTURE_DIR=/custom/path`
- Python fixtures are placed in `/tmp/gideon_fixtures/python` by default
- Override path with `GIDEON_PY_FIXTURE_DIR=/custom/path`
- Tolerance bounds are tight:
  - WMF: ≤ 1.05× R (was 1.10×)
  - FTRL: ≥ 0.9995 correlation (was 0.999)
  - GloVe: ≤ 2.0× R (was 3.0×)
- If fixtures are missing, validation gracefully skips with warning
- FM comparison is optional and skipped when `fm_xor_preds.csv` is unavailable
