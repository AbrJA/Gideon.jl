# Validation

This directory contains optional reference-based validation scripts for Gideon.jl. These are **not part of the test suite**—they're tools for validating changes and performance against reference implementations.

## Structure

- **`compare_with_r.jl`** — Validates Gideon algorithms against R implementations (rsparse package)

## Running R Reference Validation

### Step 1: Generate Fixtures

Fixtures are R reference outputs saved as CSV files. Generate them once:

```bash
cd ..
Rscript test/generate_fixtures.R
```

This creates files in `test/fixtures/`:
- `wrmf_chol_loss.txt`, `wrmf_chol_user.csv`, `wrmf_chol_item.csv`
- `wrmf_cg_loss.txt`, `wrmf_cg_user.csv`, `wrmf_cg_item.csv`
- `X_small.csv`, `X_small_dims.csv`
- `X_ftrl.csv`, `X_ftrl_dims.csv`, `y_ftrl.csv`, `ftrl_weights.csv`, `ftrl_preds.csv`
- `fm_xor_preds.csv`
- `glove_X.csv`, `glove_dims.csv`, `glove_final_cost.txt`
- `metrics_ref.csv`

**Requirements:**
- R with `rsparse` package
- May take 2-5 minutes depending on system

### Step 2: Run Validation

```bash
julia --project=. validation/compare_with_r.jl
```

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
- See `test/generate_fixtures.R` for exact versions

## Notes

- Fixtures are **NOT** stored in git—generate locally only when validating
- Fixtures are placed in `test/fixtures/` (same directory as test scripts)
- Tolerance bounds are tight:
  - WMF: ≤ 1.05× R (was 1.10×)
  - FTRL: ≥ 0.9995 correlation (was 0.999)
  - GloVe: ≤ 2.0× R (was 3.0×)
- If fixtures are missing, validation gracefully skips with warning
