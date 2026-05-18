# Ranking Metrics

```@docs
ap_at_k
map_at_k
ndcg_at_k
precision_at_k
recall_at_k
```

## Example

```julia
using Gideon, SparseArrays

# Ground truth: user 1 likes items 3, 7, 9
actual = sparse([1,1,1], [3,7,9], ones(3), 1, 10)

# Model predictions: top-4 items for user 1
predictions = [3 7 1 9]

map_at_k(predictions, actual; k=4)   # 0.833...
ndcg_at_k(predictions, actual; k=4)  # high
precision_at_k(predictions, actual; k=4)  # 0.75
recall_at_k(predictions, actual; k=4)     # 1.0
```
