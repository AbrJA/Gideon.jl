# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers shared across algorithms
# ──────────────────────────────────────────────────────────────────────────────

"""
    init_factors(rng, rank, n; scale=0.01)

Initialize a `rank × n` dense factor matrix with small random values drawn
from a normal distribution N(0, `scale`²).
"""
function init_factors(rng::AbstractRNG, rank::Int, n::Int; scale::Float64=0.01)
    randn(rng, rank, n) .* scale
end

"""
    sigmoid(x)

Numerically stable logistic sigmoid: σ(x) = 1/(1+exp(-x)).
"""
@inline function sigmoid(x::T) where {T<:AbstractFloat}
    if x >= zero(T)
        z = exp(-x)
        return one(T) / (one(T) + z)
    else
        z = exp(x)
        return z / (one(T) + z)
    end
end

"""
    log1pexp(x)

Compute `log(1 + exp(x))` in a numerically stable way (softplus).
"""
@inline function log1pexp(x::T) where {T<:AbstractFloat}
    if x > T(33.3)
        return x
    elseif x > T(-33.3)
        return log1p(exp(x))
    else
        return exp(x)
    end
end

"""
    safe_inv(x; ε=1e-12)

Safe reciprocal that avoids division by zero.
"""
@inline safe_inv(x::T; ε::T=T(1e-12)) where {T<:AbstractFloat} = one(T) / (x + ε)

"""
    link_function(family::Family, x)

Apply the GLM link function for the given family:
- `BINOMIAL` → sigmoid(x)
- `GAUSSIAN` → x (identity)
- `POISSON` → exp(x)
"""
@inline function link_function(family::Family, x::T) where {T<:AbstractFloat}
    if family == BINOMIAL
        return sigmoid(x)
    elseif family == GAUSSIAN
        return x
    else  # POISSON
        return exp(x)
    end
end

"""
    _inplace_shuffle!(v, rng) -> v

Fisher-Yates in-place shuffle — O(n) time, zero allocations beyond the vector itself.
"""
function _inplace_shuffle!(v::AbstractVector, rng::AbstractRNG)
    n = length(v)
    @inbounds for i in n:-1:2
        j = rand(rng, 1:i)
        v[i], v[j] = v[j], v[i]
    end
    v
end
