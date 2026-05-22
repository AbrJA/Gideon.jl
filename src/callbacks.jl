# ──────────────────────────────────────────────────────────────────────────────
# Callbacks — extensible hooks for training loops
# ──────────────────────────────────────────────────────────────────────────────

"""
    AbstractCallback

Base type for training callbacks. Implement `on_epoch_end(cb, info)` to
receive iteration updates.
"""
abstract type AbstractCallback end

"""
    CallbackInfo{T}

Information passed to callbacks at the end of each training iteration.

# Fields
- `epoch::Int` — current iteration number
- `loss::T` — current loss value
- `elapsed::Float64` — total seconds elapsed
- `model` — reference to the model being trained
"""
struct CallbackInfo{T,M}
    epoch::Int
    loss::T
    elapsed::Float64
    model::M
end

"""
    on_epoch_end(callback, info::CallbackInfo) -> Symbol

Called at the end of each training epoch. Return `:stop` to halt training early,
or `:continue` (default) to keep going.
"""
function on_epoch_end(::AbstractCallback, ::CallbackInfo)
    :continue
end

# ──────────────────────────────────────────────────────────────────────────────
# Built-in callbacks
# ──────────────────────────────────────────────────────────────────────────────

"""
    EarlyStoppingCallback(; patience=5, min_delta=1e-4)

Stop training if loss hasn't improved by `min_delta` for `patience` epochs.
"""
mutable struct EarlyStoppingCallback <: AbstractCallback
    patience::Int
    min_delta::Float64
    best_loss::Float64
    wait::Int
end

function EarlyStoppingCallback(; patience::Int=5, min_delta::Float64=1e-4)
    EarlyStoppingCallback(patience, min_delta, Inf, 0)
end

function on_epoch_end(cb::EarlyStoppingCallback, info::CallbackInfo)
    if info.loss < cb.best_loss - cb.min_delta
        cb.best_loss = info.loss
        cb.wait = 0
    else
        cb.wait += 1
    end
    cb.wait >= cb.patience ? :stop : :continue
end

"""
    LossHistoryCallback()

Record all loss values during training. Access via `cb.losses`.
"""
mutable struct LossHistoryCallback <: AbstractCallback
    losses::Vector{Float64}
end

LossHistoryCallback() = LossHistoryCallback(Float64[])

function on_epoch_end(cb::LossHistoryCallback, info::CallbackInfo)
    push!(cb.losses, Float64(info.loss))
    :continue
end

"""
    CheckpointCallback(; every=10, path="checkpoints")

Save model state every `every` epochs using serialization.
"""
mutable struct CheckpointCallback <: AbstractCallback
    every::Int
    path::String
end

CheckpointCallback(; every::Int=10, path::String="checkpoints") =
    CheckpointCallback(every, path)

function on_epoch_end(cb::CheckpointCallback, info::CallbackInfo)
    if info.epoch % cb.every == 0
        mkpath(cb.path)
        filepath = joinpath(cb.path, "model_epoch_$(info.epoch).jls")
        save_model(info.model, filepath)
    end
    :continue
end

"""
    LearningRateScheduler(; decay=0.99, min_lr=1e-6)

Decay learning rate geometrically each epoch (model must have a `learning_rate` field).
"""
mutable struct LearningRateScheduler <: AbstractCallback
    decay::Float64
    min_lr::Float64
end

LearningRateScheduler(; decay::Float64=0.99, min_lr::Float64=1e-6) =
    LearningRateScheduler(decay, min_lr)

function on_epoch_end(cb::LearningRateScheduler, info::CallbackInfo)
    if hasproperty(info.model, :learning_rate)
        new_lr = max(info.model.learning_rate * cb.decay, cb.min_lr)
        info.model.learning_rate = new_lr
    end
    :continue
end

"""
    run_callbacks(callbacks, info) -> Bool

Run all callbacks. Returns `true` if any callback signals `:stop`.
"""
function run_callbacks(callbacks::Vector{<:AbstractCallback}, info::CallbackInfo)
    for cb in callbacks
        result = on_epoch_end(cb, info)
        result === :stop && return true
    end
    false
end
