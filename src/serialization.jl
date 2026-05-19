# ──────────────────────────────────────────────────────────────────────────────
# Serialization — save and load Gideon models
# ──────────────────────────────────────────────────────────────────────────────

"""
    save_model(model, path::String)

Serialize a Gideon model to disk using Julia's native serialization.
Includes a version header for forward-compatibility checking.
"""
function save_model(model::AbstractSparseModel, path::String)
    open(path, "w") do io
        # Version header
        write(io, "GIDEON_v1\n")
        serialize(io, typeof(model))
        serialize(io, model)
    end
    nothing
end

"""
    load_model(path::String) -> AbstractSparseModel

Deserialize a model from disk. Verifies the version header.
"""
function load_model(path::String)
    open(path, "r") do io
        header = readline(io)
        startswith(header, "GIDEON_v") ||
            error("Invalid model file: missing GIDEON header")
        _ = deserialize(io)  # type (for validation)
        model = deserialize(io)
        model
    end
end

# Also handle EASE which isn't AbstractSparseModel... wait it is now
# (we made EASE <: AbstractSparseModel)
