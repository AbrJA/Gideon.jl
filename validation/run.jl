# validation/run.jl
# One-command runner for optional R/Python reference validation.

function _has_flag(flag::String)
    any(==(flag), ARGS)
end

function _run_cmd(cmd::Cmd; label::String)
    println("[prepare] $label")
    try
        run(cmd)
        return true
    catch err
        @warn "$label failed" exception=(err, catch_backtrace())
        return false
    end
end

function _run_script(path::String; label::String)
    println("\n[run] $label")
    include(path)
end

root = normpath(joinpath(@__DIR__, ".."))
cd(root) do
    want_r = _has_flag("--r") || _has_flag("--all") || isempty(ARGS)
    want_py = _has_flag("--python") || _has_flag("--all") || isempty(ARGS)
    do_prepare = _has_flag("--prepare")

    println("Gideon validation runner")
    println("  root: $root")
    println("  run R: $want_r")
    println("  run Python: $want_py")
    println("  prepare fixtures: $do_prepare")

    if do_prepare && want_r
        if isnothing(Sys.which("Rscript"))
            @warn "Rscript not found; skipping R fixture generation"
        else
            _run_cmd(`Rscript validation/fixtures_r.R`; label="R fixtures")
        end
    end

    if do_prepare && want_py
        py = get(ENV, "PYTHON", "python3")
        if isnothing(Sys.which(py))
            @warn "Python executable not found; skipping Python fixture generation" py
        else
            _run_cmd(`$py validation/fixtures_py.py`; label="Python fixtures")
        end
    end

    if want_r
        _run_script(joinpath(@__DIR__, "validate_r.jl"); label="R reference comparison")
    end

    if want_py
        _run_script(joinpath(@__DIR__, "validate_py.jl"); label="Python reference comparison")
    end

    println("\nValidation run complete.")
end
