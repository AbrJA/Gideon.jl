# test/test_quality.jl — Aqua.jl and JET static analysis

@testset "Aqua.jl" begin
    # 1. Gather all stdlib names dynamically
    stdlibs = [Symbol(name) for (uuid, (name, _)) in Pkg.Types.stdlibs()]

    # 2. Call the dependencies test explicitly, completely disabling extras tracking
    @testset "Compatibility" begin
        Aqua.test_deps_compat(
            Gideon;
            ignore = stdlibs,
            check_extras = false, # This flag will FINALLY be respected here
            check_weakdeps = true
        )
    end

    # 3. Call all other standard Aqua checks separately
    @testset "Unbound type parameters" begin Aqua.test_unbound_args(Gideon) end
    @testset "Undefined exports"       begin Aqua.test_undefined_exports(Gideon) end
    @testset "Stale dependencies"      begin Aqua.test_stale_deps(Gideon) end
    @testset "Piracy"                  begin Aqua.test_piracies(Gideon) end
end

@testset "JET" begin
    JET.test_package(Gideon; target_modules=(Gideon,))
end
