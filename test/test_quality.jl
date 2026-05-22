# test/test_quality.jl — Aqua.jl and JET static analysis

@testset "Aqua" begin
    Aqua.test_all(Gideon; ambiguities=false)
end

@testset "JET" begin
    JET.test_package(Gideon; target_modules=(Gideon,))
end
