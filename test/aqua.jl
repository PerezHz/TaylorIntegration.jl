using Test
using TaylorIntegration
using Aqua

@testset "Aqua tests (performance)" begin
    # This tests that we don't accidentally run into
    # https://github.com/JuliaLang/julia/issues/29393
    # Aqua.test_unbound_args(TaylorIntegration)
    ua = Aqua.detect_unbound_args_recursively(TaylorIntegration)
    @test length(ua) == 0

    # See: https://github.com/SciML/OrdinaryDiffEq.jl/issues/1750
    # Test that we're not introducing method ambiguities across deps
    ambs = Aqua.detect_ambiguities(TaylorIntegration; recursive = true)
    pkg_match(pkgname, pkdir::Nothing) = false
    pkg_match(pkgname, pkdir::AbstractString) = occursin(pkgname, pkdir)
    filter!(x -> pkg_match("TaylorIntegration", pkgdir(last(x).module)), ambs)
    for method_ambiguity in ambs
        @show method_ambiguity
    end
    if VERSION < v"1.10.0-DEV"
        @test length(ambs) == 0
    end
end

@testset "Aqua tests (additional)" begin
    Aqua.test_ambiguities(TaylorIntegration)
    Aqua.test_all(
    TaylorIntegration;
    ambiguities=false, # test ambiguities separately
    stale_deps=(ignore=[:DiffEqBase, :RecursiveArrayTools, :Requires, :StaticArrays],),
    )
end
