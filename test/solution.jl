# This file is part of the TaylorIntegration.jl package; MIT licensed

using TaylorIntegration
using TaylorSeries
using Test
using Logging
using JLD2
import Logging: Warn

@testset "Testing `solution.jl`" begin
    tv = [1.0, 2]
    xv = rand(2, 2)
    psol = Taylor1.(rand(2, 1), 2)
    nsteps = 2
    sol2 = TaylorIntegration.build_solution(tv, xv, psol, nsteps)
    @test sol2 isa TaylorSolution{Float64,Float64,2}
    @test string(sol2) == "tspan: (1.0, 2.0), x: 2 Float64 variables"
    sol1 =
        TaylorIntegration.build_solution(tv, Vector(xv[1, :]), Vector(psol[1, :]), nsteps)
    @test sol1 isa TaylorSolution{Float64,Float64,1}
    @test string(sol1) == "tspan: (1.0, 2.0), x: 1 Float64 variable"
    @test firsttime(sol1) == first(sol1.t)
    @test lasttime(sol1) == last(sol1.t)
    sol_big = TaylorSolution(BigFloat[0], BigFloat[1])
    @test zero(typeof(sol_big)).x == BigFloat[0]
    @test firsttime(sol_big) == BigFloat(0)
    @test lasttime(sol_big) == BigFloat(0)

    ξ = variables!("ξ", numvars = 2, order = 2)
    p_scalar = Vector{Taylor1{TaylorN{Float64}}}(undef, 1)
    x_scalar = Taylor1(1.0 + ξ[1], 3)
    TaylorIntegration.set_psol!(Val(true), p_scalar, 1, x_scalar)
    x_scalar[0][0] = 10.0
    @test p_scalar[1][0][0] == 1.0
    TaylorIntegration.set_psol!(Val(true), p_scalar, 1, x_scalar)
    @test p_scalar[1] == x_scalar
    stored_scalar = p_scalar[1]
    x_scalar_next = Taylor1(3.0 + ξ[2], 3)
    TaylorIntegration.set_psol!(Val(true), p_scalar, 1, x_scalar_next)
    @test stored_scalar == x_scalar
    @test p_scalar[1] == x_scalar_next
    @test p_scalar[1] !== stored_scalar

    p_vector = Matrix{Taylor1{TaylorN{Float64}}}(undef, 2, 1)
    x_vector = [Taylor1(1.0 + ξ[1], 3), Taylor1(2.0 + ξ[2], 3)]
    TaylorIntegration.set_psol!(Val(true), p_vector, 1, x_vector)
    x_vector[1][0][0] = 20.0
    @test p_vector[1, 1][0][0] == 1.0
    TaylorIntegration.set_psol!(Val(true), p_vector, 1, x_vector)
    @test p_vector[:, 1] == x_vector
    stored_vector = p_vector[:, 1]
    x_vector_next = [Taylor1(4.0 + ξ[1], 3), Taylor1(5.0 + ξ[2], 3)]
    TaylorIntegration.set_psol!(Val(true), p_vector, 1, x_vector_next)
    @test stored_vector == x_vector
    @test p_vector[:, 1] == x_vector_next
    @test all(p_vector[:, 1] .!== stored_vector)

    xv_store = Matrix{TaylorN{Float64}}(undef, 2, 1)
    x_state = [1.0 + ξ[1], 2.0 + ξ[2]]
    TaylorIntegration._store_state_column!(xv_store, 1, x_state)
    x_state[1][0] = 30.0
    @test xv_store[1, 1][0] == 1.0
    TaylorIntegration._store_state_column!(xv_store, 1, x_state)
    @test xv_store[:, 1] == x_state

    xt1 = Taylor1.([1.0, 2.0], 2)
    sol_t1 = TaylorIntegration.build_solution(tv, xt1)
    @test sol_t1.x == xt1
    @test isnothing(sol_t1.p)
    sol_t1_view = TaylorIntegration.build_solution(
        [1.0, 2.0, 3.0],
        Taylor1.([1.0, 2.0, 3.0], 2),
        nothing,
        2,
    )
    @test sol_t1_view.x == xt1
    @test isnothing(sol_t1_view.p)
    xt1_matrix = [
        Taylor1(1.0, 2) Taylor1(2.0, 2)
        Taylor1(3.0, 2) Taylor1(4.0, 2)
    ]
    sol_t1_matrix = TaylorIntegration.build_solution(tv, xt1_matrix)
    @test sol_t1_matrix.x == transpose(xt1_matrix)
    @test isnothing(sol_t1_matrix.p)

    tv = 0.1:0.1:1.1
    xv = rand(2, length(tv))
    sol = TaylorIntegration.build_solution(tv, xv, Taylor1.(xv, 2), 9)
    t1 = 0.35 + Taylor1(variables()[1], 2)
    ind, δt = TaylorIntegration.timeindex(sol, t1)
    @test ind == 3
    @test δt == t1 - sol.t[ind]
    sol_bad_time = TaylorSolution(
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
        Taylor1.([0.0, 1.0], 1),
    )
    sol_bad_time.t[2] = -1.0
    @test_throws ArgumentError TaylorIntegration.timeindex(sol_bad_time, 0.5)

    tv = collect((0:0.25:2) * pi)
    xv = Matrix(hcat(sin.(tv), cos.(tv))')
    psolv = Matrix(hcat(sin.(tv .+ Taylor1(25)), cos.(tv .+ Taylor1(25)))')
    sol = TaylorIntegration.build_solution(tv, xv, psolv, length(tv) - 2)
    @test sol(sol.t[1]) == sol.x[1, :]
    @test norm(sol(sol.t[end]) - sol.x[end, :], Inf) < 1e-14

    soldense = TaylorSolution(collect(sol.t), collect(sol.p))
    soldense_copy = TaylorSolution(copy(soldense.t), copy(soldense.x), copy(soldense.p))
    @test soldense isa TaylorIntegration.DensePropagation2{Float64,Float64}
    @test maximum(abs.(soldense.x .- sol.x)) < 2e-16
    @test soldense_copy == soldense
    @test hash(soldense_copy) == hash(soldense)
    @test TaylorSeries.order(soldense) == TaylorSeries.order(first(sol.p))
    @test iszero(zero(typeof(soldense)))
    @test convert(BigFloat, soldense).t == BigFloat.(soldense.t)
    @test convert(BigFloat, soldense).p[1] isa Taylor1{BigFloat}
    @test firsttime(soldense) == first(soldense.t)
    @test lasttime(soldense) == last(soldense.t)

    empty_p = Taylor1{Float64}[]
    @test (@inferred TaylorSolution([0.0], empty_p)).x == [0.0]
    sol_empty = TaylorSolution([0.0], [1.0], empty_p)
    @test sol_empty.x == [1.0]
    ind_empty, δt_empty = TaylorIntegration.timeindex(sol_empty, 0.0)
    @test ind_empty == 1
    @test δt_empty == 0.0
    @test sol_empty(0.0) == 1.0

    empty_matrix_p = Matrix{Taylor1{Float64}}(undef, 0, 2)
    @test (@inferred TaylorSolution([0.0], empty_matrix_p)).x == zeros(1, 2)
    sol_empty_matrix = TaylorSolution([0.0], [1.0 2.0], empty_matrix_p)
    @test sol_empty_matrix.x == [1.0 2.0]
    @test sol_empty_matrix(0.0) == [1.0, 2.0]

    p_mismatched_endpoints = [
        Taylor1([1.0, 10.0], 1),
        Taylor1([2.0, 20.0], 1),
    ]
    @test TaylorSolution([0.0, 1.0, 2.0], p_mismatched_endpoints).x == [1.0, 2.0, 22.0]
    @test TaylorSolution([0.0, 1.0, 2.0], p_mismatched_endpoints) isa
          TaylorIntegration.DensePropagation1{Float64,Float64}
    @test_throws ArgumentError TaylorSolution([0.0, 1.0], p_mismatched_endpoints)
    @test (@inferred TaylorSolution([0.0, 1.0], p_mismatched_endpoints, nothing)).x ==
          p_mismatched_endpoints

    p_matrix_mismatched_endpoints = [
        Taylor1([1.0, 10.0], 1) Taylor1([4.0, 40.0], 1)
        Taylor1([2.0, 20.0], 1) Taylor1([5.0, 50.0], 1)
    ]
    @test TaylorSolution([0.0, 1.0, 2.0], p_matrix_mismatched_endpoints).x ==
          [1.0 4.0; 2.0 5.0; 22.0 55.0]

    solrev = reverse(soldense)
    @test solrev.t == reverse(soldense.t)
    @test firsttime(solrev) == lasttime(soldense)
    @test lasttime(solrev) == firsttime(soldense)
    @test norm(solrev(solrev.t[end]) - soldense(soldense.t[1]), Inf) < 1e-14

    solflip = flipsign(soldense)
    @test solflip.t == 2 * first(soldense.t) .- soldense.t
    @test solflip.p == soldense.p(-Taylor1(TaylorSeries.order(soldense)))
    soldense_shifted = TaylorSolution(collect(soldense.t) .- 3, collect(soldense.p))
    solflip_shifted = flipsign(soldense_shifted)
    @test solflip_shifted.t == 2 * first(soldense_shifted.t) .- soldense_shifted.t
    @test issorted(solflip_shifted.t) || issorted(solflip_shifted.t, rev = true)

    jld2_path = tempname() * ".jld2"
    jldsave(jld2_path; soldense)
    solfile = JLD2.load(jld2_path, "soldense")
    rm(jld2_path)
    @test solfile == soldense

    jld2_path = tempname() * ".jld2"
    jldsave(jld2_path; sol)
    solfile = JLD2.load(jld2_path, "sol")
    rm(jld2_path)
    @test solfile == soldense
    @test solfile.t isa Vector
    @test solfile.p isa Array

    dq = TaylorSeries.variables!("dq", order = 2, numvars = 2)
    pN = soldense.p .* Taylor1(one(dq[1]), TaylorSeries.order(soldense))
    solN = TaylorSolution(soldense.t, pN)
    jld2_path = tempname() * ".jld2"
    jldsave(jld2_path; solN)
    solNfile = JLD2.load(jld2_path, "solN")
    rm(jld2_path)
    @test solNfile == solN

    solN_view = TaylorIntegration.build_solution(
        collect(solN.t),
        permutedims(solN.x),
        permutedims(solN.p),
        length(solN.t),
    )
    jld2_path = tempname() * ".jld2"
    jldsave(jld2_path; solN_view)
    solNfile = JLD2.load(jld2_path, "solN_view")
    rm(jld2_path)
    @test solNfile == solN
    @test solNfile.t isa Vector
    @test solNfile.p isa Array
end
