# This file is part of the TaylorIntegration.jl package; MIT licensed

using TaylorIntegration
using Test
using LinearAlgebra: Transpose

const _order = 28
const _abstol = 1.0E-20
const tT = Taylor1(_order)

@testset "Tests: dot{x}=x.^2, x(0) = [3.0,1.0]" begin
    function eqs_mov!(t, x, Dx)
        for i in eachindex(x)
            Dx[i] = x[i]^2
        end
        nothing
    end
    exactsol(t, x0) = x0/(1.0-x0*t)
    t0 = 0.0
    q0 = [3.0, 1.0]
    q0T = [Taylor1(q0[1], _order), Taylor1(q0[2], _order)]
    xdotT = Array{Taylor1{Float64}}(undef, length(q0))
    xaux = Array{Taylor1{Float64}}(undef, length(q0))
    tT[1] = t0
    TaylorIntegration.jetcoeffs!(eqs_mov!, tT, q0T, xdotT, xaux)
    @test q0T[1].coeffs[end] == 3.0^(_order+1)
    @test q0T[2].coeffs[end] == 1.0
    δt = (_abstol/q0T[1].coeffs[end-1])^inv(_order-1)
    @test TaylorIntegration.stepsize(q0T, _abstol) == δt

    tv, xv = taylorinteg(eqs_mov!, q0, 0.0, 0.5, _order, _abstol)
    @test length(tv) == 501
    @test isa(xv, SubArray)
    @test xv[1,:] == q0
    @test tv[end] < 1/3

    trange = 0.0:1/8:1.0
    xv = taylorinteg(eqs_mov!, q0, trange, _order, _abstol)
    @test size(xv) == (9,2)
    @test q0 == [3.0, 1.0]
    @test typeof(xv) == Transpose{eltype(q0), Array{eltype(q0),2}}
    @test xv[1,1:end] == q0
    @test (isnan(xv[4,1]) && isnan(xv[4,2]))
    @test (isnan(xv[end,1]) && isnan(xv[end,2]))
    @test abs(xv[3,2] - 4/3) ≤ eps(4/3)
    @test abs(xv[2,1] - 4.8) ≤ eps(4.8)

    tarray = vec(trange)
    xv2 = taylorinteg(eqs_mov!, q0, tarray, _order, _abstol)
    @test xv[1:3,:] == xv2[1:3,:]
    @test xv2[1:3,:] ≈ xv[1:3,:] atol=eps() rtol=0.0
    @test size(xv2) == (9,2)
    @test q0 == [3.0, 1.0]
    @test typeof(xv2) == Transpose{eltype(q0), Array{eltype(q0),2}}
    @test xv2[1,1:end] == q0
    @test (isnan(xv2[4,1]) && isnan(xv2[4,2]))
    @test (isnan(xv2[end,1]) && isnan(xv2[end,2]))
    @test abs(xv2[3,2] - 4/3) ≤ eps(4/3)
    @test abs(xv2[2,1] - 4.8) ≤ eps(4.8)
end

@testset "Test non-autonomous ODE (2): dot{x}=cos(t)" begin
    function f!(t, x, Dx)
        Dx[1] = one(t)
        Dx[2] = cos(t)
        nothing
    end
    t0 = 0//1
    tmax = 10.25*(2pi)
    abstol = 1e-20
    order = 25
    x0 = [t0, 0.0] #initial conditions such that x(t)=sin(t)
    tv, xv = taylorinteg(f!, x0, t0, tmax, order, abstol)
    @test length(tv) < 501
    @test length(xv[:,1]) < 501
    @test length(xv[:,2]) < 501
    @test xv[1,1:end] == x0
    @test tv[1] == t0
    @test xv[1,1] == x0[1]
    @test xv[1,2] == x0[2]
    @test tv[end] == xv[end,1]
    @test abs(sin(tmax)-xv[end,2]) < 1e-14

    tmax = 15*(2pi)
    tv, xv = taylorinteg(f!, x0, t0, tmax, order, abstol)
    @test length(tv) < 501
    @test length(xv[:,1]) < 501
    @test length(xv[:,2]) < 501
    @test xv[1,1:end] == x0
    @test tv[1] == t0
    @test xv[1,1] == x0[1]
    @test xv[1,2] == x0[2]
    @test tv[end] == xv[end,1]
    @test abs(sin(tmax)-xv[end,2]) < 1e-14
end
