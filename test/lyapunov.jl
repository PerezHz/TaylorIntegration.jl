# This file is part of the TaylorIntegration.jl package; MIT licensed

using TaylorSeries, TaylorIntegration
using Base.Test

const _order = 28
const _abstol = 1.0E-20

@testset "Test `stabilitymatrix!`" begin
    function lorenz!(t, x, dx) #Lorenz system ODE:
        dx[1] = σ*(x[2]-x[1])
        dx[2] = x[1]*(ρ-x[3])-x[2]
        dx[3] = x[1]*x[2]-β*x[3]
        nothing
    end
    σ = 16.0; β = 4.0; ρ = 45.92 #Lorenz system parameters
    t0 = 0.0 #the initial time
    for i in 1:10
        x0 = 10rand(3) #the initial condition
        xi = set_variables("δ", order=1, numvars=length(x0))
        δx = [ x0[1]+xi[1], x0[2]+xi[2], x0[3]+xi[3] ]
        dδx = similar(δx)
        lorenzjac = Array{eltype(x0)}(3,3)
        TaylorIntegration.stabilitymatrix!(lorenz!, t0, x0, δx, dδx, lorenzjac)
        @test trace(lorenzjac) == -(σ+one(Float64)+β)
    end
end

@testset "Test `classicalGS!`" begin
    dof = 3
    jt = rand(dof,dof)
    QH = Array{eltype(jt)}(dof,dof)
    RH = Array{eltype(jt)}(dof,dof)
    aⱼ = Array{eltype(jt)}(dof)
    qᵢ = similar(aⱼ)
    vⱼ = similar(aⱼ)
    TaylorIntegration.classicalGS!( jt, QH, RH, aⱼ, qᵢ, vⱼ )
    @test norm(jt-QH*RH, Inf) < 1e-14
    for i in 1:dof
        for j in i:dof
            if j == i
                @test isapprox( dot(QH[:,i],QH[:,j]), one(eltype(jt)) )
            else
                @test abs( dot(QH[:,i],QH[:,j]) ) < 1E-10
            end
        end
    end
    @test istriu(RH) #is RH an upper triangular matrix?
    @test prod(diag(RH) .> 0.0) #are the diagonal elements of RH positive?
end

@testset "Test Lyapunov spectrum integrator (t0, tmax): Lorenz system" begin

    x0 = [19.0, 20.0, 50.0] #the initial condition
    t0 = 0.0 #the initial time
    tmax = t0+20.0 #final time of integration

    #Lorenz system ODE:
    function lorenz!(t, x, dx)
        dx[1] = σ*(x[2]-x[1])
        dx[2] = x[1]*(ρ-x[3])-x[2]
        dx[3] = x[1]*x[2]-β*x[3]
        nothing
    end

    #Lorenz system parameters
    σ = 16.0
    β = 4.0
    ρ = 45.92

    #Calculate trace of Lorenz system Jacobian via TaylorSeries.jacobian:
    xi = set_variables("δ", order=1, numvars=length(x0))
    x0TN = [ x0[1]+xi[1], x0[2]+xi[2], x0[3]+xi[3] ]
    dx0TN = similar(x0TN)
    lorenz!(t0, x0TN, dx0TN)
    lorenztr = trace(jacobian(dx0TN)) #trace of Lorenz system Jacobian matrix

    @test lorenztr == -(σ+one(Float64)+β)

    tv, xv, λv = liap_taylorinteg(lorenz!, x0, t0, tmax, 28, _abstol; maxsteps=2)

    @test size(tv) == (3,)
    @test size(xv) == (3,3)
    @test size(λv) == (3,3)

    tv, xv, λv = liap_taylorinteg(lorenz!, x0, t0, tmax, 28, _abstol; maxsteps=2000)

    @test xv[1,:] == x0
    @test tv[1] == t0
    @test size(xv) == size(λv)
    @test isapprox(sum(λv[1,:]), lorenztr) == false
    @test isapprox(sum(λv[end,:]), lorenztr)
    mytol = 1e-4
    @test isapprox(λv[end,1], 1.47167, rtol=mytol, atol=mytol)
    @test isapprox(λv[end,2], -0.00830, rtol=mytol, atol=mytol)
    @test isapprox(λv[end,3], -22.46336, rtol=mytol, atol=mytol)
end

@testset "Test Lyapunov spectrum integrator (trange): Lorenz system" begin

    x0 = [19.0, 20.0, 50.0] #the initial condition
    t0 = 0.0 #the initial time
    tmax = t0+20.0 #final time of integration

    #Lorenz system ODE:
    function lorenz!(t, x, dx)
        dx[1] = σ*(x[2]-x[1])
        dx[2] = x[1]*(ρ-x[3])-x[2]
        dx[3] = x[1]*x[2]-β*x[3]
        nothing
    end

    #Lorenz system parameters
    σ = 16.0
    β = 4.0
    ρ = 45.92

    #Calculate trace of Lorenz system Jacobian via TaylorSeries.jacobian:
    xi = set_variables("δ", order=1, numvars=length(x0))
    x0TN = [ x0[1]+xi[1], x0[2]+xi[2], x0[3]+xi[3] ]
    dx0TN = similar(x0TN)
    lorenz!(t0, x0TN, dx0TN)
    lorenztr = trace(jacobian(dx0TN)) #trace of Lorenz system Jacobian matrix

    @test lorenztr == -(σ+one(Float64)+β)

    xw, λw = liap_taylorinteg(lorenz!, x0, t0:1.0:tmax, 28, _abstol; maxsteps=2)

    @test size(xw) == (length(t0:1.0:tmax), 3)
    @test size(λw) == (length(t0:1.0:tmax), 3)
    @test prod(isnan.(xw[2:end,:]))

    xw, λw = liap_taylorinteg(lorenz!, x0, t0:1.0:tmax, 28, _abstol; maxsteps=2000)

    @test xw[1,:] == x0
    @test size(xw) == size(λw)
    @test isapprox(sum(λw[1,:]), lorenztr) == false
    @test isapprox(sum(λw[end,:]), lorenztr)
    mytol = 1e-4
    @test isapprox(λw[end,1], 1.47167, rtol=mytol, atol=mytol)
    @test isapprox(λw[end,2], -0.00830, rtol=mytol, atol=mytol)
    @test isapprox(λw[end,3], -22.46336, rtol=mytol, atol=mytol)
end
