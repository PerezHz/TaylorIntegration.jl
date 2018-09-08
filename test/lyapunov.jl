# This file is part of the TaylorIntegration.jl package; MIT licensed

using TaylorIntegration
using LinearAlgebra: norm, tr, dot, istriu, diag, I
using Test

const _order = 28
const _abstol = 1.0E-20

#Lorenz system parameters
const σ = 16.0
const β = 4.0
const ρ = 45.92

#Lorenz system ODE:
function lorenz!(t, x, dx)
    dx[1] = σ*(x[2]-x[1])
    dx[2] = x[1]*(ρ-x[3])-x[2]
    dx[3] = x[1]*x[2]-β*x[3]
    nothing
end

#Lorenz system Jacobian (in-place):
function lorenz_jac!(jac, t, x, dx)
    jac[1,1] = -σ+0.0*x[1]; jac[1,2] = σ+0.0*x[1]; jac[1,3] = 0.0*x[1]
    jac[2,1] = ρ-x[3]; jac[2,2] = -1.0+0.0*x[1]; jac[2,3] = -x[1]
    jac[3,1] = x[2]; jac[3,2] = x[1]; jac[3,3] = -β+0.0*x[1]
    nothing
end

@testset "Test `stabilitymatrix!`" begin
    t0 = rand() #the initial time
    xi = set_variables("δ", order=1, numvars=3)
    t_ = Taylor1([t0,1],_order)
    δx = Array{TaylorN{Taylor1{Float64}}}(undef, 3)
    dδx = similar(δx)
    lorenzjac = Array{Taylor1{Float64}}(undef, 3, 3)
    for i in 1:10
        x0 = 10rand(3) #the initial condition
        x0T = Taylor1.(x0,_order)
        TaylorIntegration.stabilitymatrix!(lorenz!, t_, view(x0T,:), δx, dδx, lorenzjac)
        @test tr(lorenzjac.()) == -(1+σ+β)
    end
end

@testset "Test `classicalGS!`" begin
    dof = 3
    jt = rand(dof,dof)
    QH = Array{eltype(jt)}(undef, dof, dof)
    RH = Array{eltype(jt)}(undef, dof, dof)
    aⱼ = Array{eltype(jt)}(undef, dof)
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
    q0 = [19.0, 20.0, 50.0] #the initial condition
    t0 = 0.0 #the initial time
    tmax = t0+20.0 #final time of integration
    dof = length(q0) # degrees of freedom
    Q0 = Matrix{Float64}(I, dof, dof) # dof x dof identity matrix
    x0 = vcat(q0, reshape(Q0, dof*dof)) # initial conditions: eqs of motion and variationals

    #Taylor1 variables for evaluation of eqs of motion
    t = Taylor1(_order)
    x = Taylor1.(x0, _order)
    dx = similar(x)

    #Calculate trace of Lorenz system Jacobian via TaylorSeries.jacobian:
    xi = set_variables("δ", order=1, numvars=length(q0))
    q0TN = [ q0[1]+xi[1], q0[2]+xi[2], q0[3]+xi[3] ]
    dq0TN = similar(q0TN)
    lorenz!(t0, q0TN, dq0TN)
    jac_autodiff = jacobian(dq0TN) # Jacobian, computed via automatic differentiation
    lorenztr = tr(jac_autodiff) #trace of Lorenz system Jacobian matrix

    @test lorenztr == -(1+σ+β)

    # Compute Jacobian using previously defined `lorenz_jac!` function
    jac = Matrix{Taylor1{eltype(q0)}}(undef, dof, dof)
    lorenz_jac!(jac, t, x, dx)

    # Jacobian values should be equal
    @test constant_term.(jac) == jac_autodiff

    # Number of TaylorN variables should be equal to length of vector of initial conditions
    xi = set_variables("δ", order=1, numvars=length(q0)-1)
    @test_throws AssertionError lyap_taylorinteg(lorenz!, q0, t0, tmax, _order, _abstol; maxsteps=2)
    xi = set_variables("δ", order=1, numvars=length(q0))

    #Test lyap_taylorinteg with autodiff-computed Jacobian
    tv, xv, λv = lyap_taylorinteg(lorenz!, q0, t0, tmax, _order, _abstol; maxsteps=2)
    @test size(tv) == (3,)
    @test size(xv) == (3,3)
    @test size(λv) == (3,3)
    tv, xv, λv = lyap_taylorinteg(lorenz!, q0, t0, tmax, _order, _abstol; maxsteps=2000)
    @time tv, xv, λv = lyap_taylorinteg(lorenz!, q0, t0, tmax, _order, _abstol; maxsteps=2000)
    @test xv[1,:] == q0
    @test tv[1] == t0
    @test size(xv) == size(λv)
    @test isapprox(sum(λv[1,:]), lorenztr) == false
    @test isapprox(sum(λv[end,:]), lorenztr)
    mytol = 1e-4
    @test isapprox(λv[end,1], 1.47167, rtol=mytol, atol=mytol)
    @test isapprox(λv[end,2], -0.00830, rtol=mytol, atol=mytol)
    @test isapprox(λv[end,3], -22.46336, rtol=mytol, atol=mytol)
    # Check integration consistency (orbit should not depend on variational eqs)
    t_, x_ = taylorinteg(lorenz!, q0, t0, tmax, _order, _abstol; maxsteps=2000)
    @test t_ == tv
    @test x_ == xv

    #Test lyap_taylorinteg with user-defined Jacobian
    tv, xv, λv = lyap_taylorinteg(lorenz!, q0, t0, tmax, _order, _abstol, lorenz_jac!; maxsteps=2)
    @test size(tv) == (3,)
    @test size(xv) == (3,3)
    @test size(λv) == (3,3)
    tv, xv, λv = lyap_taylorinteg(lorenz!, q0, t0, tmax, _order, _abstol, lorenz_jac!; maxsteps=2000)
    @time tv, xv, λv = lyap_taylorinteg(lorenz!, q0, t0, tmax, _order, _abstol, lorenz_jac!; maxsteps=2000)
    @test xv[1,:] == q0
    @test tv[1] == t0
    @test size(xv) == size(λv)
    @test isapprox(sum(λv[1,:]), lorenztr) == false
    @test isapprox(sum(λv[end,:]), lorenztr)
    mytol = 1e-4
    @test isapprox(λv[end,1], 1.47167, rtol=mytol, atol=mytol)
    @test isapprox(λv[end,2], -0.00830, rtol=mytol, atol=mytol)
    @test isapprox(λv[end,3], -22.46336, rtol=mytol, atol=mytol)
    # Check integration consistency (orbit should not depend on variational eqs)
    # t_, x_ = taylorinteg(lorenz!, q0, t0, tmax, _order, _abstol; maxsteps=2000)
    @test t_ == tv
    @test x_ == xv
end

@testset "Test Lyapunov spectrum integrator (trange): Lorenz system" begin
    x0 = [19.0, 20.0, 50.0] #the initial condition
    t0 = 0.0 #the initial time
    tmax = t0+20.0 #final time of integration

    #Calculate trace of Lorenz system Jacobian via TaylorSeries.jacobian:
    xi = set_variables("δ", order=1, numvars=length(x0))
    x0TN = x0+xi
    dx0TN = similar(x0TN)
    lorenz!(t0, x0TN, dx0TN)
    lorenztr = tr(jacobian(dx0TN)) #trace of Lorenz system Jacobian matrix

    @test lorenztr == -(1+σ+β)

    xw, λw = lyap_taylorinteg(lorenz!, x0, t0:1.0:tmax, _order, _abstol; maxsteps=2)

    @test size(xw) == (length(t0:1.0:tmax), 3)
    @test size(λw) == (length(t0:1.0:tmax), 3)
    @test prod(isnan.(xw[2:end,:]))

    xw, λw = lyap_taylorinteg(lorenz!, x0, t0:1.0:tmax, _order, _abstol; maxsteps=2000)

    @test xw[1,:] == x0
    @test size(xw) == size(λw)
    @test isapprox(sum(λw[1,:]), lorenztr) == false
    @test isapprox(sum(λw[end,:]), lorenztr)
    mytol = 1e-4
    @test isapprox(λw[end,1], 1.47167, rtol=mytol, atol=mytol)
    @test isapprox(λw[end,2], -0.00830, rtol=mytol, atol=mytol)
    @test isapprox(λw[end,3], -22.46336, rtol=mytol, atol=mytol)

    xw2, λw2 = lyap_taylorinteg(lorenz!, x0, collect(t0:1.0:tmax), _order, _abstol; maxsteps=2000)

    @test xw2 == xw
    @test λw2 == λw

    @test xw2[1,:] == x0
    @test size(xw2) == size(λw2)
    @test isapprox(sum(λw2[1,:]), lorenztr) == false
    @test isapprox(sum(λw2[end,:]), lorenztr)
    mytol = 1e-4
    @test isapprox(λw2[end,1], 1.47167, rtol=mytol, atol=mytol)
    @test isapprox(λw2[end,2], -0.00830, rtol=mytol, atol=mytol)
    @test isapprox(λw2[end,3], -22.46336, rtol=mytol, atol=mytol)

    # Check integration consistency (orbit should not depend on variational eqs)
    x_ = taylorinteg(lorenz!, x0, collect(t0:1.0:tmax), _order, _abstol; maxsteps=2000)
    @test x_ == xw2
end
