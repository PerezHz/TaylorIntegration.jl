# This file is part of the TaylorIntegration.jl package; MIT licensed

using TaylorIntegration
using LinearAlgebra: norm, tr, dot, istriu, diag, I
using Test
using Logging
import Logging: Warn

@testset "Testing `lyapunov.jl`" begin

    max_iters_reached() = "Maximum number of integration steps reached; exiting.\n"

    local _order = 28
    local _abstol = 1.0E-20

    #Lorenz system parameters
    local σ = 16.0
    local β = 4.0
    local ρ = 45.92

    #Lorenz system ODE:
    function lorenz!(dx, x, p, t)
        dx[1] = σ*(x[2]-x[1])
        dx[2] = x[1]*(ρ-x[3])-x[2]
        dx[3] = x[1]*x[2]-β*x[3]
        nothing
    end

    #Lorenz system Jacobian (in-place):
    function lorenz_jac!(jac, x, p, t)
        jac[1,1] = -σ+zero(x[1]); jac[1,2] = σ+zero(x[1]); jac[1,3] = zero(x[1])
        jac[2,1] = ρ-x[3]; jac[2,2] = -1.0+zero(x[1]); jac[2,3] = -x[1]
        jac[3,1] = x[2]; jac[3,2] = x[1]; jac[3,3] = -β+zero(x[1])
        nothing
    end

    @testset "Test `stabilitymatrix!`" begin
        t0 = rand() #the initial time
        xi = set_variables("δ", order=1, numvars=3)
        t_ = Taylor1([t0,1],_order)
        δx = Array{TaylorN{Taylor1{Float64}}}(undef, 3)
        dδx = similar(δx)
        jac_auto = Array{Taylor1{Float64}}(undef, 3, 3)
        jac_user = Array{Taylor1{Float64}}(undef, 3, 3)
        _δv = Array{TaylorN{Taylor1{Float64}}}(undef, 3)
        for ind in 1:3
            _δv[ind] = one(t_)*TaylorN(Taylor1{Float64}, ind, order=1)
        end
        # test computation of jac via: autodiff and user-provided Jacobian function
        for i in 1:10
            x0 = 10rand(3) #the initial condition
            x0T = Taylor1.(x0,_order)
            TaylorIntegration.stabilitymatrix!(lorenz!, t_, x0T, δx, dδx, jac_auto, _δv, nothing)
            @test tr(constant_term.(jac_auto)) == -(1+σ+β)
            TaylorIntegration.stabilitymatrix!(lorenz!, t_, x0T, δx, dδx, jac_user, _δv, nothing, lorenz_jac!)
            @test tr(constant_term.(jac_user)) == -(1+σ+β)
            @test jac_user == jac_auto
        end
    end

    @testset "Test `classicalGS!`" begin
        dof = 3
        jt = rand(dof, dof)
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
        lorenz!(dq0TN, q0TN, nothing, t0)
        jac_autodiff = TaylorSeries.jacobian(dq0TN) # Jacobian, computed via automatic differentiation
        lorenztr = tr(jac_autodiff) #trace of Lorenz system Jacobian matrix

        @test lorenztr == -(1+σ+β)

        # Compute Jacobian using previously defined `lorenz_jac!` function
        jac = Matrix{Taylor1{eltype(q0)}}(undef, dof, dof)
        lorenz_jac!(jac, x, nothing, t)
        @test tr(constant_term.(jac)) == -(1+σ+β)

        # Jacobian values should be equal
        @test constant_term.(jac) == jac_autodiff

        # Number of TaylorN variables should be equal to length of vector of initial conditions
        xi = set_variables("δ", order=1, numvars=length(q0)-1)
        @test_throws AssertionError lyap_taylorinteg(lorenz!, q0, t0, tmax, _order, _abstol; maxsteps=2)
        xi = set_variables("δ", order=1, numvars=length(q0))

        #Test lyap_taylorinteg with autodiff-computed Jacobian, maxsteps=5
        sol = (@test_logs (Warn, max_iters_reached()) lyap_taylorinteg(
            lorenz!, q0, t0, tmax, _order, _abstol, nothing; maxsteps=5))
        tv, xv, λv = sol.t, sol.x, sol.λ
        @test size(tv) == (6,)
        @test size(xv) == (6,3)
        @test size(λv) == (6,3)

        #Test lyap_taylorinteg with user-defined Jacobian, maxsteps=5
        sol2 = (@test_logs (Warn, max_iters_reached()) lyap_taylorinteg(
            lorenz!, q0, t0, tmax, _order, _abstol, nothing, lorenz_jac!; maxsteps=5))
        tv2, xv2, λv2 = sol2.t, sol2.x, sol2.λ
        @test size(tv2) == (6,)
        @test size(xv2) == (6,3)
        @test size(λv2) == (6,3)
        @test tv == tv2
        @test xv == xv2
        @test λv == λv2

        #Test lyap_taylorinteg with autodiff-computed Jacobian, maxsteps=2000
        sol = (@test_logs min_level=Logging.Warn lyap_taylorinteg(
            lorenz!, q0, t0, tmax, _order, _abstol; maxsteps=2000))
        tv, xv, λv = sol.t, sol.x, sol.λ
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
        sol_ = taylorinteg(lorenz!, q0, t0, tmax, _order, _abstol; maxsteps=2000)
        t_, x_ = sol_.t, sol_.x
        @test t_ == tv
        @test x_ == xv
        # test backward integration
        solb = (@test_logs (Warn, max_iters_reached()) lyap_taylorinteg(
            lorenz!, q0, t0, -tmax, _order, _abstol; maxsteps=2000))
        tvb, xvb, λvb = solb.t, solb.x, solb.λ
        @test isapprox(sum(λvb[1,:]), lorenztr) == false
        @test isapprox(sum(λvb[end,:]), lorenztr)

        #Test lyap_taylorinteg with user-defined Jacobian, maxsteps=2000
        sol2 = (@test_logs min_level=Logging.Warn lyap_taylorinteg(
            lorenz!, q0, t0, tmax, _order, _abstol, nothing, lorenz_jac!; maxsteps=2000))
        tv2, xv2, λv2 = sol2.t, sol2.x, sol2.λ
        @test xv2[1,:] == q0
        @test tv2[1] == t0
        @test size(xv2) == size(λv2)
        @test tv == tv2
        @test xv == xv2
        @test λv == λv2
        @test isapprox(sum(λv2[1,:]), lorenztr) == false
        @test isapprox(sum(λv2[end,:]), lorenztr)
        mytol = 1e-4
        @test isapprox(λv2[end,1], 1.47167, rtol=mytol, atol=mytol)
        @test isapprox(λv2[end,2], -0.00830, rtol=mytol, atol=mytol)
        @test isapprox(λv2[end,3], -22.46336, rtol=mytol, atol=mytol)
        # Check integration consistency (orbit should not depend on variational eqs)
        @test t_ == tv2
        @test x_ == xv2
        # test backward integration
        sol2b = (@test_logs (Warn, max_iters_reached()) lyap_taylorinteg(
            lorenz!, q0, t0, -tmax, _order, _abstol, nothing, lorenz_jac!; maxsteps=2000))
        tv2b, xv2b, λv2b = sol2b.t, sol2b.x, sol2b.λ
        @test isapprox(sum(λv2b[1,:]), lorenztr) == false
        @test isapprox(sum(λv2b[end,:]), lorenztr)
    end

    @testset "Test Lyapunov spectrum integrator (trange): Lorenz system" begin
        q0 = [19.0, 20.0, 50.0] #the initial condition
        t0 = 0.0 #the initial time
        tmax = t0+20.0 #final time of integration

        #Calculate trace of Lorenz system Jacobian via TaylorSeries.jacobian:
        xi = set_variables("δ", order=1, numvars=length(q0))
        q0TN = q0+xi
        dq0TN = similar(q0TN)
        lorenz!(dq0TN, q0TN, nothing, t0)
        lorenztr = tr(TaylorSeries.jacobian(dq0TN)) #trace of Lorenz system Jacobian matrix
        @test lorenztr == -(1+σ+β)

        trange = t0:1.0:tmax
        solw = (@test_logs (Warn, max_iters_reached()) lyap_taylorinteg(
            lorenz!, q0, trange, _order, _abstol; maxsteps=5))
        xw, λw = solw.x, solw.λ
        @test trange == solw.t
        @test xw[1,:] == q0
        @test size(xw) == (length(trange), 3)
        @test λw[1,:] == zeros(3)
        @test size(λw) == size(xw)
        @test all(isnan.(xw[2:end,:]))
        @test all(isnan.(λw[2:end,:]))
        solw2 = (@test_logs (Warn, max_iters_reached()) lyap_taylorinteg(
            lorenz!, q0, vec(trange), _order, _abstol; maxsteps=5))
        xw2, λw2 = solw2.x, solw2.λ
        @test trange == solw2.t
        @test xw[1, :] == xw2[1, :]
        @test all(isnan.(xw2[2:end,:]))
        @test λw2[1, :] == zeros(3)
        @test all(isnan.(λw2[2:end,:]))
        @test size(xw2) == (length(trange), 3)
        @test size(λw2) == size(xw2)
        solw_ = (@test_logs (Warn, max_iters_reached()) lyap_taylorinteg(
            lorenz!, q0, trange, _order, _abstol, nothing, lorenz_jac!; maxsteps=5))
        xw_, λw_ = solw_.x, solw_.λ
        @test trange == solw_.t
        @test xw_[1,:] == q0
        @test all(isnan.(xw_[2:end,:]))
        @test size(xw_) == (length(trange), 3)
        @test size(λw_) == size(xw_)
        @test λw_[1, :] == zeros(3)
        @test all(isnan.(λw_[2:end,:]))
        solw2_ = (@test_logs (Warn, max_iters_reached()) lyap_taylorinteg(
            lorenz!, q0, vec(trange), _order, _abstol, nothing, lorenz_jac!; maxsteps=5))
        xw2_, λw2_ = solw2_.x, solw2_.λ
        @test vec(trange) == solw2_.t
        @test xw_[1, :] == xw2_[1, :]
        @test all(isnan.(xw2_[2:end,:]))
        @test λw2_[1, :] == zeros(3)
        @test all(isnan.(λw2_[2:end,:]))
        @test size(xw2_) == (length(trange), 3)
        @test size(λw2_) == size(xw2_)

        solw = (@test_logs min_level=Logging.Warn lyap_taylorinteg(
            lorenz!, q0, trange, _order, _abstol; maxsteps=2000))
        xw, λw = solw.x, solw.λ
        @test trange == solw.t
        @test xw[1,:] == q0
        @test size(xw) == (length(trange), length(q0))
        @test size(λw) == (length(trange), length(q0))
        @test isapprox(sum(λw[1,:]), lorenztr) == false
        @test isapprox(sum(λw[end,:]), lorenztr)
        solz = (@test_logs min_level=Logging.Warn lyap_taylorinteg(
            lorenz!, q0, trange[1], trange[end], _order, _abstol; maxsteps=2000))
        tz, xz, λz = solz.t, solz.x, solz.λ
        @test λw[end,:] == λz[end,:]
        @test xw[end,:] == xz[end,:]
        @test tz[end] == trange[end]
        mytol = 1e-4
        @test isapprox(λw[end,1], 1.47167, rtol=mytol, atol=mytol)
        @test isapprox(λw[end,2], -0.00831, rtol=mytol, atol=mytol)
        @test isapprox(λw[end,3], -22.46336, rtol=mytol, atol=mytol)
        solw_ = (@test_logs min_level=Logging.Warn lyap_taylorinteg(
            lorenz!, q0, trange, _order, _abstol, nothing, lorenz_jac!; maxsteps=2000))
        xw_, λw_ = solw_.x, solw_.λ
        @test xw == xw_
        @test λw == λw_
        solz_ = (@test_logs min_level=Logging.Warn lyap_taylorinteg(
            lorenz!, q0, trange[1], trange[end], _order, _abstol, nothing, lorenz_jac!; maxsteps=2000))
        tz_, xz_, λz_ = solz_.t, solz_.x, solz_.λ
        @test λw_[end,:] == λz_[end,:]
        @test xw_[end,:] == xz_[end,:]
        @test tz_[end] == trange[end]

        solw2 = (@test_logs min_level=Logging.Warn lyap_taylorinteg(
            lorenz!, q0, vec(trange), _order, _abstol; maxsteps=2000))
        xw2, λw2 = solw2.x, solw2.λ
        @test vec(trange) == solw2.t
        @test xw2 == xw
        @test λw2 == λw
        @test xw2[1,:] == q0
        @test size(xw) == (length(trange), length(q0))
        @test isapprox(sum(λw2[1,:]), lorenztr) == false
        @test isapprox(sum(λw2[end,:]), lorenztr)
        solz2 = (@test_logs min_level=Logging.Warn lyap_taylorinteg(
            lorenz!, q0, trange[1], trange[end], _order, _abstol; maxsteps=2000))
        tz2, xz2, λz2 = solz2.t, solz2.x, solz2.λ
        @test λw2[end,:] == λz2[end,:]
        @test xw2[end,:] == xz2[end,:]
        @test tz2[end] == trange[end]
        mytol = 1e-4
        @test isapprox(λw2[end,1], 1.47167, rtol=mytol, atol=mytol)
        @test isapprox(λw2[end,2], -0.00831, rtol=mytol, atol=mytol)
        @test isapprox(λw2[end,3], -22.46336, rtol=mytol, atol=mytol)
        solw2_ = (@test_logs min_level=Logging.Warn lyap_taylorinteg(
            lorenz!, q0, vec(trange), _order, _abstol, nothing, lorenz_jac!; maxsteps=2000))
        xw2_, λw2_ = solw2_.x, solw2.λ
        @test xw2 == xw2_
        @test λw2 == λw2_
        solz2_ = (@test_logs min_level=Logging.Warn lyap_taylorinteg(
            lorenz!, q0, trange[1], trange[end], _order, _abstol, nothing, lorenz_jac!; maxsteps=2000))
        tz2_, xz2_, λz2_ = solz2_.t, solz2_.x, solz2.λ
        @test λw2_[end,:] == λz2_[end,:]
        @test xw2_[end,:] == xz2_[end,:]
        @test tz2_[end] == trange[end]

        # Check integration consistency (orbit should not depend on variational eqs)
        sol_ = taylorinteg(lorenz!, q0, vec(trange), _order, _abstol; maxsteps=2000)
        x_ = sol_.x
        @test x_ == xw2

    end

end
