# This file is part of the TaylorIntegration.jl package; MIT licensed

using TaylorIntegration
using Test
using LinearAlgebra: norm
using Elliptic

# Constants for the integrations
const _order = 20
const _abstol = 1.0E-20
const t0 = 0.0
const tf = 1000.0

# Scalar integration
@testset "Scalar case: xdot(t,x) = b-x^2" begin
    b1 = 3.0
    @taylorize xdot1(t, x) = b1-x^2
    @test length(methods(TaylorIntegration.jetcoeffs!)) == 3
    @test (@isdefined xdot1)

    x0 = 1.0
    tv1, xv1 = taylorinteg( xdot1, x0, t0, tf, _order, _abstol, maxsteps=1000,
        parse_eqs=false)
    tv1p, xv1p = taylorinteg( xdot1, x0, t0, tf, _order, _abstol, maxsteps=1000)

    @test length(tv1) == length(tv1p)
    @test iszero( norm(tv1-tv1p, Inf) )
    @test iszero( norm(xv1-xv1p, Inf) )

    # Now using `local` constants
    @taylorize xdot2(t, x) = (local b2 = 3; b2-x^2)
    @test length(methods(TaylorIntegration.jetcoeffs!)) == 4
    @test (@isdefined xdot2)

    tv2, xv2 = taylorinteg( xdot2, x0, t0, tf, _order, _abstol, maxsteps=1000,
        parse_eqs=false)
    tv2p, xv2p = taylorinteg( xdot2, x0, t0, tf, _order, _abstol, maxsteps=1000)

    @test length(tv2) == length(tv2p)
    @test iszero( norm(tv2-tv2p, Inf) )
    @test iszero( norm(xv2-xv2p, Inf) )

    # Comparing integrations
    @test length(tv1p) == length(tv2p)
    @test iszero( norm(tv1p-tv2p, Inf) )
    @test iszero( norm(xv1p-xv2p, Inf) )
end


@testset "Scalar case: xdot(t,x) = -9.81" begin
    xdot1(t, x) = -9.81 + zero(t) # `zero(t)` is needed; cf #20
    @taylorize xdot1_parsed(t, x) = -9.81 + zero(t) # `zero(t)` can be avoided here

    @test (@isdefined xdot1_parsed)

    tv1, xv1   = taylorinteg( xdot1, 10.0, 1.0, tf, _order, _abstol)
    tv1p, xv1p = taylorinteg( xdot1_parsed, 10.0, 1.0, tf, _order, _abstol)

    @test length(tv1) == length(tv1p)
    @test iszero( norm(tv1-tv1p, Inf) )
    @test iszero( norm(xv1-xv1p, Inf) )

    # Now using `local`
    @taylorize function xdot2(t, x)
        local ggrav = -9.81 + zero(t)  # zero(t) is needed for the parse_eqs=false case
        tmp = ggrav  # needed to avoid an error when parsing
    end

    @test (@isdefined xdot2)

    tv2, xv2   = taylorinteg( xdot2, 10.0, 1.0, tf, _order, _abstol, parse_eqs=false)
    tv2p, xv2p = taylorinteg( xdot2, 10.0, 1.0, tf, _order, _abstol)

    @test length(tv2) == length(tv2p)
    @test iszero( norm(tv2-tv2p, Inf) )
    @test iszero( norm(xv2-xv2p, Inf) )

    # Comparing both integrations
    @test length(tv1p) == length(tv2p)
    @test iszero( norm(tv1p-tv2p, Inf) )
    @test iszero( norm(xv1p-xv2p, Inf) )
end


# Pendulum integrtf = 100.0
@testset "Integration of the pendulum" begin
    @taylorize function pendulum!(t, x, dx)
        dx[1] = x[2]
        dx[2] = -sin( x[1] )
        nothing
    end

    @test (@isdefined pendulum!)
    q0 = [pi-0.001, 0.0]
    tv2, xv2 = taylorinteg(pendulum!, q0, t0, tf, _order, _abstol, parse_eqs=false)
    tv2p, xv2p = taylorinteg(pendulum!, q0, t0, tf, _order, _abstol)

    @test length(tv2) == length(tv2p)
    @test iszero( norm(tv2-tv2p, Inf) )
    @test iszero( norm(xv2-xv2p, Inf) )
end


# Complex dependent variables
@testset "Complex dependent variable" begin
    cc = complex(0.0,1.0)
    @taylorize eqscmplx(t,x) = cc*x

    @test (@isdefined eqscmplx)
    cx0 = complex(1.0, 0.0)
    tv1, xv1 = taylorinteg(eqscmplx, cx0, t0, tf, _order, _abstol, maxsteps=1500,
        parse_eqs=false)
    tv1p, xv1p = taylorinteg(eqscmplx, cx0, t0, tf, _order, _abstol, maxsteps=1500)

    @test length(tv1) == length(tv1p)
    @test iszero( norm(tv1-tv1p, Inf) )
    @test iszero( norm(xv1-xv1p, Inf) )

    # Using `local` for the constant value
    @taylorize eqscmplx2(t,x) = (local cc1 = Complex(0.0,1.0); cc1*x)

    @test (@isdefined eqscmplx2)
    tv2, xv2 = taylorinteg(eqscmplx2, cx0, t0, tf, _order, _abstol, maxsteps=1500,
        parse_eqs=false)
    tv2p, xv2p = taylorinteg(eqscmplx2, cx0, t0, tf, _order, _abstol, maxsteps=1500)

    @test length(tv2) == length(tv2p)
    @test iszero( norm(tv2-tv2p, Inf) )
    @test iszero( norm(xv2-xv2p, Inf) )

    # Comparing both integrations
    @test length(tv1p) == length(tv2p)
    @test iszero( norm(tv1p-tv2p, Inf) )
    @test iszero( norm(xv1p-xv2p, Inf) )
end


@testset "Time-dependent integration (with and without `local` vars)" begin
    @taylorize function integ_cos1(t, x)
        y = cos(t)
        return y
    end
    @taylorize function integ_cos2(t, x)
        local y = cos(t)  # allows to calculate directly `cos(t)` *once*
        yy = y   # needed to avoid an error
        return yy
    end

    @test (@isdefined integ_cos1)
    @test (@isdefined integ_cos2)

    tv11, xv11 = taylorinteg(integ_cos1, 0.0, 0.0, pi, _order, _abstol, parse_eqs=false)
    tv12, xv12 = taylorinteg(integ_cos1, 0.0, 0.0, pi, _order, _abstol)
    @test length(tv11) == length(tv12)
    @test iszero( norm(tv11-tv12, Inf) )
    @test iszero( norm(xv11-xv12, Inf) )

    tv21, xv21 = taylorinteg(integ_cos2, 0.0, 0.0, pi, _order, _abstol, parse_eqs=false)
    tv22, xv22 = taylorinteg(integ_cos2, 0.0, 0.0, pi, _order, _abstol)
    @test length(tv21) == length(tv22)
    @test iszero( norm(tv21-tv22, Inf) )
    @test iszero( norm(xv21-xv22, Inf) )

    @test iszero( norm(tv12-tv22, Inf) )
    @test iszero( norm(xv12-xv22, Inf) )
end


# Multiple (3) pendula
@testset "Multiple pendula" begin
    NN = 3
    nnrange = 1:3
    @taylorize function multpendula1!(t, x, dx)
        for i in nnrange
            dx[i] = x[NN+i]
            dx[i+NN] = -sin( x[i] )
        end
        return nothing
    end

    @test (@isdefined multpendula1!)
    q0 = [pi-0.001, 0.0, pi-0.001, 0.0,  pi-0.001, 0.0]
    tv1, xv1 = taylorinteg(multpendula1!, q0, t0, tf, _order, _abstol,
        maxsteps=1000, parse_eqs=false)
    tv1p, xv1p = taylorinteg(multpendula1!, q0, t0, tf, _order, _abstol,
        maxsteps=1000)

    @test length(tv1) == length(tv1p)
    @test iszero( norm(tv1-tv1p, Inf) )
    @test iszero( norm(xv1-xv1p, Inf) )

    @taylorize function multpendula2!(t, x, dx)
        local NN = 3
        local nnrange = 1:NN
        for i in nnrange
            dx[i] = x[NN+i]
            dx[i+NN] = -sin( x[i] )
        end
        return nothing
    end

    @test (@isdefined multpendula2!)
    tv2, xv2 = taylorinteg(multpendula2!, q0, t0, tf, _order, _abstol,
        maxsteps=1000, parse_eqs=false)
    tv2p, xv2p = taylorinteg(multpendula2!, q0, t0, tf, _order, _abstol,
        maxsteps=1000)

    @test length(tv2) == length(tv2p)
    @test iszero( norm(tv2-tv2p, Inf) )
    @test iszero( norm(xv2-xv2p, Inf) )

    # Comparing both integrations
    @test length(tv1) == length(tv2)
    @test iszero( norm(tv1-tv2, Inf) )
    @test iszero( norm(xv1-xv2, Inf) )
end


# Kepler problem
# Redefining some constants
const _order = 28
const tf = 2π*100.0
@testset "Kepler problem (using `^`)" begin
    mμ = -1.0
    @taylorize function kepler1!(t, q, dq)
        r_p3d2 = (q[1]^2+q[2]^2)^1.5

        dq[1] = q[3]
        dq[2] = q[4]
        dq[3] = mμ*q[1]/r_p3d2
        dq[4] = mμ*q[2]/r_p3d2

        return nothing
    end

    NN = 2
    @taylorize function kepler2!(t, q, dq)
        r2 = zero(q[1])
        for i = 1:NN
            r2_aux = r2 + q[i]^2
            r2 = r2_aux
        end
        r_p3d2 = r2^(3/2)
        for j = 1:NN
            dq[j] = q[NN+j]
            dq[NN+j] = mμ*q[j]/r_p3d2
        end

        nothing
    end

    @test (@isdefined kepler1!)
    @test (@isdefined kepler2!)
    q0 = [0.2, 0.0, 0.0, 3.0]
    tv1, xv1 = taylorinteg(kepler1!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    tv1p, xv1p = taylorinteg(kepler1!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)

    tv6p, xv6p = taylorinteg(kepler2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    tv7p, xv7p = taylorinteg(kepler2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)

    @test length(tv1) == length(tv1p) == length(tv6p)
    @test iszero( norm(tv1-tv1p, Inf) )
    @test iszero( norm(xv1-xv1p, Inf) )
    @test iszero( norm(tv1-tv6p, Inf) )
    @test iszero( norm(xv1-xv6p, Inf) )
    @test iszero( norm(tv7p-tv6p, Inf) )
    @test iszero( norm(xv7p-xv6p, Inf) )

    @taylorize function kepler3!(t, q, dq)
        local mμ = -1.0
        r_p3d2 = (q[1]^2+q[2]^2)^1.5

        dq[1] = q[3]
        dq[2] = q[4]
        dq[3] = mμ*q[1]/r_p3d2
        dq[4] = mμ*q[2]/r_p3d2

        return nothing
    end

    @taylorize function kepler4!(t, q, dq)
        local mμ = -1.0
        local NN = 2
        r2 = zero(q[1])
        for i = 1:NN
            r2_aux = r2 + q[i]^2
            r2 = r2_aux
        end
        r_p3d2 = r2^(3/2)
        for j = 1:NN
            dq[j] = q[NN+j]
            dq[NN+j] = mμ*q[j]/r_p3d2
        end

        nothing
    end

    @test (@isdefined kepler3!)
    @test (@isdefined kepler4!)
    tv3, xv3 = taylorinteg(kepler3!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    tv3p, xv3p = taylorinteg(kepler3!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)

    tv4, xv4 = taylorinteg(kepler4!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    tv4p, xv4p = taylorinteg(kepler4!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)

    @test length(tv3) == length(tv3p) == length(tv4)
    @test iszero( norm(tv3-tv3p, Inf) )
    @test iszero( norm(xv3-xv3p, Inf) )
    @test iszero( norm(tv3-tv4, Inf) )
    @test iszero( norm(xv3-xv4, Inf) )
    @test iszero( norm(tv4p-tv4, Inf) )
    @test iszero( norm(xv4p-xv4, Inf) )

    # Comparing both integrations
    @test iszero( norm(tv1p-tv3p, Inf) )
    @test iszero( norm(xv1p-xv3p, Inf) )
end


@testset "Kepler problem (using `sqrt`)" begin
    mμ = -1.0
    @taylorize function kepler1!(t, q, dq)
        r = sqrt(q[1]^2+q[2]^2)
        r_p3d2 = r^3

        dq[1] = q[3]
        dq[2] = q[4]
        dq[3] = mμ*q[1]/r_p3d2
        dq[4] = mμ*q[2]/r_p3d2

        return nothing
    end
    NN = 2
    @taylorize function kepler2!(t, q, dq)
        r2 = zero(q[1])
        for i = 1:NN
            r2_aux = r2 + q[i]^2
            r2 = r2_aux
        end
        r = sqrt(r2)
        r_p3d2 = r^3
        for j = 1:NN
            dq[j] = q[NN+j]
            dq[NN+j] = mμ*q[j]/r_p3d2
        end

        nothing
    end

    @test (@isdefined kepler1!)
    @test (@isdefined kepler2!)
    q0 = [0.2, 0.0, 0.0, 3.0]
    tv1, xv1 = taylorinteg(kepler1!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    tv1p, xv1p = taylorinteg(kepler1!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)

    tv2, xv2 = taylorinteg(kepler2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    tv2p, xv2p = taylorinteg(kepler2!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)

    @test length(tv1) == length(tv1p) == length(tv2)
    @test iszero( norm(tv1-tv1p, Inf) )
    @test iszero( norm(xv1-xv1p, Inf) )
    @test iszero( norm(tv1-tv2, Inf) )
    @test iszero( norm(xv1-xv2, Inf) )
    @test iszero( norm(tv2p-tv2, Inf) )
    @test iszero( norm(xv2p-xv2, Inf) )

    @taylorize function kepler3!(t, q, dq)
        local mμ = -1.0
        r = sqrt(q[1]^2+q[2]^2)
        r_p3d2 = r^3

        dq[1] = q[3]
        dq[2] = q[4]
        dq[3] = mμ*q[1]/r_p3d2
        dq[4] = mμ*q[2]/r_p3d2

        return nothing
    end

    @taylorize function kepler4!(t, q, dq)
        local NN = 2
        local mμ = -1.0
        r2 = zero(q[1])
        for i = 1:NN
            r2_aux = r2 + q[i]^2
            r2 = r2_aux
        end
        r = sqrt(r2)
        r_p3d2 = r^3
        for j = 1:NN
            dq[j] = q[NN+j]
            dq[NN+j] = mμ*q[j]/r_p3d2
        end

        nothing
    end

    @test (@isdefined kepler3!)
    @test (@isdefined kepler4!)
    tv3, xv3 = taylorinteg(kepler3!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    tv3p, xv3p = taylorinteg(kepler3!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)

    tv4, xv4 = taylorinteg(kepler4!, q0, t0, tf, _order, _abstol,
        maxsteps=500000, parse_eqs=false)
    tv4p, xv4p = taylorinteg(kepler4!, q0, t0, tf, _order, _abstol,
        maxsteps=500000)

    @test length(tv3) == length(tv3p) == length(tv4)
    @test iszero( norm(tv3-tv3p, Inf) )
    @test iszero( norm(xv3-xv3p, Inf) )
    @test iszero( norm(tv3-tv4, Inf) )
    @test iszero( norm(xv3-xv4, Inf) )
    @test iszero( norm(tv4p-tv4, Inf) )
    @test iszero( norm(xv4p-xv4, Inf) )

    # Comparing both integrations
    @test iszero( norm(tv1p-tv3p, Inf) )
    @test iszero( norm(xv1p-xv3p, Inf) )
end


const tf = 20.0
@testset "Lyapunov spectrum and `@taylorize`" begin
    #Lorenz system parameters
    σ = 16.0
    β = 4.0
    ρ = 45.92

    #Lorenz system ODE:
    @taylorize function lorenz1!(t, x, dx)
        dx[1] = σ*(x[2]-x[1])
        dx[2] = x[1]*(ρ-x[3])-x[2]
        dx[3] = x[1]*x[2]-β*x[3]
        nothing
    end

    #Lorenz system Jacobian (in-place):
    function lorenz1_jac!(jac, t, x)
        jac[1,1] = -σ+zero(x[1])
        jac[2,1] = ρ-x[3]
        jac[3,1] = x[2]
        jac[1,2] = σ+zero(x[1])
        jac[2,2] = -1.0+zero(x[1])
        jac[3,2] = x[1]
        jac[1,3] = zero(x[1])
        jac[2,3] = -x[1]
        jac[3,3] = -β+zero(x[1])
        nothing
    end

    q0 = [19.0, 20.0, 50.0] #the initial condition
    xi = set_variables("δ", order=1, numvars=length(q0))

    tv1, lv1, xv1 = lyap_taylorinteg(lorenz1!, q0, t0, tf, _order, _abstol,
        maxsteps=2000, parse_eqs=false);

    tv1p, lv1p, xv1p = lyap_taylorinteg(lorenz1!, q0, t0, tf, _order, _abstol,
        maxsteps=2000);

    @test tv1 == tv1p
    @test lv1 == lv1p
    @test xv1 == xv1p

    tv2, lv2, xv2 = lyap_taylorinteg(lorenz1!, q0, t0, tf, _order, _abstol,
        lorenz1_jac!, maxsteps=2000, parse_eqs=false);

    tv2p, lv2p, xv2p = lyap_taylorinteg(lorenz1!, q0, t0, tf, _order, _abstol,
        lorenz1_jac!, maxsteps=2000,  parse_eqs=true);

    @test tv2 == tv2p
    @test lv2 == lv2p
    @test xv2 == xv2p

    # Comparing both integrations (lorenz1)
    @test tv1 == tv2
    @test lv1 == lv2
    @test xv1 == xv2

    #Lorenz system ODE:
    @taylorize function lorenz2!(t, x, dx)
        #Lorenz system parameters
        local σ = 16.0
        local β = 4.0
        local ρ = 45.92

        dx[1] = σ*(x[2]-x[1])
        dx[2] = x[1]*(ρ-x[3])-x[2]
        dx[3] = x[1]*x[2]-β*x[3]
        nothing
    end

    #Lorenz system Jacobian (in-place):
    function lorenz2_jac!(jac, t, x)
        #Lorenz system parameters
        local σ = 16.0
        local β = 4.0
        local ρ = 45.92

        jac[1,1] = -σ+zero(x[1])
        jac[2,1] = ρ-x[3]
        jac[3,1] = x[2]
        jac[1,2] = σ+zero(x[1])
        jac[2,2] = -1.0+zero(x[1])
        jac[3,2] = x[1]
        jac[1,3] = zero(x[1])
        jac[2,3] = -x[1]
        jac[3,3] = -β+zero(x[1])
        nothing
    end

    tv3, lv3, xv3 = lyap_taylorinteg(lorenz2!, q0, t0, tf, _order, _abstol,
        maxsteps=2000, parse_eqs=false);

    tv3p, lv3p, xv3p = lyap_taylorinteg(lorenz2!, q0, t0, tf, _order, _abstol,
        maxsteps=2000);

    @test tv3 == tv3p
    @test lv3 == lv3p
    @test xv3 == xv3p

    tv4, lv4, xv4 = lyap_taylorinteg(lorenz2!, q0, t0, tf, _order, _abstol,
        lorenz2_jac!, maxsteps=2000, parse_eqs=false);

    tv4p, lv4p, xv4p = lyap_taylorinteg(lorenz2!, q0, t0, tf, _order, _abstol,
        lorenz2_jac!, maxsteps=2000,  parse_eqs=true);

    @test tv4 == tv4p
    @test lv4 == lv4p
    @test xv4 == xv4p

    # Comparing both integrations (lorenz2)
    @test tv3 == tv4
    @test lv3 == lv4
    @test xv3 == xv4

    # Comparing both integrations (lorenz1! vs lorenz2!)
    @test tv1 == tv3
    @test lv1 == lv3
    @test xv1 == xv3
end


@testset "Tests for throwing errors" begin
    # Wrong number of arguments
    ex = :(function f_p!(t, x, dx, y)
        dx[1] = x[2]
        dx[2] = -sin( x[1] )
    end)
    @test_throws ArgumentError TaylorIntegration._make_parsed_jetcoeffs(ex)

    # `&&` is not yet implemented
    ex = :(function f_p!(t, x)
        true && x
    end)
    @test_throws ArgumentError TaylorIntegration._make_parsed_jetcoeffs(ex)

    # a is not an Expr; String
    ex = :(function f_p!(t, x)
        "a"
    end)
    @test_throws ArgumentError TaylorIntegration._make_parsed_jetcoeffs(ex)

    # KeyError: key :fname not found
    ex = :(begin
        x=1
        x+x
    end)
    @test_throws KeyError TaylorIntegration._make_parsed_jetcoeffs(ex)

    # BoundsError
    ex = :(function f_p!(t, x)
        local cos(t)
    end)
    @test_throws BoundsError TaylorIntegration._make_parsed_jetcoeffs(ex)
end


@testset "Jet transport with @taylorize macro" begin
    @taylorize function pendulum!(t, x, dx)
        dx[1] = x[2]
        dx[2] = -sin( x[1] )
        nothing
    end

    varorder = 2 #the order of the variational expansion
    p = set_variables("ξ", numvars=2, order=varorder) #TaylorN steup
    q0 = [1.3, 0.0] #the initial conditions
    q0TN = q0 + p #parametrization of a small neighbourhood around the initial conditions
    # T is the librational period == 4Elliptic.K(sin(q0[1]/2)^2)
    T = 4Elliptic.K(sin(q0[1]/2)^2) # equals 7.019250311844546
    integstep = 0.25*T #the time interval between successive evaluations of the solution vector

    #the time range
    tr = t0:integstep:T;
    #note that as called below, taylorinteg uses the parsed jetcoeffs! method by default
    xvp = taylorinteg(pendulum!, q0, tr, _order, _abstol, maxsteps=100)

    # "warmup" for jet transport integration
    xvTN = taylorinteg(pendulum!, q0TN, tr, _order, _abstol, maxsteps=1, parse_eqs=false)
    xvTN = taylorinteg(pendulum!, q0TN, tr, _order, _abstol, maxsteps=1)
    @test size(xvTN) == (5,2)
    #jet transport integration with parsed jetcoeffs!
    xvTNp = taylorinteg(pendulum!, q0TN, tr, _order, _abstol, maxsteps=100)
    #jet transport integration with non-parsed jetcoeffs!
    xvTN = taylorinteg(pendulum!, q0TN, tr, _order, _abstol, maxsteps=100, parse_eqs=false)
    @test xvTN == xvTNp
    @test norm(xvTNp[:,:]() - xvp, Inf) < 1E-15

    dq = 0.0001rand(2)
    q1 = q0 + dq
    yv = taylorinteg(pendulum!, q1, tr, _order, _abstol, maxsteps=100)
    yv_jt = xvTNp[:,:](dq)
    @test norm(yv-yv_jt, Inf) < 1E-11

    dq = 0.001
    t = Taylor1([0.0, 1.0], 10)
    x0T1 = q0+[0t,t]
    q1 = q0+[0.0,dq]
    tv, xv = taylorinteg(pendulum!, q1, t0, 2T, _order, _abstol)
    tvT1, xvT1 = taylorinteg(pendulum!, x0T1, t0, 2T, _order, _abstol, parse_eqs=false)
    tvT1p, xvT1p = taylorinteg(pendulum!, x0T1, t0, 2T, _order, _abstol)
    @test tvT1 == tvT1p
    @test xvT1 == xvT1p
    xv_jt = xvT1p[:,:](dq)
    @test norm(xv_jt[end,:]-xv[end,:]) < 20eps(norm(xv[end,:]))
end